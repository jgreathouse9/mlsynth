"""Typed containers for STACKEDSC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...config_models import BaseEstimatorResults


class StackedUnitFit(BaseModel):
    """One treated unit's synthetic control, on its own event clock."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    label: Any = Field(..., description="Treated unit identifier.")
    adoption_time: Any = Field(..., description="First treated period.")
    base_period: Any = Field(
        ..., description="T0i, the period the series are indexed to.")
    horizons: np.ndarray = Field(
        ..., description="Event time e, ascending, including negatives.")
    tau: np.ndarray = Field(
        ..., description="Gap at each horizon; percent of the base level when "
                         "`normalize`, outcome units otherwise.")
    donor_weights: Dict[str, float] = Field(
        ..., description="Weight on each donor, summing to one.")
    agg_weight: float = Field(
        ..., description="This unit's share in the event-time average.")
    pre_rmse: float = Field(
        ..., description="Root mean squared gap over the reported pre-window.")


class StackedEventStudy(BaseModel):
    """The aggregate: a weighted average of the per-unit gaps."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    horizons: np.ndarray = Field(..., description="Event time e, ascending.")
    tau: np.ndarray = Field(..., description="Weighted mean gap at each e.")
    n_units: np.ndarray = Field(
        ..., description="Treated units contributing at each e.")


class StackedDesign(BaseModel):
    """What the fit actually did, as opposed to what was asked for."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cohorts: List[Any] = Field(..., description="Distinct adoption times.")
    n_treated: int = Field(..., description="Treated units fitted.")
    n_donors: int = Field(..., description="Never-treated donors available.")
    backend: str = Field(..., description="Predictor-weight rule used.")
    normalized: bool = Field(
        ..., description="Whether series were indexed to the base period.")
    bias_corrected: bool = Field(
        ..., description="Whether the Abadie-L'Hour correction was applied.")
    shared_donor_pool: bool = Field(
        ..., description=(
            "Whether every treated unit in every cohort faced the same donor "
            "pool. False when a donor predicate binds, since that gives each "
            "treated unit its own design matrix."))


class StackedPlacebo(BaseModel):
    """The permutation distribution and the two statistics read off it.

    ``paths`` holds every placebo gap path, one block per treated unit with a
    row per donor in that unit's pool; ``placebo_averages`` holds the ``S``
    sampled averages built from them. Both are kept because the RMSPE ranking
    and the variance interval are computed from the averages, while the
    individual paths are what a reader plots behind the estimate.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    horizons: np.ndarray = Field(
        ..., description="Event time e, matching the event study.")
    n_samples: int = Field(..., description="S, placebo averages sampled.")
    n_fits: int = Field(
        ..., description="Simplex programs solved to build the placebo paths.")
    donor_pool: str = Field(
        ..., description="Whether the treated unit joined each placebo's pool.")
    placebo_averages: np.ndarray = Field(
        ..., description="(S, n_horizons) sampled placebo average gaps.")
    paths: Dict[str, np.ndarray] = Field(
        ..., description=(
            "Placebo gap paths per treated unit, (n_donors, n_horizons), "
            "keyed by treated unit id as a string."))
    se: np.ndarray = Field(
        ..., description="Placebo standard deviation at each horizon.")
    ci_lower: np.ndarray = Field(
        ..., description="Lower placebo-variance interval by horizon.")
    ci_upper: np.ndarray = Field(
        ..., description="Upper placebo-variance interval by horizon.")
    p_variance: np.ndarray = Field(
        ..., description=(
            "Two-sided normal p-value by horizon, NaN where the placebo "
            "spread is degenerate (as at e = -1, where every gap is zero)."))
    rmspe_actual: Dict[int, float] = Field(
        ..., description=(
            "Post-over-pre mean squared gap ratio for the estimate, keyed by "
            "the last post horizon E it runs through."))
    rmspe_p: Dict[int, float] = Field(
        ..., description="RMSPE-ranked p-value at each E.")
    att_se: float = Field(
        ..., description="Placebo standard deviation of the post-period ATT.")
    att_p_value: float = Field(
        ..., description="Two-sided placebo-variance p-value for the ATT.")
    att_ci_lower: float = Field(..., description="Lower interval for the ATT.")
    att_ci_upper: float = Field(..., description="Upper interval for the ATT.")
    confidence_level: float = Field(
        ..., description="1 - alpha, the interval's nominal coverage.")


class STACKEDSCResults(BaseEstimatorResults):
    """What :meth:`mlsynth.STACKEDSC.fit` returns.

    The standardized sub-models carry the aggregate; the three fields below are
    the estimator-specific outputs. ``per_unit`` matters more here than in most
    estimators: the aggregate is a weighted mean of those paths, and with a
    donor pool small relative to the pre-window the individual weight vectors
    are not identified even where the average is.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    per_unit: Dict[str, StackedUnitFit] = Field(
        default_factory=dict,
        description="Per-treated-unit fits, keyed by unit id as a string.")
    event_study: Optional[StackedEventStudy] = Field(
        default=None, description="Weighted mean gap by event time.")
    design: Optional[StackedDesign] = Field(
        default=None, description="What the fit did, as opposed to what was "
                                  "requested.")
    placebo: Optional[StackedPlacebo] = Field(
        default=None, description=(
            "The sampled-placebo-average permutation distribution and its two "
            "statistics. None unless `inference='placebo'` was requested; the "
            "same object is also reachable as `inference.details`."))
