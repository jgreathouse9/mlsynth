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
    batched: bool = Field(
        ..., description=(
            "Whether each cohort was solved as one multiple-right-hand-side "
            "program. False when a donor predicate binds, since that gives "
            "each treated unit its own design matrix."))


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
