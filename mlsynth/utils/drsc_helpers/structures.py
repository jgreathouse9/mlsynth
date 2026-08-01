"""Typed containers for DRSC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    InferenceResults,
    MethodDetailsResults,
    WeightsResults,
)


@dataclass(frozen=True)
class DRSCCell:
    """One (unit, time) cell: an intercept-prepended design and its outcomes."""

    X: np.ndarray          # (n_it, k)
    y: np.ndarray          # (n_it,)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class DRSCInputs:
    """Preprocessed micro-panel."""

    cells: Dict[Tuple[Any, Any], DRSCCell]
    unit_names: List[Any]           # index 0 is the treated unit
    covariates: List[str]
    pre_periods: List[Any]
    post_periods: List[Any]
    treated_unit_name: Any
    outcome: str

    @property
    def J(self) -> int:
        """Number of donors."""
        return len(self.unit_names) - 1

    @property
    def donor_names(self) -> List[Any]:
        return self.unit_names[1:]

    @property
    def n_total(self) -> int:
        return int(sum(c.n for c in self.cells.values()))


@dataclass(frozen=True)
class ConditionalEffect:
    """The estimate at one evaluation point, for one post-treatment period.

    Attributes
    ----------
    label : str
        The caller's name for the evaluation point.
    x : np.ndarray
        The design vector, intercept first.
    delta : np.ndarray
        Pointwise CDF difference over the outcome grid (eq. 3).
    f_hat : float
        Integrated squared effect (eq. 2), the scalar summary.
    std_error : float
        Standard error of ``f_hat``.
    lower_bound : float
        One-sided lower confidence bound (eq. 10); 0 when uninformative.
    t_stat : float
        Supremum statistic on the full grid (eq. 14).
    crit_value : float
        Simulated critical value at the configured level.
    p_value : float
        Full-support p-value.
    t_stat_focused, crit_value_focused, p_value_focused, lower_bound_focused
        The same quantities restricted to ``focus_region``; NaN when no
        focus region was given.
    """

    label: str
    x: np.ndarray
    delta: np.ndarray
    f_hat: float
    std_error: float
    lower_bound: float
    t_stat: float
    crit_value: float
    p_value: float
    t_stat_focused: float = float("nan")
    crit_value_focused: float = float("nan")
    p_value_focused: float = float("nan")
    lower_bound_focused: float = float("nan")


class DRSCResults(BaseEstimatorResults):
    """Container returned by :meth:`mlsynth.DRSC.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: the headline scalar is
    lifted into the standardized sub-models so the flat accessors resolve
    through the base contract, while the conditional objects -- which have no
    analogue in a scalar-ATT estimator -- stay on
    :attr:`conditional_effects`.

    ``effects.att`` is the integrated squared effect at the *first* evaluation
    point. That is a summary of convenience: this estimator's output is a
    function of the covariate value, and reading only ``att`` discards what it
    was built to show.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    conditional_effects: Dict[str, ConditionalEffect] = Field(
        default_factory=dict,
        description="Per-evaluation-point estimates, keyed by the caller's label.")
    outcome_grid: Optional[np.ndarray] = Field(
        default=None, description="Outcome grid y_l the DR was fitted on.")
    gram_condition_number: Optional[float] = Field(
        default=None,
        description=("Condition number of the Gram matrix. Large values (~1e5 "
                     "on the paper's panel) are normal and mean the weights "
                     "need float64; see the docs."))
    n_negative_weights: Optional[int] = Field(
        default=None,
        description="Donors receiving negative weight; expected, not a fault.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _populate(self) -> "DRSCResults":
        if self.effects is not None or not self.conditional_effects:
            return self
        first = next(iter(self.conditional_effects.values()))
        object.__setattr__(self, "effects", EffectsResults(
            att=float(first.f_hat), att_std_err=float(first.std_error),
            additional_effects={k: float(v.f_hat)
                                for k, v in self.conditional_effects.items()}))
        object.__setattr__(self, "inference", InferenceResults(
            p_value=float(first.p_value), ci_lower=float(first.lower_bound),
            ci_upper=float("inf"), standard_error=float(first.std_error),
            method="supremum test (Gaussian process)",
            details={k: {"p_value": float(v.p_value),
                         "p_value_focused": float(v.p_value_focused),
                         "t_stat": float(v.t_stat)}
                     for k, v in self.conditional_effects.items()}))
        return self
