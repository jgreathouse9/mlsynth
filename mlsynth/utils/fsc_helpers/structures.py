"""Typed containers for FSC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class FSCInputs:
    """The ``(N, T, M)`` cube of embedded outcomes, treated unit first."""

    values: np.ndarray                 # (N, T, M)
    grid: np.ndarray                   # (M,) argument values, ascending
    unit_names: List[Any]              # index 0 is the treated unit
    time_labels: List[Any]
    pre_periods: int
    outcome: str
    argument: str

    @property
    def n_donors(self) -> int:
        return int(self.values.shape[0]) - 1

    @property
    def donor_names(self) -> List[Any]:
        return list(self.unit_names[1:])

    @property
    def post_periods(self) -> int:
        return int(self.values.shape[1]) - self.pre_periods

    @property
    def treated_unit_name(self) -> Any:
        return self.unit_names[0]


@dataclass(frozen=True)
class FSCCurve:
    """One period's observed object, its counterfactual, and their difference.

    Attributes
    ----------
    time : Any
        The period label.
    is_post : bool
        Whether the period is post-treatment.
    observed, counterfactual, effect : numpy.ndarray
        Values on the argument grid. ``effect`` is ``observed - counterfactual``,
        the causal effect of section 2.1 read in the Hilbert space.
    magnitude : float
        ``||effect||_H``, the paper's ``d_t`` -- the length of the geodesic
        between the treated and counterfactual objects.
    lower, upper : numpy.ndarray or None
        Pointwise conformal band on the counterfactual, absent unless a band
        was requested.
    """

    time: Any
    is_post: bool
    observed: np.ndarray
    counterfactual: np.ndarray
    effect: np.ndarray
    magnitude: float
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None


class FSCResults(BaseEstimatorResults):
    """Container returned by :meth:`mlsynth.FSC.fit`.

    An :class:`~mlsynth.config_models.EffectResult`. The estimand here is a
    curve per period, so the standardized sub-models carry a scalar summary and
    the objects themselves live on :attr:`curves`.

    The summary is the grid average of each object. That choice is deliberate:
    averaging is linear, so ``estimated_gap`` on :attr:`time_series` really is
    ``observed_outcome - counterfactual_outcome`` and the usual accessors mean
    what they say. It is also interpretable in the applications -- the grid
    average of an age-specific fertility curve is proportional to the total
    fertility rate, and of a quantile function is the distribution's mean.
    ``effects.att`` is that summary averaged over the post-treatment periods.

    The magnitude ``d_t = ||Y_1t - Y^N_1t||_H`` is reported alongside on each
    curve and, averaged over post-periods, as
    ``effects.additional_effects['mean_effect_magnitude']``. It is non-negative
    by construction, which is why it is not the headline number.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    curves: List[FSCCurve] = Field(
        default_factory=list,
        description="Per-period observed / counterfactual / effect objects.")
    grid: Optional[np.ndarray] = Field(
        default=None, description="Argument grid the objects are sampled on.")
    fsc_weights: Optional[np.ndarray] = Field(
        default=None,
        description="Simplex weights from eq. (1), in donor order.")
    augmented_weights: Optional[np.ndarray] = Field(
        default=None,
        description=("Ridge augmented weights from eq. (9); None when "
                     "augment=False. They sum to one but may be negative."))
    pre_treatment_fit: Optional[float] = Field(
        default=None,
        description=("Pre-treatment fit of the reported estimator, "
                     "sqrt(sum_t ||Y_1t - sum_i w_i Y_it||^2)."))
    pre_treatment_fit_fsc: Optional[float] = Field(
        default=None,
        description="The same quantity for the unaugmented FSC weights.")
    ridge_lambda: Optional[float] = Field(
        default=None, description="Penalty used in eq. (9).")
    ridge_lambda_relative: Optional[float] = Field(
        default=None,
        description=("The cross-validated penalty as a multiple of the "
                     "design's own scale (the mean eigenvalue of the Gram "
                     "matrix). This is the quantity `lambda_bounds` searches "
                     "over, and it is comparable across datasets in a way the "
                     "absolute penalty is not. None when the penalty was "
                     "supplied rather than selected."))
    lambda_selected_by_cv: Optional[bool] = Field(
        default=None,
        description="Whether `ridge_lambda` came from cross-validation.")
    placebo_p_values: Optional[np.ndarray] = Field(
        default=None,
        description=("Per-post-period placebo p-values, aligned to the "
                     "post-treatment periods. Rank-limited: with N units the "
                     "smallest attainable value is 1/N."))
    n_negative_weights: Optional[int] = Field(
        default=None,
        description=("Donors given negative weight by the augmentation -- "
                     "extrapolation outside the donor hull, expected when the "
                     "simplex fit is imperfect, not a fault."))
    metadata: Dict[str, Any] = Field(default_factory=dict)
