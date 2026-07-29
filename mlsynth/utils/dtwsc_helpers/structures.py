"""Typed containers for the DTWSC estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class DTWSCInputs:
    """Preprocessed panel for DTWSC.

    Attributes
    ----------
    y : np.ndarray
        Treated unit's outcome path, shape ``(T,)``.
    donor_matrix : np.ndarray
        Donor outcomes, shape ``(T, J)``.
    donor_names : list
    time_labels : np.ndarray
    n_pre : int
        Number of pre-treatment periods.
    treated_name : Any
    """

    y: np.ndarray
    donor_matrix: np.ndarray
    donor_names: List[Any]
    time_labels: np.ndarray
    n_pre: int
    treated_name: Any

    @property
    def T(self) -> int:
        return int(self.y.size)

    @property
    def J(self) -> int:
        return int(self.donor_matrix.shape[1])

    @property
    def n_post(self) -> int:
        return self.T - self.n_pre


class DTWSCResults(BaseEstimatorResults):
    """Container returned by :meth:`mlsynth.DTWSC.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: alongside the
    DTWSC-specific fields below it exposes the standardized sub-models
    (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``pre_rmse`` / ``donor_weights``.

    Parameters
    ----------
    inputs : DTWSCInputs
    warped_donor_matrix : np.ndarray
        Donor outcomes after resampling onto the treated unit's clock, shape
        ``(T, J)``. ``NaN`` in the tail of a column means the warp compressed
        that donor past the end of the panel -- the reference behaves the same
        way, and those cells are genuinely unobserved rather than zero.
    cutoffs : dict
        ``{donor: period}`` -- the donor period its own pre-treatment window
        ends on, once its speed is accounted for. A cutoff above the treatment
        index means the donor was running slow.
    pre_period_speeds : dict
        ``{donor: array}`` -- learned speed per donor period up to the cutoff.
        Above one means the donor moves slower than the treated unit and is
        stretched onto its clock; below one, the reverse.
    post_period_speeds : dict
        ``{donor: array}`` -- speeds carried into the post-period by the
        second-phase window search.
    warp_applied : bool
        False when ``warp=False``, i.e. the result is the standard-SC baseline.
    placebo_band : dict, optional
        With ``inference="placebo"``: ``lower`` / ``upper`` are the pointwise
        quantiles of the placebo gaps for the warped fit, ``lower_unwarped`` /
        ``upper_unwarped`` the same for the unwarped fit. The paper's claim is
        that warping narrows the band, so both are kept.
    placebo_gaps : dict, optional
        ``{donor: gap path}`` from the unperturbed pool -- the individual
        placebo lines behind the band.
    efficiency_test : dict, optional
        The paper's t-test on ``log(MSE_DSC / MSE_SC)`` with ICC-corrected
        degrees of freedom: ``t``, ``p``, ``df``, ``icc``, ``vif``,
        ``mean_log_ratio``, ``mse_reduction``, ``n_runs``, ``n_pools``. A
        negative ``t`` means warping achieved the lower post-treatment MSE.
    metadata : dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: DTWSCInputs
    warped_donor_matrix: np.ndarray
    cutoffs: Dict[Any, int]
    pre_period_speeds: Dict[Any, Any]
    post_period_speeds: Dict[Any, Any]
    warp_applied: bool = True
    placebo_band: Optional[Dict[str, Any]] = None
    placebo_gaps: Optional[Dict[str, Any]] = None
    efficiency_test: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)
