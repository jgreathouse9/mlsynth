"""Structured containers for the Synthetic Business Cycle (SBC) estimator.

Implements the containers used by

    Shi, Z., Xi, J., & Xie, H. (2025). "A Synthetic Business Cycle Approach
    to Counterfactual Analysis with Nonstationary Macroeconomic Data."
    arXiv:2505.22388.

The SBC procedure (Section 3.1) decomposes each unit's outcome into a
trend (forecast from its own AR(p) past values at horizon h) and a cycle
(residual). The treated trend is extrapolated from its own history; the
treated cycle is imputed by a standard SCM on the donor pool's cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SBCInputs:
    """Pre-processed panel data for SBC.

    Parameters
    ----------
    Y_full : np.ndarray
        Full outcome matrix of shape ``(T, N)`` with the treated unit as
        column 0 (rows = time, columns = units). This matches the
        ``datautils.dataprep`` convention.
    T : int
        Total number of time periods.
    T0 : int
        Number of pre-treatment periods.
    N : int
        Total number of units (target + donors).
    treated_unit_name : str
        Identifier of the treated unit.
    donor_names : Sequence
        Identifiers for the donor units in column order.
    time_labels : np.ndarray
        Time labels in original order, length ``T``.
    Ywide : object
        Wide pandas frame from ``dataprep`` (rows = time, cols = unit).
    y_target : np.ndarray
        Observed treated unit series, length ``T``.
    """

    Y_full: np.ndarray
    T: int
    T0: int
    N: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray
    Ywide: object
    y_target: np.ndarray


@dataclass(frozen=True)
class HamiltonFit:
    """Per-unit Hamilton filter fit.

    Parameters
    ----------
    coefficients : np.ndarray
        Length-``p + 1`` regression coefficients ``(alpha_0, alpha_1, ..., alpha_p)``
        from Eq. (2) of the paper:
        ``tau_t = alpha_0 + alpha_1 * Y_{t-h} + ... + alpha_p * Y_{t-h-p+1}``.
    trend_pre : np.ndarray
        Length-``T0`` fitted trend over the pre-treatment window. Entries
        for ``t < h + p - 1`` are ``np.nan`` (the projection requires
        ``p`` lags shifted back by ``h``).
    cycle_pre : np.ndarray
        Length-``T0`` cyclical residual ``Y_pre - trend_pre``. NaN where
        ``trend_pre`` is NaN.
    h : int
        Forecasting horizon used.
    p : int
        Number of lags used.
    """

    coefficients: np.ndarray
    trend_pre: np.ndarray
    cycle_pre: np.ndarray
    h: int
    p: int


@dataclass(frozen=True)
class SBCDesign:
    """SBC fitted design.

    Parameters
    ----------
    weights : np.ndarray
        Length ``N - 1`` synthetic control weights on the donor cycles,
        from Eq. (3) of the paper.
    weights_mode : str
        ``"simplex"`` (non-negative, sum to 1) or ``"unrestricted"``
        (intercept + free coefficients, vertical-regression style).
    intercept : float or None
        Fitted intercept; ``None`` when ``weights_mode == "simplex"``.
    treated_hamilton : HamiltonFit
        Hamilton fit for the treated unit (column 0).
    donor_hamiltons : list of HamiltonFit
        Hamilton fits for the donor units (columns 1..N-1).
    trend_forecast : np.ndarray
        Treated trend projected forward over post-treatment periods,
        length ``T - T0``. Computed by applying the treated AR
        coefficients to its own lags (Step 2 of the paper).
    cycle_forecast : np.ndarray
        Synthetic cycle for the treated unit over post-treatment periods,
        length ``T - T0``. Equals ``donor_cycles[:, post] @ weights``
        (Step 3 of the paper).
    counterfactual_post : np.ndarray
        Combined post-treatment counterfactual ``trend_forecast +
        cycle_forecast``, length ``T - T0``. This is the SBC estimate of
        ``Y_{1, t}(0)`` for ``t > T_0``.
    pre_cycle_rmse : float
        Pre-treatment RMSE of the SC fit on the cycle (``c_{1,t} - sum
        w_i c_{i,t}``) over the effective training window.
    """

    weights: np.ndarray
    weights_mode: str
    intercept: Optional[float]
    treated_hamilton: HamiltonFit
    donor_hamiltons: list
    trend_forecast: np.ndarray
    cycle_forecast: np.ndarray
    counterfactual_post: np.ndarray
    pre_cycle_rmse: float


class SBCResults(BaseEstimatorResults):
    """Public ``SBC.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    it populates the standardized sub-models (``effects``, ``time_series``,
    ``weights``, ``fit_diagnostics``, ``method_details``) so the flat accessors
    ``att`` / ``counterfactual`` / ``gap`` / ``donor_weights`` / ``pre_rmse``
    resolve through the base contract. The SBC-specific fields below carry the
    decomposition detail.

    Parameters
    ----------
    inputs : SBCInputs
        Pre-processed panel data.
    design : SBCDesign
        Hamilton fits, donor weights, and post-treatment counterfactual.
    counterfactual_full : np.ndarray
        Length-``T`` series. Pre-treatment entries equal the observed
        ``y_target``; post-treatment entries equal ``design.counterfactual_post``.
        (Mirrored into ``res.counterfactual`` via the contract.)
    treatment_effect : np.ndarray
        Length-``T`` series of ``y_target - counterfactual_full`` (mirrored
        into ``res.gap``).
    weights_by_donor : dict
        Mapping ``donor_label -> weight`` (mirrored into ``res.donor_weights``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SBCInputs
    design: SBCDesign
    counterfactual_full: np.ndarray
    treatment_effect: np.ndarray
    weights_by_donor: dict


# Resolve string annotations (module uses ``from __future__ import annotations``).
SBCResults.model_rebuild()
