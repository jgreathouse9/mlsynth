"""Typed result containers for Partially Pooled SCM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PPSCMInputs:
    """Pre-processed staggered-adoption panel for PPSCM.

    Parameters
    ----------
    Y_treated_pre : np.ndarray
        Pre-treatment outcomes for treated units, shape ``(L, J)``,
        column ``j`` corresponds to treated unit ``j`` at offsets
        ``T_j - L, ..., T_j - 1``.
    Y_donors_pre : np.ndarray
        Donor pre-treatment outcomes stacked across treated units,
        shape ``(L, N, J)``: ``Y_donors_pre[:, :, j]`` is the
        ``(L, N)`` slice of donor outcomes at the ``L`` periods
        immediately preceding treated unit ``j``'s adoption.
    Y_treated_post : np.ndarray
        Post-treatment outcomes for each treated unit, shape ``(K+1, J)``,
        rows aligned to event times ``0, 1, ..., K`` with
        ``K = min_j (T - T_j)``.
    Y_donors_post : np.ndarray
        Donor outcomes at the same event times, shape ``(K+1, N, J)``.
    L : int
        Number of pre-treatment lags used for each treated unit.
    K : int
        Maximum event-time horizon.
    J : int
        Number of treated units.
    N : int
        Number of donor units.
    treated_unit_names : Sequence
        Labels of the treated units in column order.
    donor_names : Sequence
        Labels of the donor units in column order.
    adoption_periods : np.ndarray
        Length-``J`` array of (1-based) adoption-period indices.
    time_labels : np.ndarray
        Time labels of the full panel.
    Ywide : Any
        Wide outcome frame produced by ``dataprep`` (rows = time,
        columns = unit), preserved for plotting.
    outcome : str
        Outcome variable name.
    """

    Y_treated_pre: np.ndarray
    Y_donors_pre: np.ndarray
    Y_treated_post: np.ndarray
    Y_donors_post: np.ndarray
    L: int
    K: int
    J: int
    N: int
    treated_unit_names: Sequence
    donor_names: Sequence
    adoption_periods: np.ndarray
    time_labels: np.ndarray
    Ywide: Any
    outcome: str


@dataclass(frozen=True)
class PPSCMDesign:
    """Output of the partially-pooled QP.

    Parameters
    ----------
    Gamma : np.ndarray
        Optimal weight matrix, shape ``(N, J)``. Each column is on the
        simplex (sum to 1, non-negative).
    nu_used : float
        Value of ``nu`` actually used (resolved from ``"auto"`` if
        requested).
    lam : float
        Frobenius-norm regularization applied.
    q_sep : float
        Per-unit imbalance ``q_sep(Gamma)`` at the chosen solution.
    q_pool : float
        Pooled imbalance ``q_pool(Gamma)`` at the chosen solution.
    q_sep_baseline : float
        Imbalance at the ``nu = 0`` (separate-SCM) baseline; used as
        the normalization for ``q_tilde_sep``.
    q_pool_baseline : float
        Imbalance at the ``nu = 0`` baseline; used as the normalization
        for ``q_tilde_pool``.
    frontier : Dict[float, Tuple[float, float]]
        When ``nu`` was auto-selected, the dict maps each swept ``nu``
        to its ``(q_sep, q_pool)`` pair. Empty when ``nu`` was supplied
        explicitly.
    solver_status : str
        cvxpy solver status string from the final fit.
    """

    Gamma: np.ndarray
    nu_used: float
    lam: float
    q_sep: float
    q_pool: float
    q_sep_baseline: float
    q_pool_baseline: float
    frontier: Dict[float, Tuple[float, float]]
    solver_status: str


@dataclass(frozen=True)
class PPSCMEventStudy:
    """Per-horizon ATT trajectory and its jackknife inference.

    Parameters
    ----------
    horizons : np.ndarray
        Length ``K+1`` array of event-time horizons ``0, 1, ..., K``.
    tau : np.ndarray
        ATT estimates ``ATT_k = (1/J) sum_j tau_{j, k}`` aligned with
        ``horizons``.
    se : np.ndarray
        Jackknife standard errors aligned with ``tau``.
    ci : np.ndarray
        Wald confidence band of shape ``(K+1, 2)``.
    """

    horizons: np.ndarray
    tau: np.ndarray
    se: np.ndarray
    ci: np.ndarray


@dataclass(frozen=True)
class PPSCMInference:
    """Jackknife inference summary for the overall ATT.

    Parameters
    ----------
    att : float
        Overall ATT averaged across treated units and event times.
    se : float
        Jackknife standard error.
    ci : Tuple[float, float]
        Wald 95-percent confidence interval at significance
        ``alpha`` (default 0.05).
    method : str
        ``"jackknife"`` or ``"none"``.
    """

    att: float
    se: float
    ci: Tuple[float, float]
    method: str


@dataclass(frozen=True)
class PPSCMResults:
    """Public ``PPSCM.fit()`` return container.

    Parameters
    ----------
    inputs : PPSCMInputs
        Pre-processed panel.
    design : PPSCMDesign
        Weight matrix, ``nu_used``, and imbalance diagnostics.
    event_study : PPSCMEventStudy
        ATT trajectory with jackknife CIs.
    inference : PPSCMInference
        Overall ATT, jackknife SE, CI.
    pre_rmse : float
        Sample RMSE of the per-treated-unit pre-treatment fit (this is
        ``q_sep`` reported on the raw outcome scale, distinct from the
        normalized ``q_tilde_sep`` used internally).
    donor_weights : Dict[Any, Dict[Any, float]]
        ``{treated_unit_name: {donor_name: gamma_ij}}`` for downstream
        inspection.
    demean : bool
        Whether outcomes were demeaned before fitting (paper's
        "intercept shift" extension).
    """

    inputs: PPSCMInputs
    design: PPSCMDesign
    event_study: PPSCMEventStudy
    inference: PPSCMInference
    pre_rmse: float
    donor_weights: Dict[Any, Dict[Any, float]]
    demean: bool
