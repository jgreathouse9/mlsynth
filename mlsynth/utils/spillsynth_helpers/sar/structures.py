"""Frozen dataclasses for the SPILLSYNTH ``method='sar'`` subpackage.

Sakaguchi & Tagawa (2026), *Identification and Bayesian Inference for
Synthetic Control Methods with Spillover Effects*. The estimator models the
control outcomes with a spatial-autoregressive (SAR) panel and recovers both
the treatment effect on the treated unit and the spillover effects on the
untreated units, relaxing SUTVA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SARInputs:
    """Preprocessed panel + spatial weights for the SAR spillover SCM.

    Attributes
    ----------
    Y : np.ndarray
        Full outcome panel ``(N+1, T)``; row 0 is the treated unit and rows
        ``1 .. N`` the control units (same order as ``control_labels``).
    Y0 : np.ndarray
        Treated-unit outcomes ``(T,)``.
    Yc : np.ndarray
        Control outcomes ``(T, N)``.
    X : np.ndarray, optional
        Time-varying covariate cube ``(T, N, K)`` for the controls (raw), or
        ``None`` when no covariates are used.
    Wn : np.ndarray
        Row-normalised control-to-control spatial weights ``(N, N)``.
    wn : np.ndarray
        Treated-to-control spatial-weight vector ``(N,)`` (sums to one).
    T0 : int
        Number of pre-treatment periods.
    treated_label : Any
    control_labels : Tuple[Any, ...]
    time_labels : np.ndarray
    covariate_names : Tuple[Any, ...]
    """

    Y: np.ndarray
    Y0: np.ndarray
    Yc: np.ndarray
    Wn: np.ndarray
    wn: np.ndarray
    T0: int
    treated_label: Any
    control_labels: Tuple[Any, ...]
    time_labels: np.ndarray
    X: Optional[np.ndarray] = None
    covariate_names: Tuple[Any, ...] = ()

    @property
    def N(self) -> int:
        return self.Yc.shape[1]

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    @property
    def pre_time(self) -> np.ndarray:
        return self.time_labels[: self.T0]

    @property
    def post_time(self) -> np.ndarray:
        return self.time_labels[self.T0:]


@dataclass(frozen=True)
class SARFit:
    """SAR spillover-SCM fit artifacts (``method='sar'``).

    The accessor names ``att_sp`` / ``gap_sp`` / ``counterfactual_sp`` /
    ``att_scm`` / ``gap_scm`` / ``counterfactual_scm`` / ``spillover_panel``
    mirror the other SPILLSYNTH methods so the top-level
    :class:`~mlsynth.utils.spillsynth_helpers.structures.SpillSynthResults`
    accessors route transparently.

    Attributes
    ----------
    att_sp : float
        Spillover-adjusted ATT on the treated unit (post-period mean), plugged
        at the posterior means ``(alpha_hat, rho_hat)``.
    att_scm : float
        Standard SCM ATT (the ``rho = 0`` special case).
    gap_sp, gap_scm : np.ndarray
        Per-post-period treatment effect, spillover-adjusted and SCM.
    counterfactual_sp, counterfactual_scm : np.ndarray
        Post-period treated counterfactual under each model.
    spillover_panel : dict
        ``{control_label: per-post-period spillover effect}`` (posterior mean).
    ate_ci : tuple of float
        Credible interval for the spillover-adjusted ATT (over the ``rho``
        posterior, ``alpha`` fixed at ``alpha_hat`` -- the paper's convention).
    rho_hat : float
    rho_ci : tuple of float
    rho_draws : np.ndarray
    sigma2_hat : float
    beta_hat : np.ndarray, optional
    alpha_hat : np.ndarray
        Posterior-mean synthetic weights.
    alpha_labels : tuple
        Control labels aligned to ``alpha_hat``.
    acc_rho : float
        Metropolis acceptance rate for ``rho``.
    p_factors : int
    ci_level : float
    """

    att_sp: float
    att_scm: float
    gap_sp: np.ndarray
    gap_scm: np.ndarray
    counterfactual_sp: np.ndarray
    counterfactual_scm: np.ndarray
    spillover_panel: Dict[Any, np.ndarray]
    ate_ci: Tuple[float, float]
    rho_hat: float
    rho_ci: Tuple[float, float]
    rho_draws: np.ndarray
    sigma2_hat: float
    alpha_hat: np.ndarray
    alpha_labels: Tuple[Any, ...]
    acc_rho: float
    p_factors: int
    ci_level: float
    beta_hat: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
