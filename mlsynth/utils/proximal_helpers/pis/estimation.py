"""Proximal Inference with Surrogates (PIS) -- full-sample two-stage GMM.

Implements the surrogate estimator of Liu, Tchetgen Tchetgen and Varjao
(2023, arXiv:2308.09527). Stage 1 fits donor coefficients ``alpha`` on the
pre-period; Stage 2 projects the post-period residual onto surrogate
outcomes ``X`` instrumented by surrogate proxies ``Z1``. Closes with the
joint GMM sandwich variance of the ATT (HAC/Bartlett middle).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError
from ..inference import hac


def estimate_pi_surrogate(
    outcome_vector: np.ndarray,
    design_matrix_main: np.ndarray,
    instrument_matrix_main: np.ndarray,
    instrument_matrix_surrogate: np.ndarray,
    surrogate_outcome_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_effect_eval: int,
    total_periods: int,
    hac_truncation_lag: int,
    aux_covariates_main_1: Optional[np.ndarray] = None,
    aux_covariates_main_2: Optional[np.ndarray] = None,
    aux_covariates_surrogate: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Proximal Inference with surrogates (PIS).

    Stage 1 fits donor coefficients ``alpha`` on the pre-period
    (``Z0' (Y - W alpha) = 0``). Stage 2 projects the post-period
    residual onto surrogate outcomes ``X`` instrumented by surrogate
    proxies ``Z1`` (``Z1' (Y - W alpha - X gamma) = 0``); ``X gamma`` is
    the time-varying effect. The ATT SE is the joint GMM sandwich variance.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome, shape ``(total_periods,)``.
    design_matrix_main : np.ndarray
        Donor outcomes ``W``.
    instrument_matrix_main : np.ndarray
        Donor proxies ``Z0`` (instruments for ``W``).
    instrument_matrix_surrogate : np.ndarray
        Surrogate proxies ``Z1`` (instruments for ``X``).
    surrogate_outcome_matrix : np.ndarray
        Surrogate outcomes ``X``.
    num_pre_treatment_periods, num_post_periods_for_effect_eval, total_periods : int
        Pre/post/total period counts.
    hac_truncation_lag : int
        Bartlett bandwidth.
    aux_covariates_main_1, aux_covariates_main_2 : np.ndarray, optional
        Optional covariates augmenting ``W`` and ``Z0``.
    aux_covariates_surrogate : np.ndarray, optional
        Optional covariates augmenting ``X`` and ``Z1``.

    Returns
    -------
    tau : float
        ATT (mean post-period time-varying effect).
    taut : np.ndarray
        Time-varying effect over all periods (pre-period entries are the
        Stage-1 residuals), shape ``(total_periods,)``.
    alpha : np.ndarray
        Donor coefficients (original donors only).
    se_tau : float
        Standard error of the ATT (``np.nan`` if GMM inference fails).
    """

    W_aug, Z0_aug = design_matrix_main, instrument_matrix_main
    X_aug, Z1_aug = surrogate_outcome_matrix, instrument_matrix_surrogate
    if (
        aux_covariates_main_1 is not None
        and aux_covariates_main_2 is not None
        and aux_covariates_surrogate is not None
    ):
        Z0_aug = np.column_stack((instrument_matrix_main, aux_covariates_main_2, aux_covariates_main_1))
        W_aug = np.column_stack((design_matrix_main, aux_covariates_main_2, aux_covariates_main_1))
        Z1_aug = np.column_stack((instrument_matrix_surrogate, aux_covariates_surrogate))
        X_aug = np.column_stack((surrogate_outcome_matrix, aux_covariates_surrogate))

    if not (W_aug.shape[1] == Z0_aug.shape[1] and X_aug.shape[1] == Z1_aug.shape[1]):
        raise MlsynthConfigError(
            "Dimension mismatch after augmentation for main or surrogate matrices."
        )

    # Stage 1: donor coefficients on the pre-period.
    Z0W_pre = Z0_aug[:num_pre_treatment_periods].T @ W_aug[:num_pre_treatment_periods]
    Z0Y_pre = Z0_aug[:num_pre_treatment_periods].T @ outcome_vector[:num_pre_treatment_periods]
    alpha_aug = np.linalg.solve(Z0W_pre, Z0Y_pre)

    # Stage 2: project post-period residual onto surrogate outcomes.
    tau_hat_post = outcome_vector[num_pre_treatment_periods:] - W_aug[num_pre_treatment_periods:] @ alpha_aug
    Z1X_post = Z1_aug[num_pre_treatment_periods:].T @ X_aug[num_pre_treatment_periods:]
    Z1tau_post = Z1_aug[num_pre_treatment_periods:].T @ tau_hat_post
    gamma = np.linalg.solve(Z1X_post, Z1tau_post)

    taut = X_aug @ gamma
    taut[:num_pre_treatment_periods] = (outcome_vector - W_aug @ alpha_aug)[:num_pre_treatment_periods]
    tau = float(np.mean(taut[num_pre_treatment_periods : num_pre_treatment_periods + num_post_periods_for_effect_eval]))

    # Stacked moment conditions U0 (Stage 1), U1 (Stage 2), U2 (ATT).
    U0 = Z0_aug.T * (outcome_vector - W_aug @ alpha_aug).reshape(1, -1)
    U0[:, num_pre_treatment_periods:] = 0
    U1 = Z1_aug.T * (outcome_vector - W_aug @ alpha_aug - X_aug @ gamma).reshape(1, -1)
    U1[:, :num_pre_treatment_periods] = 0
    U2 = X_aug @ gamma - tau
    U2[:num_pre_treatment_periods] = 0
    U = np.column_stack((U0.T, U1.T, U2))

    dimZ0, dimZ1 = Z0_aug.shape[1], Z1_aug.shape[1]
    dimW, dimX = W_aug.shape[1], X_aug.shape[1]
    # All Jacobian blocks scaled by total_periods.
    G = np.zeros((dimZ0 + dimZ1 + 1, dimW + dimX + 1))
    G[:dimZ0, :dimW] = Z0W_pre / total_periods
    G[dimZ0 : dimZ0 + dimZ1, :dimW] = (Z1_aug[num_pre_treatment_periods:].T @ W_aug[num_pre_treatment_periods:]) / total_periods
    G[dimZ0 : dimZ0 + dimZ1, dimW : dimW + dimX] = (Z1_aug[num_pre_treatment_periods:].T @ X_aug[num_pre_treatment_periods:]) / total_periods
    G[-1, dimW : dimW + dimX] = -np.sum(X_aug[num_pre_treatment_periods:], axis=0) / total_periods
    G[-1, -1] = (total_periods - num_pre_treatment_periods) / total_periods

    omega = hac(U, hac_truncation_lag)
    try:
        G_inv = np.linalg.inv(G)
        cov = G_inv @ omega @ G_inv.T
        var_tau = cov[-1, -1] / total_periods
        se_tau = float(np.sqrt(var_tau)) if var_tau >= 0 else np.nan
    except np.linalg.LinAlgError:
        se_tau = np.nan

    alpha_original = alpha_aug[: design_matrix_main.shape[1]]
    return tau, taut, alpha_original, se_tau
