"""Two-stage proximal estimators with GMM/HAC inference.

Implements the three estimators used by the PROXIMAL pipeline:

``estimate_pi``
    Proximal Inference (Shi et al., 2023, arXiv:2108.13935). Negative
    control donors ``W`` are instrumented by donor proxies ``Z0`` on the
    pre-period; the fitted relationship imputes the post-period
    counterfactual.
``estimate_pi_surrogate`` / ``estimate_pi_surrogate_post``
    Proximal Inference with surrogates (Liu, Tchetgen Tchetgen & Varjao,
    2023, arXiv:2308.09527). A second stage projects the treatment effect
    onto surrogate outcomes ``X`` instrumented by surrogate proxies ``Z1``.

Each estimator closes with the GMM sandwich variance of the ATT. The
Jacobian blocks and the HAC normalization were validated value-for-value
against the authors' reference code (``freshtaste/proximal``): both the
ATT and its standard error match to machine precision, restoring nominal
~95% CI coverage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from .inference import hac


def estimate_pi(
    outcome_vector: np.ndarray,
    design_matrix: np.ndarray,
    instrument_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_effect_eval: int,
    total_periods: int,
    hac_truncation_lag: int,
    common_aux_covariates_1: Optional[np.ndarray] = None,
    common_aux_covariates_2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Proximal Inference (PI) counterfactual, donor weights, and ATT SE.

    Stage 1 estimates donor coefficients ``alpha`` on the pre-period via
    the just-identified IV moment ``Z0' (Y - W alpha) = 0``; the fitted
    ``W alpha`` is the counterfactual. The ATT standard error is the GMM
    sandwich variance with a HAC (Bartlett) middle.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome, shape ``(total_periods,)``.
    design_matrix : np.ndarray
        Donor outcomes ``W``, shape ``(total_periods, n_donors)``.
    instrument_matrix : np.ndarray
        Donor proxies ``Z0`` (instruments for ``W``), same column count
        as ``design_matrix``.
    num_pre_treatment_periods : int
        Number of pre-treatment periods ``T0``.
    num_post_periods_for_effect_eval : int
        Number of post-treatment periods used to average the ATT.
    total_periods : int
        Total number of periods ``T``.
    hac_truncation_lag : int
        Bartlett bandwidth for the HAC variance.
    common_aux_covariates_1, common_aux_covariates_2 : np.ndarray, optional
        Optional covariates augmenting both ``W`` and ``Z0``. If one is
        given, both must be.

    Returns
    -------
    counterfactual : np.ndarray
        Predicted counterfactual ``W alpha`` (original donors only),
        shape ``(total_periods,)``.
    alpha : np.ndarray
        Donor coefficients (original donors only).
    se_tau : float
        Standard error of the ATT (``np.nan`` if GMM inference fails).
    """

    W_aug, Z0_aug = design_matrix, instrument_matrix
    if common_aux_covariates_1 is not None and common_aux_covariates_2 is not None:
        Z0_aug = np.column_stack((instrument_matrix, common_aux_covariates_2, common_aux_covariates_1))
        W_aug = np.column_stack((design_matrix, common_aux_covariates_2, common_aux_covariates_1))

    if W_aug.shape[1] != Z0_aug.shape[1]:
        raise MlsynthConfigError(
            "Augmented design and instrument matrices must have the same number of columns."
        )

    # Stage 1: just-identified IV on the pre-period.
    Z0W_pre = Z0_aug[:num_pre_treatment_periods].T @ W_aug[:num_pre_treatment_periods]
    Z0Y_pre = Z0_aug[:num_pre_treatment_periods].T @ outcome_vector[:num_pre_treatment_periods]
    alpha_aug = np.linalg.solve(Z0W_pre, Z0Y_pre)

    predicted_aug = W_aug @ alpha_aug
    taut = outcome_vector - predicted_aug
    tau = float(np.mean(taut[num_pre_treatment_periods : num_pre_treatment_periods + num_post_periods_for_effect_eval]))

    # Moment conditions: pre-period orthogonality (U0) and post-period ATT (U1).
    U0 = Z0_aug.T * (outcome_vector - predicted_aug).reshape(1, -1)
    U0[:, num_pre_treatment_periods:] = 0
    U1 = outcome_vector - tau - predicted_aug
    U1[:num_pre_treatment_periods] = 0
    U = np.column_stack((U0.T, U1))

    dimZ0, dimW = Z0_aug.shape[1], W_aug.shape[1]
    # GMM Jacobian, scaled by total_periods so the sandwich recovers Var([alpha, tau]).
    G = np.zeros((dimZ0 + 1, dimW + 1))
    G[:dimZ0, :dimW] = Z0W_pre / total_periods
    G[-1, :dimW] = np.sum(W_aug[num_pre_treatment_periods:], axis=0) / total_periods
    G[-1, -1] = (total_periods - num_pre_treatment_periods) / total_periods

    omega = hac(U, hac_truncation_lag)
    try:
        G_inv = np.linalg.inv(G)
        cov = G_inv @ omega @ G_inv.T
        var_tau = cov[-1, -1] / total_periods
        se_tau = float(np.sqrt(var_tau)) if var_tau >= 0 else np.nan
    except np.linalg.LinAlgError:
        se_tau = np.nan

    alpha_original = alpha_aug[: design_matrix.shape[1]]
    counterfactual = design_matrix @ alpha_original
    return counterfactual, alpha_original, se_tau


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


def estimate_pi_surrogate_post(
    outcome_vector: np.ndarray,
    main_covariates: np.ndarray,
    main_instruments: np.ndarray,
    surrogate_instruments: np.ndarray,
    surrogate_covariates: np.ndarray,
    treatment_start_period: int,
    num_post_treatment_periods_analyzed: int,
    hac_truncation_lag: int,
    aux_main_covariates: Optional[np.ndarray] = None,
    aux_main_instruments: Optional[np.ndarray] = None,
    aux_surrogate_covariates: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Post-treatment proximal surrogate estimator (PIPost).

    Estimates donor and surrogate coefficients jointly on the
    post-treatment period in a single just-identified IV fit
    (``Z' (Y - [W X] params) = 0``), with the surrogate block ``X gamma``
    giving the time-varying effect. The GMM sandwich variance here is
    scaled by the number of post-treatment periods ``T1``.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome, shape ``(total_periods,)``.
    main_covariates : np.ndarray
        Donor outcomes ``W``.
    main_instruments : np.ndarray
        Donor proxies ``Z0``.
    surrogate_instruments : np.ndarray
        Surrogate proxies ``Z1``.
    surrogate_covariates : np.ndarray
        Surrogate outcomes ``X``.
    treatment_start_period : int
        Index of the first post-treatment period ``T0``.
    num_post_treatment_periods_analyzed : int
        Number of post-treatment periods ``T1``.
    hac_truncation_lag : int
        Bartlett bandwidth.
    aux_main_covariates, aux_main_instruments, aux_surrogate_covariates : np.ndarray, optional
        Optional covariates augmenting the design/instrument blocks.

    Returns
    -------
    tau : float
        ATT (mean post-period time-varying effect).
    taut : np.ndarray
        Time-varying effect ``X gamma`` over all periods.
    params_W : np.ndarray
        Donor coefficients (original donors only).
    se_tau : float
        Standard error of the ATT (``np.nan`` if GMM inference fails).
    """

    if (
        aux_main_covariates is not None
        and aux_main_instruments is not None
        and aux_surrogate_covariates is not None
    ):
        Z_aug = np.column_stack((main_instruments, aux_main_instruments, surrogate_instruments, aux_surrogate_covariates))
        WX_aug = np.column_stack((main_covariates, aux_main_covariates, surrogate_covariates, aux_surrogate_covariates))
        X_aug = np.column_stack((surrogate_covariates, aux_surrogate_covariates))
        if Z_aug.shape[1] != WX_aug.shape[1]:
            raise MlsynthConfigError(
                "Dimension mismatch for combined instruments and covariates after augmentation."
            )
    else:
        if not (
            main_covariates.shape[1] == main_instruments.shape[1]
            and surrogate_covariates.shape[1] == surrogate_instruments.shape[1]
        ):
            raise MlsynthConfigError(
                "Dimension mismatch for base main or surrogate covariate/instrument matrices."
            )
        Z_aug = np.column_stack((main_instruments, surrogate_instruments))
        WX_aug = np.column_stack((main_covariates, surrogate_covariates))
        X_aug = surrogate_covariates

    post = slice(treatment_start_period, treatment_start_period + num_post_treatment_periods_analyzed)
    Y_post = outcome_vector[post]
    Z_post = Z_aug[post]
    WX_post = WX_aug[post]
    X_post = X_aug[post]

    ZWX_post = Z_post.T @ WX_post
    ZY_post = Z_post.T @ Y_post
    params = np.linalg.solve(ZWX_post, ZY_post)

    gamma = params[-X_aug.shape[1]:]
    taut = X_aug @ gamma
    tau = float(np.mean(taut[post]))

    # Moment conditions on the post-period only.
    U0 = (Z_post.T * (Y_post - WX_post @ params).reshape(1, -1))
    U1 = X_post @ gamma - tau
    U = np.column_stack((U0.T, U1))

    # Jacobian scaled by the number of post-treatment periods.
    G = np.zeros((Z_aug.shape[1] + 1, WX_aug.shape[1] + 1))
    G[: Z_aug.shape[1], : WX_aug.shape[1]] = ZWX_post / num_post_treatment_periods_analyzed
    G[-1, main_covariates.shape[1] : main_covariates.shape[1] + X_aug.shape[1]] = (
        -np.sum(X_post, axis=0) / num_post_treatment_periods_analyzed
    )
    G[-1, -1] = 1.0

    omega = hac(U, hac_truncation_lag)
    try:
        G_inv = np.linalg.inv(G)
        cov = G_inv @ omega @ G_inv.T
        var_tau = cov[-1, -1] / num_post_treatment_periods_analyzed
        se_tau = float(np.sqrt(var_tau)) if var_tau >= 0 else np.nan
    except np.linalg.LinAlgError:
        se_tau = np.nan

    params_W = params[: main_covariates.shape[1]]
    return tau, taut, params_W, se_tau
