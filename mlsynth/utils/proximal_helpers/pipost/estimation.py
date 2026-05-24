"""Post-treatment-only proximal surrogate estimator (PIPost).

Implements the post-only variant of Liu, Tchetgen Tchetgen and Varjao
(2023, arXiv:2308.09527): donor and surrogate coefficients are estimated
jointly from a single post-treatment IV fit, using ``(Z0, Z1)`` to
instrument ``(W, X)``. The GMM sandwich variance is scaled by the number of
post-treatment periods ``T1``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError
from ..inference import hac


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
