"""Proximal Inference (PI) -- donors-only, two-proxy GMM.

Implements the PI estimator of Shi, Li, Miao, Hu and Tchetgen Tchetgen
(2023, arXiv:2108.13935): donor outcomes ``W`` are negative-control
outcomes instrumented by donor proxies ``Z0`` on the pre-period, and the
fitted relationship imputes the post-period counterfactual. Closes with the
GMM sandwich variance of the ATT (HAC/Bartlett middle), validated
value-for-value against the authors' reference code (``freshtaste/proximal``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError
from ..inference import hac


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
