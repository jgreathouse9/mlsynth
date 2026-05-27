"""Synthetic Regressing Control (SRC) estimation core.

Relocated from the legacy ``estutils`` module so the SRC estimator is
self-contained. Implements the three steps of Chen-Shi-Cao (SRC):

* :func:`get_theta` -- per-donor alignment coefficients (and the aligned
  pre-period donor matrix);
* :func:`get_sigmasq` -- noise-variance estimate from the pre-period fit;
* :func:`src_optimize` -- the penalised non-negative weight QP and the
  pre/post counterfactual prediction;
* :func:`SRCest` -- the top-level driver returning the full counterfactual,
  donor weights, and alignment coefficients.

Depends only on numpy and cvxpy.
"""

from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import numpy as np

_SOLVER_OSQP = "OSQP"


def get_theta(
    treated_outcome_pre_treatment: np.ndarray,
    donor_outcomes_pre_treatment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-donor alignment coefficients and the aligned pre-period donor matrix.

    ``theta_j`` regresses the demeaned treated pre-period on each demeaned donor
    pre-period individually; the aligned matrix scales each donor column by its
    ``theta_j``.
    """
    treated_dm = treated_outcome_pre_treatment - np.mean(treated_outcome_pre_treatment)
    donor_dm = donor_outcomes_pre_treatment - np.mean(donor_outcomes_pre_treatment, axis=0)
    num = donor_dm.T @ treated_dm
    den = np.sum(donor_dm ** 2, axis=0)
    theta = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    aligned = donor_outcomes_pre_treatment * theta
    return theta, aligned


def get_sigmasq(
    treated_outcome_pre_treatment: np.ndarray,
    donor_outcomes_pre_treatment: np.ndarray,
) -> float:
    """Estimate the SRC noise variance from the pre-period projection residual."""
    T0 = len(treated_outcome_pre_treatment)
    Q = np.eye(T0) - np.ones((T0, T0)) / T0          # demeaning matrix
    G = donor_outcomes_pre_treatment.T @ Q @ donor_outcomes_pre_treatment
    diag = np.diag(G)
    inv_diag = np.divide(1.0, diag, out=np.zeros_like(diag), where=diag != 0)
    P = donor_outcomes_pre_treatment @ np.diag(inv_diag) @ donor_outcomes_pre_treatment.T
    projected = P @ Q @ treated_outcome_pre_treatment
    resid = Q @ treated_outcome_pre_treatment - Q @ projected
    return float(np.linalg.norm(resid) ** 2)


def src_optimize(
    treated_outcome_pre_treatment: np.ndarray,
    donor_outcomes_pre_treatment: np.ndarray,
    donor_outcomes_post_treatment: np.ndarray,
    aligned_donor_outcomes_pre_treatment: np.ndarray,
    alignment_coefficients: np.ndarray,
    noise_variance_estimate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SRC weight QP + pre/post counterfactual prediction (SRC paper Algorithm 1)."""
    num_donors = donor_outcomes_pre_treatment.shape[1]
    w = cp.Variable(num_donors)
    loss = cp.sum_squares(
        treated_outcome_pre_treatment - aligned_donor_outcomes_pre_treatment @ w
    )
    penalty = 2 * noise_variance_estimate * cp.sum(w)
    problem = cp.Problem(cp.Minimize(loss + penalty), [w >= 0, w <= 1])
    problem.solve(solver=_SOLVER_OSQP)
    w_opt = w.value

    mean_y1 = treated_outcome_pre_treatment.mean()
    mean_Y0 = donor_outcomes_pre_treatment.mean(axis=0)
    pred_pre = mean_y1 + (donor_outcomes_pre_treatment - mean_Y0) @ (w_opt * alignment_coefficients)
    pred_post = mean_y1 + (donor_outcomes_post_treatment - mean_Y0) @ (w_opt * alignment_coefficients)
    return pred_pre, pred_post, w_opt, alignment_coefficients


def SRCest(
    treated_outcome_full_period: np.ndarray,
    donor_outcomes_full_period: np.ndarray,
    num_post_treatment_periods: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic Regressing Control counterfactual, weights, and alignment coefficients.

    Parameters
    ----------
    treated_outcome_full_period : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    donor_outcomes_full_period : np.ndarray
        Donor outcomes over all periods, shape ``(T, J)``.
    num_post_treatment_periods : int
        Number of post-treatment periods.

    Returns
    -------
    (counterfactual_full, weights, theta) : tuple of np.ndarray
    """
    T, _ = donor_outcomes_full_period.shape
    T0 = T - num_post_treatment_periods
    y_pre = treated_outcome_full_period[:T0]
    X_pre = donor_outcomes_full_period[:T0]
    X_post = donor_outcomes_full_period[T0:]

    theta, X_pre_aligned = get_theta(y_pre, X_pre)
    sigma2 = get_sigmasq(y_pre, X_pre)
    pred_pre, pred_post, w_opt, _ = src_optimize(
        y_pre, X_pre, X_post, X_pre_aligned, theta, sigma2
    )
    counterfactual_full = np.concatenate([pred_pre, pred_post])
    return counterfactual_full, w_opt, theta
