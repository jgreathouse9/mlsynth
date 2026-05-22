"""Doudchenko-Imbens (2017) inference for NSC.

Estimates the per-period variance of the SC estimator by

    sigma_t^2 ~= mean_k MSE( Y_{k, t}, sum_j w_{j -> k} Y_{j, t} )

where ``w_{j -> k}`` are the NSC weights obtained by predicting
donor ``k`` from the other donors using the same ``(a_star, b_star)``
the treated-unit fit selected. The variance estimate is then used to
build period-by-period normal confidence intervals around the gap
``tau_hat_t = Y_{1, t} - Y_{1, t}^{SC}`` and, by averaging across the
post-period, a CI for the ATT.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .optimization import design_eigenvalues, fit_nsc
from .structures import NSCInference


def doudchenko_imbens_inference(
    treated_outcome: np.ndarray,
    donor_outcomes: np.ndarray,
    counterfactual: np.ndarray,
    Z0: np.ndarray,
    T0: int,
    a_star: float,
    b_star: float,
    alpha: float = 0.05,
) -> NSCInference:
    """Compute per-period and ATT inference from the leave-one-control fits.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcome matrix, shape ``(T, J)``.
    counterfactual : np.ndarray
        SC imputation of the treated outcome, shape ``(T,)``.
    Z0 : np.ndarray
        Donor matching matrix, shape ``(J, p)``.
    T0 : int
        Number of pre-treatment periods.
    a_star, b_star : float
        Dimensionless tuning parameters that produced ``counterfactual``.
        Used unchanged for the leave-one-control re-fits.
    alpha : float, default 0.05
        Two-sided significance level.
    """

    T = treated_outcome.shape[0]
    J = donor_outcomes.shape[1]

    gap = treated_outcome - counterfactual

    if J < 3 or T <= T0:
        return NSCInference(
            method="none",
            alpha=float(alpha),
            gap=gap,
        )

    # Leave-one-control: for each donor k, re-fit NSC weights using the
    # other J-1 donors as the donor pool, target = donor k. Predict
    # donor k's outcome at every period; record per-period squared
    # residuals; average across k to get the variance estimate.
    sq_resids = np.zeros((J, T), dtype=float)
    successes = np.zeros(T, dtype=int)
    for k in range(J):
        mask = np.ones(J, dtype=bool)
        mask[k] = False
        Z0_loo = Z0[mask]
        Z1_loo = Z0[k]
        Y0_loo = donor_outcomes[:, mask]
        Y_target = donor_outcomes[:, k]
        eig_loo = design_eigenvalues(Z0_loo)
        try:
            w_loo, _, _ = fit_nsc(
                Z1_loo, Z0_loo, a_star, b_star, eigvals=eig_loo
            )
        except Exception:
            continue
        pred = Y0_loo @ w_loo
        sq_resids[k] = (Y_target - pred) ** 2
        successes += 1

    successes = np.where(successes > 0, successes, 1)
    period_variance = sq_resids.sum(axis=0) / successes
    period_se = np.sqrt(np.clip(period_variance, a_min=0.0, a_max=None))

    z = float(norm.ppf(1.0 - alpha / 2.0))
    gap_lower = gap - z * period_se
    gap_upper = gap + z * period_se

    post_var = period_variance[T0:]
    if post_var.size:
        att = float(np.mean(gap[T0:]))
        att_se = float(np.sqrt(np.mean(post_var)) / np.sqrt(post_var.size))
        att_lower = att - z * att_se
        att_upper = att + z * att_se
        if att_se > 0:
            p_value = 2.0 * (1.0 - float(norm.cdf(abs(att) / att_se)))
        else:
            p_value = float("nan")
    else:
        att = att_se = att_lower = att_upper = float("nan")
        p_value = float("nan")

    return NSCInference(
        method="doudchenko_imbens",
        alpha=float(alpha),
        period_variance=period_variance,
        period_se=period_se,
        gap=gap,
        gap_lower=gap_lower,
        gap_upper=gap_upper,
        att=att,
        att_se=att_se,
        att_lower=att_lower,
        att_upper=att_upper,
        p_value=p_value,
    )
