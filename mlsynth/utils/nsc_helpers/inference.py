"""Doudchenko-Imbens (2017) inference for NSC, R-faithful flavour.

Mirrors the placebo-style variance estimator used in the reference
R implementation (``NSC.R``, lines 200-220): for each donor ``j``,
hold it out and refit the NSC weights using the other ``J - 1`` donors
PLUS one randomly drawn extra donor (keeping the pool size at ``J``).
Record the *full-period* residual ``Y_{j,t} - sum_k w_k Y_{k,t}`` for
that fold. The per-period standard error is

    se_t = sqrt(sum_j perr_{j,t}^2 / (J - 1))

and the gap CI at significance ``alpha`` is the normal interval
``tau_hat_t ± z_{1-alpha/2} * se_t``. The ATT CI is built from the
average of the per-period post-period SEs in the same way as the
R script's reporting (an arithmetic mean of the ITE over the
post-period, with SE = average post-period SE / sqrt(T1)).
"""

from __future__ import annotations

from typing import Union

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
    seed: Union[int, np.random.Generator, None] = 123,
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
        Standardized donor matching matrix, shape ``(J, p)``.
    T0 : int
        Number of pre-treatment periods.
    a_star, b_star : float
        Dimensionless tuning parameters that produced ``counterfactual``.
        Used unchanged for the leave-one-control re-fits.
    alpha : float, default 0.05
        Two-sided significance level.
    seed : int, Generator, or None, default 123
        Seed (or RNG) for the extra-donor draws. Matches the reference
        R script's ``set.seed(123)``.
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

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Per-fold residuals: shape (J, T).
    perr = np.full((J, T), np.nan, dtype=float)
    for j in range(J):
        other = np.array([k for k in range(J) if k != j])
        idx = int(rng.choice(other))
        ZJ = np.vstack([Z0[other], Z0[idx][None, :]])
        Z1_j = Z0[j]
        try:
            eig = design_eigenvalues(ZJ)
            w, _, _ = fit_nsc(Z1_j, ZJ, a_star, b_star, eigvals=eig)
        except Exception:
            continue
        Y_pool = np.column_stack(
            [donor_outcomes[:, other], donor_outcomes[:, idx][:, None]]
        )
        perr[j] = donor_outcomes[:, j] - Y_pool @ w

    # Per-period SE = sqrt(sum_j perr_{j,t}^2 / (J - 1)), ignoring NaN folds.
    sq = perr ** 2
    valid = ~np.isnan(sq)
    sum_sq = np.where(valid, sq, 0.0).sum(axis=0)
    # Denominator is (number_of_valid_folds - 1); fall back to J-1 if all valid.
    n_valid = valid.sum(axis=0)
    denom = np.where(n_valid > 1, n_valid - 1, 1)
    period_variance = sum_sq / denom
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
