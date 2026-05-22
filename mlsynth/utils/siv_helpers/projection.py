"""Inference procedures for SIV.

Two paths are exposed:

* ``"asymptotic"`` — Gaussian CI from the heteroskedasticity-robust
  standard error attached to each :class:`SIVEstimate`. Valid under
  the regime of Theorem 4 (``JT_1 \\to \\infty`` and
  ``J / T_0 \\to 0``).
* ``"conformal"`` — split-conformal test described in Section 5.2 of
  the paper. The pre-period is split into a training block and a
  "blank" block; reduced-form event-study coefficients are computed
  on the debiased post-period and on the blank pre-period, and a
  permutation test inverts the null ``H_0 : \\theta_l = \\theta_k``
  for all ``l \\le T_0`` and ``k > T_0`` (which, under the design
  assumptions, is equivalent to ``H_0 : \\theta = 0``).
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
from scipy.stats import norm

from .structures import SIVEstimate, SIVInference, SIVInputs, SIVWeights


def asymptotic_ci(
    estimate: SIVEstimate,
    alpha: float = 0.05,
) -> SIVInference:
    """Gaussian CI + two-sided p-value from the IV sandwich SE."""

    z = float(norm.ppf(1.0 - alpha / 2.0))
    if not np.isfinite(estimate.se):
        ci_lower = float("nan")
        ci_upper = float("nan")
        p_value = float("nan")
    else:
        ci_lower = estimate.theta_hat - z * estimate.se
        ci_upper = estimate.theta_hat + z * estimate.se
        if estimate.se > 0:
            p_value = float(
                2.0 * (1.0 - norm.cdf(abs(estimate.theta_hat) / estimate.se))
            )
        else:
            p_value = float("nan")
    return SIVInference(
        method="asymptotic",
        alpha=float(alpha),
        theta_hat=float(estimate.theta_hat),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
    )


def _event_study_coefs(
    Y_tilde: np.ndarray, Z_tilde: np.ndarray, T0: int,
) -> np.ndarray:
    """Per-period reduced-form coefficient: ``theta_t = Z_t' Y_t / Z_t' Z_t``.

    Returns a length-``T`` vector; pre-period entries that have no
    instrument variation (Z = 0) return 0 so they don't pollute the
    permutation distribution.
    """

    T = Y_tilde.shape[1]
    coefs = np.zeros(T, dtype=float)
    for t in range(T):
        z_t = Z_tilde[:, t]
        y_t = Y_tilde[:, t]
        denom = float(z_t @ z_t)
        if denom > 1e-12:
            coefs[t] = float(z_t @ y_t) / denom
    return coefs


def split_conformal_inference(
    inputs: SIVInputs,
    weights: SIVWeights,
    estimate: SIVEstimate,
    alpha: float = 0.05,
    max_permutations: int = 5000,
    seed: int = 0,
) -> SIVInference:
    """Split-conformal permutation test on event-study coefficients.

    Procedure (paper Section 5.2):

      1. ``T_b = inputs.T0_train`` defines a training block
         ``[0, T_b)`` and a blank block ``[T_b, T_0)``.
      2. Compute per-period reduced-form coefficients
         ``theta_t`` over the *blank* and *post* periods (the
         training block was already used to fit the SC weights, so
         residuals there are not exchangeable).
      3. Build the permutation distribution of the statistic
         ``S(theta) = mean(|theta|)`` by drawing all (or up to
         ``max_permutations``) length-``T_1`` subsets of the combined
         blank+post coefficient vector.
      4. Return the p-value ``Pr(S(theta_pi) >= S(theta_obs))``.

    The CI bounds are obtained by grid-inverting the same test
    statistic, treating each candidate ``theta_0`` by subtracting
    ``theta_0`` from the *post*-period event-study coefficients
    before computing the statistic.
    """

    T0_train = inputs.T0_train
    if T0_train is None or T0_train < 1 or T0_train >= inputs.T0:
        return SIVInference(
            method="conformal",
            alpha=float(alpha),
            theta_hat=float(estimate.theta_hat),
        )

    coefs = _event_study_coefs(weights.Y_tilde, weights.Z_tilde, inputs.T0)

    blank_coefs = coefs[T0_train: inputs.T0]
    post_coefs = coefs[inputs.T0:]
    n_post = post_coefs.size
    pool = np.concatenate([blank_coefs, post_coefs])
    n_pool = pool.size

    if n_pool < n_post + 1 or n_post == 0:
        return SIVInference(
            method="conformal",
            alpha=float(alpha),
            theta_hat=float(estimate.theta_hat),
            event_study_coefs=coefs,
        )

    rng = np.random.default_rng(seed)

    def _stat(vec: np.ndarray) -> float:
        return float(np.mean(np.abs(vec)))

    observed_stat = _stat(post_coefs)

    # Build the permutation distribution. Use full enumeration when
    # cheap, random sampling otherwise.
    from math import comb

    total_perms = comb(n_pool, n_post)
    if total_perms <= max_permutations:
        perm_stats = np.empty(total_perms, dtype=float)
        for idx, subset in enumerate(combinations(range(n_pool), n_post)):
            perm_stats[idx] = _stat(pool[list(subset)])
    else:
        perm_stats = np.empty(max_permutations, dtype=float)
        for idx in range(max_permutations):
            subset = rng.choice(n_pool, size=n_post, replace=False)
            perm_stats[idx] = _stat(pool[subset])

    p_value = float(np.mean(perm_stats >= observed_stat))

    # CI by grid-inverting the test. Use a generous neighborhood
    # around theta_hat measured in standard errors.
    se = estimate.se if np.isfinite(estimate.se) and estimate.se > 0 else 1.0
    grid = np.linspace(
        estimate.theta_hat - 8.0 * se,
        estimate.theta_hat + 8.0 * se,
        401,
    )
    accepted = []
    # Equivalent reduced-form null: subtracting theta_0 from post means
    # we test whether (post - theta_0) sits inside the blank distribution.
    for theta_0 in grid:
        adjusted_post = post_coefs - theta_0 * estimate.beta_first_stage
        s_adj = _stat(adjusted_post)
        if np.mean(perm_stats >= s_adj) > alpha:
            accepted.append(float(theta_0))
    if accepted:
        ci_lower = min(accepted)
        ci_upper = max(accepted)
    else:
        ci_lower = estimate.theta_hat - 4.0 * se
        ci_upper = estimate.theta_hat + 4.0 * se

    return SIVInference(
        method="conformal",
        alpha=float(alpha),
        theta_hat=float(estimate.theta_hat),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        event_study_coefs=coefs,
        permutation_pvalue=float(p_value),
    )
