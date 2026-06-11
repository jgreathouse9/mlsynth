"""Residual-resampling inference for the Grossi et al. (2025) estimator.

Implements the re-sampling procedure of Section 3.3 and the pivotal
bias-corrected confidence intervals of eqs. 3.6-3.7.

1. For each clean control, fit its synthetic control from the *other* clean
   controls and record the residual vector ``e_i = Y_i - Yhat_i`` over all
   periods.
2. Resample the control residual vectors with replacement and add them to the
   fitted control paths to form pseudo control outcomes ``Y*``.
3. Re-fit each treated-cluster unit's synthetic control against the pseudo
   control outcomes (with the penalty ``lambda`` fixed at its observed-data
   value, for speed) and recompute the direct / average-spillover effects.
4. Repeat ``n_boot`` times and form pivotal bias-corrected CIs.

Only the penalized backend (the paper's estimator) and outcome-only matching
carry inference; for other backends ``n_boot`` is ignored.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ...bilevel import (
    BilevelProblem, bias_corrected_gaps, penalized_weights, simplex_lstsq,
    solve_penalized,
)


def _std_cov(p_t: np.ndarray, P_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ctr = P_d.mean(axis=1)
    scl = P_d.std(axis=1) + 1e-8
    return (p_t - ctr) / scl, (P_d - ctr[:, None]) / scl[:, None]


def _fit_weights(y_target: np.ndarray, Y_donors: np.ndarray, T0: int,
                 *, predictors=None, ti=None, donor_idx=None,
                 lam=None) -> np.ndarray:
    """Weights for ``y_target`` against ``Y_donors`` (donors as columns, shape
    ``(T, J)``).  Penalized (outcomes-in-levels + std covariates) with fixed
    ``lam`` when covariates are present; plain simplex least squares otherwise.
    """
    if predictors is None:
        return simplex_lstsq(Y_donors[:T0], y_target[:T0])
    x1, X0 = _std_cov(predictors[ti], predictors[donor_idx].T)
    m1 = np.concatenate([y_target[:T0], x1])
    M0 = np.vstack([Y_donors[:T0], X0])
    return penalized_weights(m1, M0, float(lam))


def residual_resampling(
    Y: np.ndarray, T0: int, p: int, N: int,
    predictors: Optional[np.ndarray],
    lam_cluster: List[Optional[float]],
    *, n_boot: int, ci_level: float, seed: int,
    bias_correct: bool,
    direct_att: np.ndarray, avg_spill_att: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(direct_ci, avg_spillover_ci)``, each shape ``(T1, 2)``.

    ``direct_att`` / ``avg_spill_att`` are the per-period point estimates
    (length ``T1``) used to centre the pivotal bias-corrected intervals.
    """
    rng = np.random.default_rng(seed)
    T = Y.shape[1]
    T1 = T - T0
    cluster = list(range(p + 1))
    clean = np.arange(p + 1, N)
    nc = clean.size

    # 1. Clean-control fits + residual vectors (leave-one-out among controls).
    Yhat_c = np.zeros((nc, T))
    resid = np.zeros((nc, T))
    for a, j in enumerate(clean):
        others = clean[clean != j]
        if predictors is None:
            w = simplex_lstsq(Y[others][:, :T0].T, Y[j][:T0])
        else:
            x1, X0 = _std_cov(predictors[j], predictors[others].T)
            prob = BilevelProblem(
                y1_pre=Y[j][:T0], Y0_pre=Y[others][:, :T0].T, X1=x1, X0=X0)
            w = solve_penalized(prob).W
        Yhat = Y[others].T @ w
        Yhat_c[a] = Yhat
        resid[a] = Y[j] - Yhat

    # 2-3. Bootstrap: resample residuals, re-fit cluster units on pseudo controls.
    boot_direct = np.zeros((n_boot, T1))
    boot_spill = np.zeros((n_boot, T1))
    for b in range(n_boot):
        Ystar = Yhat_c + resid[rng.integers(0, nc, size=nc)]   # (nc, T)
        Ystar_full = np.empty((N, T))                          # only clean rows used
        Ystar_full[clean] = Ystar
        eff = []
        for ci, i in enumerate(cluster):
            w = _fit_weights(Y[i], Ystar.T, T0, predictors=predictors,
                             ti=i, donor_idx=clean, lam=lam_cluster[ci])
            Yhat_i = Ystar.T @ w
            if bias_correct and predictors is not None:
                x1, X0 = _std_cov(predictors[i], predictors[clean].T)
                gap = bias_corrected_gaps(w, x1, X0, Y[i], Ystar.T)
            else:
                gap = Y[i] - Yhat_i
            eff.append(gap[T0:])
        boot_direct[b] = eff[0]
        boot_spill[b] = np.mean(np.vstack(eff[1:]), axis=0)

    # 4. Pivotal bias-corrected CIs (eqs. 3.6-3.7).
    alpha = 1.0 - ci_level

    def _ci(boot: np.ndarray, point: np.ndarray) -> np.ndarray:
        q_lo = np.quantile(boot, alpha / 2, axis=0)
        q_hi = np.quantile(boot, 1 - alpha / 2, axis=0)
        med = np.quantile(boot, 0.5, axis=0)
        bias = point - med
        lower = 2 * point - q_hi - bias
        upper = 2 * point - q_lo - bias
        return np.column_stack([lower, upper])

    return _ci(boot_direct, direct_att), _ci(boot_spill, avg_spill_att)
