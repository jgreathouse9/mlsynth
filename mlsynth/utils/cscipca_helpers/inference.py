"""Moving-block conformal inference for CSC-IPCA (Wang 2024, Sec. 3.1).

Implements the sharp-null / block-permutation conformal procedure of
Chernozhukov, Wuthrich & Zhu (2021), as in the reference implementation. For a
candidate effect ``theta`` the treated series is adjusted under the null
``Y_it - theta``, the model is re-estimated, and the post-treatment residual is
compared against the pre-treatment residuals by permutation. A candidate is
retained in the ``(1 - alpha)`` band when its permutation p-value is at least
``alpha``.

Two products:

* a per-period band (block length 1) -- the primary "effect over time" output;
* an ATT band (moving block of length ``n_post``, constant-effect null).

The factors ``F`` are estimated from the control units only and so do not
depend on ``theta``; they are computed once per inference window and reused
across the null grid.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .als import als_estimate, counterfactual, solve_gamma
from .structures import CSCIPCADesign, CSCIPCAInputs, CSCIPCAInference


def _moving_block_pvalue(resid: np.ndarray, block: int) -> float:
    """Permutation p-value comparing the post-block statistic to all blocks.

    The post-block is the last ``block`` residuals; ``T`` circular blocks of
    the same length are formed, each scored by mean absolute residual, and the
    p-value is the share of blocks whose score is at least the post-block's.
    """
    u = np.abs(np.asarray(resid, dtype=float))
    T = u.shape[0]
    stats = np.array([np.roll(u, s)[-block:].mean() for s in range(T)])
    return float(np.mean(stats >= stats[0]))


def _refit_residual(
    y_win: np.ndarray, Xt_win: np.ndarray, F_win: np.ndarray, K: int
) -> np.ndarray:
    """Residual of the treated window after re-solving the mapping under a null."""
    gamma = solve_gamma(y_win[None], Xt_win[None], F_win, K)
    yhat = counterfactual(Xt_win[None], gamma, F_win)[0]
    return y_win - yhat


def _invert_grid(
    grid: np.ndarray, pvalues: np.ndarray, alpha: float
) -> Tuple[float, float]:
    """Band = [min, max] of grid points whose p-value is at least ``alpha``."""
    keep = grid[pvalues >= alpha]
    if keep.size == 0:  # pragma: no cover - the point estimate always survives
        return float("nan"), float("nan")
    return float(keep.min()), float(keep.max())


def cscipca_conformal(
    inputs: CSCIPCAInputs,
    design: CSCIPCADesign,
    n_factors: int,
    max_iter: int,
    tol: float,
    alpha: float,
    n_nulls: int,
    null_grid_scale: float,
) -> CSCIPCAInference:
    """Per-period and ATT conformal bands for the CSC-IPCA effect."""
    K = int(n_factors)
    T, T0 = inputs.T, inputs.T0
    n_post = T - T0
    Yc = inputs.control_outcomes.T            # (N_co, T)
    Xc = inputs.control_covariates            # (N_co, T, L)
    y = inputs.treated_outcome
    Xt = inputs.treated_covariates
    tau = np.asarray(design.tau, dtype=float)
    half = max(float(null_grid_scale) * float(design.pre_rmse), 1e-8)

    # ----- per-period bands (block length 1) -----
    ci_lo = np.empty(n_post)
    ci_hi = np.empty(n_post)
    p_zero = np.empty(n_post)
    pre_idx = list(range(T0))
    for p in range(n_post):
        tp = T0 + p
        window = pre_idx + [tp]
        F_win, _g, _n, _c = als_estimate(Yc[:, window], Xc[:, window, :], K, max_iter, tol)
        Xt_win = Xt[window]
        y_win = y[window]

        grid = np.linspace(tau[p] - half, tau[p] + half, int(n_nulls))
        pv = np.empty(grid.shape[0])
        for j, theta in enumerate(grid):
            adj = y_win.copy()
            adj[-1] -= theta
            u = _refit_residual(adj, Xt_win, F_win, K)
            pv[j] = _moving_block_pvalue(u, 1)
        ci_lo[p], ci_hi[p] = _invert_grid(grid, pv, alpha)

        u0 = _refit_residual(y_win, Xt_win, F_win, K)     # null theta = 0
        p_zero[p] = _moving_block_pvalue(u0, 1)

    # ----- ATT band (moving block length n_post, constant-effect null) -----
    F_full, _g, _n, _c = als_estimate(Yc, Xc, K, max_iter, tol)
    att_hat = float(design.att)
    grid = np.linspace(att_hat - half, att_hat + half, int(n_nulls))
    pv = np.empty(grid.shape[0])
    for j, theta in enumerate(grid):
        adj = y.copy()
        adj[T0:] -= theta
        u = _refit_residual(adj, Xt, F_full, K)
        pv[j] = _moving_block_pvalue(u, n_post)
    att_lo, att_hi = _invert_grid(grid, pv, alpha)
    u0 = _refit_residual(y, Xt, F_full, K)
    att_p = _moving_block_pvalue(u0, n_post)

    return CSCIPCAInference(
        alpha=float(alpha),
        tau=tau,
        ci_lower_t=ci_lo,
        ci_upper_t=ci_hi,
        p_value_zero_t=p_zero,
        att=att_hat,
        att_p_value=att_p,
        att_lower=att_lo,
        att_upper=att_hi,
    )
