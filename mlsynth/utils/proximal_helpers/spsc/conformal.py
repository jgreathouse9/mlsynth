"""Conformal prediction intervals for SPSC (Park & Tchetgen Tchetgen 2025, Sec. 3.5).

Constructs pointwise prediction intervals for the per-period treatment
effect ``xi_t = y^1_{0t} - y^0_{0t}`` by inverting the permutation test of
Chernozhukov, Wuthrich and Zhu (2021). For a candidate effect ``xi`` at a
post-period ``s``, the treated outcome is "un-treated" (``y_s - xi``),
appended to the pre-period sample, the synthetic-control weights are
re-fit (with the ridge penalty held fixed), and a conformal p-value is
formed from the rank of the appended residual among all residuals. The
interval is the set of ``xi`` not rejected at the target level.

This is a faithful port of the ``conformal.interval`` branch of the
authors' reference R package (``github.com/qkrcks0218/SPSC``), and unlike
the asymptotic GMM standard error it remains valid with a short
post-treatment period.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .estimation import _poly_basis, _ridge_ginv, _scaled_detrend_basis


def _refit_gamma(y_aug, W_aug, D_aug, lam, detrend, degree=1):
    """Re-fit the ridge SC weights on an augmented all-pre sample (fixed lam).

    Uses the same ``ginv`` ridge solve as the point fit (:func:`_ridge_ginv`):
    a plain ``solve`` at the small CV penalty makes the refit gamma swing with
    the single appended period, which inflates and skews the conformal band
    away from the reference.
    """
    if detrend:
        eta = np.linalg.lstsq(D_aug, y_aug, rcond=None)[0]
        g = np.column_stack([D_aug, _poly_basis(y_aug - D_aug @ eta, degree)])
    else:
        g = _poly_basis(y_aug, degree)
    N = W_aug.shape[1]
    GY = (g * y_aug[:, None]).mean(0)
    GW = np.stack([(g * W_aug[:, n: n + 1]).mean(0) for n in range(N)], axis=1)
    return _ridge_ginv(GW, GY, lam)


def _conformal_pvalue(residuals: np.ndarray) -> float:
    """Rank-based conformal p-value: fraction of |residual| >= the appended one.

    The appended (hypothesized post) period is the last row. Mirrors the
    reference ``Calculate.PValue`` with a single post period.
    """
    s_base = abs(residuals[-1])
    return float(np.mean(np.abs(residuals) >= s_base))


def conformal_intervals(
    outcome_vector: np.ndarray,
    donor_outcomes: np.ndarray,
    num_pre_treatment_periods: int,
    gamma: np.ndarray,
    ridge_lambda: float,
    detrend: bool,
    spline_df: int,
    att_se: float,
    periods: Optional[Sequence[int]] = None,
    alpha: float = 0.05,
    window: float = 25.0,
    grid_size: int = 101,
    basis_degree: int = 1,
    att_degree: int = 0,
    detrend_basis: str = "bspline",
    detrend_degree: int = 1,
) -> Dict[str, np.ndarray]:
    """Pointwise conformal prediction intervals for the per-period effect.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes ``W``, shape ``(T, N)``.
    num_pre_treatment_periods : int
        ``T0``.
    gamma : np.ndarray
        Point-estimate SC weights (used to center the search grid).
    ridge_lambda : float
        log10 ridge penalty held fixed during the inversion.
    detrend : bool
        Whether the SPSC fit detrends (must match the point fit).
    spline_df : int
        Detrend B-spline degrees of freedom.
    att_degree : int, default 0
        ATT-basis polynomial degree of the point fit (reference ``att.ft``);
        only affects the detrend ``Scale`` via the HAC bandwidth, so it must
        match the point fit.
    detrend_basis : {"bspline", "poly"}, default "bspline"
        Detrend trend family of the point fit (must match it).
    detrend_degree : int, default 1
        Polynomial degree when ``detrend_basis="poly"``.
    att_se : float
        Asymptotic ATT standard error, used to scale the search grid. If
        not finite, a data-driven width is used.
    periods : sequence of int, optional
        Post-treatment period indices (absolute, in ``[T0, T)``) to cover.
        Defaults to every post-treatment period.
    alpha : float, default 0.05
        Target miscoverage (95% interval).
    window : float, default 25.0
        Half-width of the (SE-scaled) coarse search grid.
    grid_size : int, default 101
        Number of coarse grid points (the reference uses 101).

    Returns
    -------
    dict
        ``{"periods": int array, "lower": float array, "upper": float
        array}`` -- prediction interval for ``xi_t`` at each covered period.
    """

    y = np.asarray(outcome_vector, dtype=float).ravel()
    W = np.asarray(donor_outcomes, dtype=float)
    T, N = W.shape
    T0 = int(num_pre_treatment_periods)
    if periods is None:
        periods = list(range(T0, T))
    periods = [int(p) for p in periods]

    # Refit on the SAME rescaled detrend basis the point fit used: the ridge
    # ginv truncation depends on the instrument magnitudes, so an unscaled basis
    # gives a different gamma and a band that drifts from the reference.
    D_full = (_scaled_detrend_basis(y, W, T0, spline_df, ridge_lambda,
                                    basis_degree, att_degree, detrend_basis,
                                    detrend_degree)
              if detrend else None)
    pre_idx = np.arange(T0)

    # Achievable discrete level (reference: max of {2/(T0+1),...,1} <= alpha).
    levels = (np.arange(1, T0 + 1) + 1) / (T0 + 1)
    eligible = levels[levels <= alpha]
    valid_p = float(eligible.max()) if eligible.size else 1.0 / (T0 + 1)

    def pvalue(idx: int, xi: float) -> float:
        rows = np.append(pre_idx, idx)
        y_aug = y[rows].copy()
        y_aug[-1] = y[idx] - xi
        W_aug = W[rows]
        D_aug = D_full[rows] if detrend else None
        g = _refit_gamma(y_aug, W_aug, D_aug, ridge_lambda, detrend, basis_degree)
        return _conformal_pvalue(y_aug - W_aug @ g)

    def interval_for(idx: int):
        point = y[idx] - W[idx] @ gamma
        unit = att_se if (att_se is not None and np.isfinite(att_se) and att_se > 0) else \
            float(np.std(y[T0:] - (W[T0:] @ gamma))) or 1.0
        center = grid_size // 2
        grid = point + np.linspace(-window, window, grid_size) * unit
        accept = np.array([pvalue(idx, xi) >= valid_p for xi in grid])
        if not accept[center]:
            # Point itself rejected (degenerate); fall back to the SE band.
            return point - 1.96 * unit, point + 1.96 * unit
        # Consecutive accepted run containing the center.
        lo = center
        while lo - 1 >= 0 and accept[lo - 1]:
            lo -= 1
        hi = center
        while hi + 1 < grid_size and accept[hi + 1]:
            hi += 1
        # Narrow refinement just outside each edge.
        step = grid[1] - grid[0]
        lower = grid[lo]
        if lo > 0:
            fine = np.linspace(grid[lo] - step, grid[lo], 51)
            acc = np.array([pvalue(idx, xi) >= valid_p for xi in fine])
            lower = fine[acc].min() if acc.any() else grid[lo]
        upper = grid[hi]
        if hi < grid_size - 1:
            fine = np.linspace(grid[hi], grid[hi] + step, 51)
            acc = np.array([pvalue(idx, xi) >= valid_p for xi in fine])
            upper = fine[acc].max() if acc.any() else grid[hi]
        return float(lower), float(upper)

    lowers, uppers = [], []
    for idx in periods:
        lb, ub = interval_for(idx)
        lowers.append(lb)
        uppers.append(ub)

    return {
        "periods": np.asarray(periods),
        "lower": np.asarray(lowers),
        "upper": np.asarray(uppers),
    }
