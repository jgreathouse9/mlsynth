"""Speed-only warm-start accelerator for the simplex-constrained least-squares
solver used across the bilevel SCM engine (outcome-only SC, ridge augmentation,
the conformal / placebo hot loops).

The exact solver is the primal active-set method in :mod:`.active_set`. It builds
the active set one pivot at a time, each pivot an ``lstsq`` on the current free
set, so a *cold* solve on a large donor pool (hundreds of donors) can take
seconds -- the cost is dominated by discovering *which* donors are in the support,
not by the final small solve on that support.

This module discovers the support cheaply. It collapses the pre-period (time)
dimension into the Gram matrix ``G = B^T B`` once and runs FISTA (accelerated
projected gradient, Beck & Teboulle 2009) with the exact Euclidean projection
onto the probability simplex (Held, Wolfe & Crowder 1974; Duchi et al. 2008).
FISTA converges to the neighbourhood of the optimum -- and hence the right
support -- in a few hundred cheap iterations; the exact active-set then certifies
from that warm start in a handful of pivots.

The accelerator is *speed only*. It supplies a feasible warm start; the exact
active-set (with the cvxpy fallback) still determines the returned weights, so
the answer is unchanged for any warm start. Correctness therefore never depends
on FISTA converging, the Lipschitz estimate being tight, or the problem being
well-conditioned -- only the runtime does.
"""
from __future__ import annotations

import numpy as np

# Engage the accelerator only for donor pools at least this wide. Below it the
# cold active-set already solves in well under a pivot's worth of FISTA overhead
# (measured crossover ~ J = 60-80; typical SC panels -- Prop 99 J = 38, Basque
# J = 16 -- stay comfortably on the untouched fast path). Also skipped whenever
# the caller already supplies a warm start (e.g. the conformal / placebo loops).
ACCEL_MIN_DONORS = 80


def simplex_project(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of ``v`` onto ``{w >= 0, sum w = 1}``.

    The sort-based method of Held, Wolfe & Crowder (1974) / Duchi et al. (2008):
    ``O(n log n)``, returns the unique projection ``(v - theta)_+`` for the
    threshold ``theta`` that makes the result sum to one.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.shape[0]
    if n == 1:
        return np.ones(1)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0                 # cond[0] is always True (u[0]-(u[0]-1) = 1 > 0)
    rho = ind[cond][-1]
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def _spectral_bound(G: np.ndarray, iters: int = 50) -> float:
    """An estimate of ``lambda_max(G)`` (deterministic power iteration), inflated
    by 10% so it is, in practice, an upper bound -- FISTA's stable step is
    ``1 / L`` with ``L >= lambda_max`` of the gradient. A loose or even invalid
    bound only slows the warm start (the simplex projection keeps every iterate
    bounded); it can never make the final active-set answer wrong.
    """
    n = G.shape[0]
    v = np.random.default_rng(0).standard_normal(n)   # fixed seed -> deterministic
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        Gv = G @ v
        nrm = np.linalg.norm(Gv)
        if nrm == 0.0:
            return 0.0
        v = Gv / nrm
        lam = float(v @ (G @ v))
    return lam * 1.1


def fista_warm_start(B: np.ndarray, A: np.ndarray, *,
                     max_iter: int = 400, tol: float = 1e-7) -> np.ndarray:
    """A feasible, near-optimal warm start for ``min ||A - B w||^2`` on the
    simplex, via Gram-collapsed FISTA.

    Deterministic. Returns a point on the simplex; it is intended to seed the
    exact active-set solver, not to be used as the final solution.
    """
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    J = B.shape[1]
    if J == 1:
        return np.ones(1)
    G = B.T @ B
    c = B.T @ A
    L = 2.0 * _spectral_bound(G)
    if not np.isfinite(L) or L <= 0.0:        # degenerate (e.g. all-zero donors)
        return np.full(J, 1.0 / J)
    w = np.full(J, 1.0 / J)
    y = w.copy()
    t = 1.0
    for _ in range(max_iter):
        w_new = simplex_project(y - (2.0 * (G @ y - c)) / L)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
        t = t_new
    return w
