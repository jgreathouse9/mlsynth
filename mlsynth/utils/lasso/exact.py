"""Exact lasso via Langlois & Darbon's differential-inclusion homotopy.

Langlois, G. P., & Darbon, J. (2025). *"A fast algorithm for solving the lasso
problem exactly without homotopy using differential inclusions."*
arXiv:2507.05562.

The algorithm solves

.. math::

   \\min_x\\ \\lVert x \\rVert_1 + \\frac{1}{2t}\\lVert A x - b \\rVert_2^2,
   \\qquad t > 0,

*exactly* (to machine precision) by tracing the trajectory of a projected
dynamical system. Unlike coordinate descent, there is no convergence tolerance:
the inner cone projection is an exact (warm-started, active-set) non-negative
least squares, and the outer step uses the paper's closed-form maximal descent
time, so the returned solution is the true minimiser.

:func:`lasso_exact_path` exploits the homotopy structure: walking a *descending*
penalty grid, each solve resumes from the previous (sparser) solution's dual
point and support, so the entire regularisation path costs about one sweep --
far cheaper than refitting per penalty, and exact at every grid point.

This maps to scikit-learn's parametrisation by ``t = n_samples * alpha`` (the
penalty multiplied by the sample size), so ``lasso_exact(A, b, n*alpha)`` and
``sklearn.Lasso(alpha)`` target the same objective.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_TOL = 1e-12


def _solve_psd(G: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Solve ``G x = h`` for small PSD ``G``; Cholesky fast path, ``lstsq``
    fallback when ``G`` is rank-deficient (collinear / duplicate columns)."""
    try:
        L = np.linalg.cholesky(G)
        return np.linalg.solve(L.T, np.linalg.solve(L, h))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate active block
        return np.linalg.lstsq(G, h, rcond=None)[0]


def _nnls_warm(M: np.ndarray, r: np.ndarray,
               passive: Optional[np.ndarray] = None,
               tol: float = _TOL) -> Tuple[np.ndarray, np.ndarray]:
    """``min_{u>=0} ||M u - r||^2`` (Lawson-Hanson), warm-started from ``passive``.

    Passive solves use the precomputed Gram matrix (normal equations + Cholesky),
    so each inner least squares is ``O(|P|^3)`` rather than a fresh SVD. Returns
    ``(u, support_indices)``.
    """
    m, k = M.shape
    maxit = 30 * (k + 1)
    G = M.T @ M
    h = M.T @ r
    u = np.zeros(k)
    inP = np.zeros(k, dtype=bool)

    def passive_solve():
        P = np.where(inP)[0]
        s = np.zeros(k)
        if len(P):
            s[P] = _solve_psd(G[np.ix_(P, P)], h[P])
        return s, P

    if passive is not None and len(passive):
        inP[passive] = True
        for _ in range(k + 1):
            s, P = passive_solve()
            if np.all(s[P] > tol):
                u = np.where(inP, s, 0.0)
                break
            bad = P[s[P] <= tol]
            alpha = np.min(u[bad] / (u[bad] - s[bad]))
            u = u + alpha * (s - u)
            inP[P[u[P] <= tol]] = False
            u[~inP] = 0.0
        else:  # pragma: no cover - warm start failed to become feasible
            inP[:] = False
            u[:] = 0.0

    for _ in range(maxit):
        w = h - G @ u
        Z = ~inP
        if not Z.any() or np.max(w[Z]) <= tol:
            break
        inP[np.where(Z)[0][np.argmax(w[Z])]] = True
        for _ in range(maxit):
            s, P = passive_solve()
            if np.all(s[P] > tol):
                u = np.where(inP, s, 0.0)
                break
            bad = P[s[P] <= tol]
            alpha = np.min(u[bad] / (u[bad] - s[bad]))
            u = u + alpha * (s - u)
            inP[P[u[P] <= tol]] = False
            u[~inP] = 0.0
    return u, np.where(inP)[0]


def lasso_exact(A: np.ndarray, b: np.ndarray, t: float,
                p_init: Optional[np.ndarray] = None,
                passive: Optional[np.ndarray] = None,
                maxit: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact minimiser of ``||x||_1 + (1/2t) ||A x - b||^2`` (``t > 0``).

    ``p_init`` / ``passive`` warm-start the dual point and active support (used by
    :func:`lasso_exact_path`). Returns ``(x, p, support)`` -- the primal solution,
    the dual point (a valid warm start for a smaller ``t``), and the active set.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    n = A.shape[1]
    p = (b / np.max(np.abs(A.T @ b))) if p_init is None else p_init.copy()
    last_E = None
    stall = 0
    for _ in range(maxit):
        c = -(A.T @ p)
        E = np.abs(c) >= 1.0 - 1e-9
        sgn = np.sign(c)
        sgn[sgn == 0] = 1.0
        Eidx = np.where(E)[0]
        M = A[:, Eidx] * sgn[Eidx]
        r = b + t * p
        seed = None
        if passive is not None:
            pos = {g: i for i, g in enumerate(Eidx)}
            seed = np.array([pos[g] for g in passive if g in pos], dtype=int)
        uE, locP = _nnls_warm(M, r, passive=seed)
        passive = Eidx[locP]
        d = M @ uE - r
        g = sgn * (A.T @ d)
        hh = sgn * (A.T @ p)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (np.sign(g) - hh) / g
        ratio[~np.isfinite(ratio)] = np.inf
        ratio[ratio <= 1e-12] = np.inf
        Delta = ratio.min()
        if t > 0 and (not np.isfinite(Delta) or t * Delta >= 1.0 - 1e-12):
            x = np.zeros(n)
            x[Eidx] = sgn[Eidx] * uE
            return x, p + d / t, passive
        Ekey = tuple(Eidx.tolist())
        if Ekey == last_E and Delta < 1e-11:        # anti-cycling on a degenerate stall
            stall += 1
            if stall > 20:
                Delta = max(Delta, 1e-9)
            if stall > 60:  # pragma: no cover - hard stall backstop
                x = np.zeros(n)
                x[Eidx] = sgn[Eidx] * uE
                return x, p + d / t, passive
        else:
            stall = 0
        last_E = Ekey
        p = p + Delta * d
    raise RuntimeError("lasso_exact: Algorithm 1 did not converge within maxit")  # pragma: no cover


def lasso_exact_path(A: np.ndarray, b: np.ndarray,
                     t_grid: np.ndarray) -> np.ndarray:
    """Exact coefficients at each ``t`` in a *descending* grid, warm-started.

    Each solve resumes from the previous (larger ``t``, sparser) solution's dual
    point and support, so the whole path costs about one homotopy sweep. Returns
    coefficients of shape ``(n_features, len(t_grid))``.
    """
    A = np.asarray(A, float)
    n = A.shape[1]
    coefs = np.zeros((n, len(t_grid)))
    p = None
    passive = None
    for k, t in enumerate(t_grid):
        x, p, passive = lasso_exact(A, b, float(t), p_init=p, passive=passive)
        coefs[:, k] = x
    return coefs
