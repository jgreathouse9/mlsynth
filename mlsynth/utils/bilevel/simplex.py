"""Self-contained simplex-constrained least squares (no external QP solver).

These primitives replace ``Opt.SCopt`` for the bilevel SCM experiment. The
workhorse is an accelerated projected-gradient (FISTA) solver for

    min_w  ||A w - b||^2   s.t.   w >= 0,  sum(w) = 1,

built from two small, independently testable pieces: a Euclidean projection
onto the probability simplex and a power-iteration estimate of the gradient's
Lipschitz constant.
"""

from __future__ import annotations

import warnings

import numpy as np

_EPS = 1e-12


def project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``{w >= 0, sum(w) = z}``.

    Uses the exact sort-based algorithm (Held, Wolfe & Crowder 1974; Duchi et
    al. 2008): :math:`O(n \\log n)` and exact.
    """
    if z <= 0:
        raise ValueError(f"simplex radius z must be positive, got {z}.")
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 1:
        return np.array([z])
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _lipschitz_constant(A: np.ndarray, iters: int = 40) -> float:
    """Lipschitz constant of grad ``||A w - b||^2`` = ``2 * lambda_max(A'A)``.

    ``lambda_max`` is estimated by power iteration on ``A'A``.
    """
    n = A.shape[1]
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    nx = np.linalg.norm(x)
    if nx < _EPS:
        return 1.0
    x /= nx
    lam = 1.0
    for _ in range(iters):
        y = A.T @ (A @ x)
        ny = np.linalg.norm(y)
        if ny < _EPS:
            return _EPS
        x = y / ny
        lam = float(x @ (A.T @ (A @ x)))
    return 2.0 * lam + _EPS


def simplex_lstsq(
    A: np.ndarray,
    b: np.ndarray,
    *,
    max_iter: int = 2000,
    tol: float = 1e-9,
    warn: bool = False,
) -> np.ndarray:
    """Minimize ``||A w - b||^2`` over the probability simplex via FISTA.

    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape ``(m, n)``.
    b : np.ndarray
        Target vector, shape ``(m,)``.
    max_iter, tol : int, float
        Stopping controls.
    warn : bool
        If ``True``, emit a :class:`RuntimeWarning` when ``max_iter`` is
        exhausted before the step norm falls below ``tol`` (i.e. FISTA did not
        converge). Off by default so the inner-loop callers stay silent.

    Returns
    -------
    np.ndarray
        Weights of shape ``(n,)`` on the simplex.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]
    if n == 1:
        return np.array([1.0])

    step = 1.0 / _lipschitz_constant(A)
    w = np.full(n, 1.0 / n)
    z = w.copy()
    t = 1.0
    AtA = A.T @ A
    Atb = A.T @ b
    for _ in range(max_iter):
        grad = 2.0 * (AtA @ z - Atb)
        w_new = project_simplex(z - step * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.linalg.norm(w_new - w) < tol:
            return w_new
        w, t = w_new, t_new
    if warn:
        warnings.warn(
            f"simplex_lstsq did not converge within max_iter={max_iter} "
            f"(tol={tol}); returned weights may be sub-optimal.",
            RuntimeWarning,
            stacklevel=2,
        )
    return w


def mspe(y1: np.ndarray, Y0: np.ndarray, w: np.ndarray) -> float:
    """Mean squared prediction error ``mean((y1 - Y0 w)^2)``."""
    resid = y1 - Y0 @ w
    return float(np.mean(resid ** 2))
