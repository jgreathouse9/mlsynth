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


def project_simplex_cols(V: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Project every COLUMN of ``V`` onto ``{w >= 0, sum(w) = z}``.

    The column-wise form of :func:`project_simplex`, and exact in the same
    sense: same sort-based algorithm (Held, Wolfe & Crowder 1974; Duchi et al.
    2008), just carried out along ``axis=0`` with the per-column threshold
    located by a single ``argmax`` over the reversed condition array instead of
    a Python loop over columns.

    Parameters
    ----------
    V : np.ndarray
        Points to project, shape ``(n, k)`` -- one column per problem.
    z : float
        Simplex radius; each column sums to this.

    Returns
    -------
    np.ndarray
        Shape ``(n, k)``, every column on the simplex.
    """
    if z <= 0:
        raise ValueError(f"simplex radius z must be positive, got {z}.")
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D (n, k); got shape {V.shape}.")
    n, k = V.shape
    if n == 1:
        return np.full((1, k), z)
    U = -np.sort(-V, axis=0)                       # descending within column
    cssv = np.cumsum(U, axis=0) - z
    ind = np.arange(1, n + 1)[:, None]
    cond = (U - cssv / ind) > 0                    # a True block, then False
    rho = n - 1 - np.argmax(cond[::-1], axis=0)    # index of the last True
    theta = cssv[rho, np.arange(k)] / (rho + 1.0)
    return np.maximum(V - theta, 0.0)


def simplex_lstsq_batch(
    A: np.ndarray,
    B: np.ndarray,
    *,
    ridge: float = 0.0,
    max_iter: int = 20000,
    tol: float = 1e-11,
    check_every: int = 25,
) -> np.ndarray:
    """``min_W ||A W - B||_F^2`` with every column of ``W`` on the simplex.

    One program with many right-hand sides, not many programs. ``A`` is shared,
    so the Gram matrix, the ridge and the step size are formed once and
    amortised over ``B``'s columns; each FISTA iteration is then a single
    ``(n, n) @ (n, k)`` matmul plus a column-wise projection. This is the shape
    a stacked synthetic-control design has whenever the treated units in a
    cohort share a donor pool.

    Parameters
    ----------
    A : np.ndarray
        Shared design matrix, shape ``(m, n)``.
    B : np.ndarray
        Targets, shape ``(m, k)``; a 1-D array is treated as a single column.
    ridge : float
        L2 penalty added to the Gram diagonal. Not a solver tolerance -- it
        changes the estimand, shrinking the weights toward uniform.
    max_iter, tol, check_every : int, float, int
        Stopping controls; the objective is tested every ``check_every`` steps.

    Returns
    -------
    np.ndarray
        Shape ``(n, k)``, every column on the simplex.

    Notes
    -----
    Convergence is judged on the objective, unlike :func:`simplex_lstsq`, which
    stops on the step norm. The difference matters when ``A`` is rank
    deficient: the optimum is then a face rather than a point, and the iterate
    keeps drifting along it long after the loss has settled, so a step-norm
    rule never fires and the solver runs to ``max_iter`` every time.

    A consequence worth knowing before comparing the two: on such a design this
    function and a loop over :func:`simplex_lstsq` reach the same objective but
    generally *different* weights. Neither is wrong -- where on the optimal face
    a solver lands is not identified by the data. Compare losses, not weights.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError(
            f"B must be (m, k) with m={A.shape[0]} to match A; got {B.shape}."
        )
    n, k = A.shape[1], B.shape[1]
    if n == 1:
        return np.ones((1, k))

    G = A.T @ A
    if ridge:
        G = G + float(ridge) * np.eye(n)
    C = A.T @ B
    bb = float((B * B).sum())
    lam = float(np.linalg.eigvalsh(G)[-1])
    step = 1.0 / (2.0 * lam + _EPS)

    W = np.full((n, k), 1.0 / n)
    Z = W.copy()
    t = 1.0
    prev = None
    for it in range(max_iter):
        W_new = project_simplex_cols(Z - step * 2.0 * (G @ Z - C))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = W_new + ((t - 1.0) / t_new) * (W_new - W)
        W, t = W_new, t_new
        if (it + 1) % check_every == 0:
            obj = float(((G @ W) * W).sum() - 2.0 * (C * W).sum() + bb)
            if prev is not None and prev - obj <= tol * max(1.0, abs(prev)):
                break
            prev = obj
    return W


def mspe(y1: np.ndarray, Y0: np.ndarray, w: np.ndarray) -> float:
    """Mean squared prediction error ``mean((y1 - Y0 w)^2)``."""
    resid = y1 - Y0 @ w
    return float(np.mean(resid ** 2))
