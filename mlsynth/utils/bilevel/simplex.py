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
from typing import Optional

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


def project_simplex_cols(V: np.ndarray, z: float = 1.0, *,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
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
    mask : np.ndarray, optional
        Boolean ``(n, k)``. Where ``True``, the coordinate is excluded: it comes
        back exactly zero and takes no part in locating the threshold, so the
        result is the projection of the remaining coordinates onto their own
        simplex. This is how a family of leave-one-out programs is run as one
        batch -- see :func:`simplex_lstsq_loo`.

    Returns
    -------
    np.ndarray
        Shape ``(n, k)``, every column on the simplex.

    Notes
    -----
    Exclusion is implemented by replacing a masked entry with a value provably
    below that column's threshold rather than by a fixed sentinel. Since every
    projected weight is at most ``z``, the threshold satisfies
    ``theta >= max(v_kept) - z``, so ``max(v_kept) - z - 1`` is always strictly
    below it. A hard-coded constant would instead be a latent bug: large enough
    for one panel's units and too small for another's.
    """
    if z <= 0:
        raise ValueError(f"simplex radius z must be positive, got {z}.")
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D (n, k); got shape {V.shape}.")
    n, k = V.shape
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != V.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match V's {V.shape}.")
        kept = ~mask
        if not kept.any(axis=0).all():
            dead = np.flatnonzero(~kept.any(axis=0))[:5].tolist()
            raise ValueError(
                f"mask excludes every coordinate of column(s) {dead}; there is "
                f"no projection onto an empty simplex.")
        floor = np.where(kept, V, -np.inf).max(axis=0) - z - 1.0
        V = np.where(mask, floor[None, :], V)
    if n == 1:
        return np.full((1, k), z)
    U = -np.sort(-V, axis=0)                       # descending within column
    cssv = np.cumsum(U, axis=0) - z
    ind = np.arange(1, n + 1)[:, None]
    cond = (U - cssv / ind) > 0                    # a True block, then False
    rho = n - 1 - np.argmax(cond[::-1], axis=0)    # index of the last True
    theta = cssv[rho, np.arange(k)] / (rho + 1.0)
    W = np.maximum(V - theta, 0.0)
    if mask is not None:
        W[mask] = 0.0                              # exactly, not to a tolerance
    return W


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


def simplex_lstsq_loo(
    M: np.ndarray,
    n_targets: Optional[int] = None,
    *,
    ridge: float = 0.0,
    max_iter: int = 20000,
    tol: float = 1e-11,
    check_every: int = 25,
) -> np.ndarray:
    """Leave-one-out simplex least squares over the columns of ``M``.

    For each ``j < n_targets`` solves

        ``min ||M[:, k != j] w - M[:, j]||^2``  s.t. ``w >= 0``, ``sum(w) = 1``

    as one batched program rather than ``n_targets`` separate ones. This is the
    shape of an in-space placebo test: every donor is cast as treated in turn
    against the others.

    The saving is structural rather than incidental. Every subproblem's design
    is ``M`` with one column deleted, so every subproblem's Gram is a submatrix
    of ``M'M``; and every subproblem's target is itself a column of ``M``, so
    the cross term ``A'b`` is a column of that same Gram. One ``M'M`` therefore
    supplies the whole family, and no product with the data appears inside the
    iteration at all. On the Walmart panel's cohort shape (7 pre-treatment rows,
    39 donors) that is 18.0s of looping against 0.65s here, objectives within
    0.01 percent.

    Parameters
    ----------
    M : np.ndarray
        Columns to fit against one another, shape ``(m, n)`` with ``n >= 2``.
    n_targets : int, optional
        How many of ``M``'s leading columns are cast as treated. Columns past
        it are donors to every subproblem but are never themselves a target --
        the border a permutation placebo test needs, where the treated unit
        joins each placebo's pool without being a placebo. Defaults to ``n``.
    ridge : float
        L2 penalty on the Gram diagonal. Not a solver tolerance -- it changes
        the estimand, shrinking the weights toward uniform.
    max_iter, tol, check_every : int, float, int
        Stopping controls; objectives are tested every ``check_every`` steps,
        as in :func:`simplex_lstsq_batch` and for the same reason.

    Returns
    -------
    np.ndarray
        Shape ``(n, n_targets)``. Column ``j`` holds the weights fitting
        ``M[:, j]``, with ``W[j, j]`` exactly zero.

    See Also
    --------
    simplex_lstsq_batch : many targets against a *shared* design.

    Warnings
    --------
    Unlike :func:`simplex_lstsq_batch`, convergence is judged per column and the
    iteration continues until every column has settled. Testing the summed
    objective instead would let a column that has plateaued be stopped early
    while its neighbours are still improving; on a rank-deficient design that
    cost up to 3 percent of the fit on individual columns.

    Notes
    -----
    On a rank-deficient design -- fewer rows than columns, which is the usual
    case here -- the optimum is a face rather than a point, so this function and
    a loop over :func:`simplex_lstsq` reach the same objective and generally
    different weights. Neither is wrong: where on the face a solver lands is not
    identified by the data. Compare losses, not weights, and read averages over
    the family rather than an individual column.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D (m, n); got shape {M.shape}.")
    n = M.shape[1]
    if n < 2:
        raise ValueError(
            f"M has {n} column(s); leaving one out needs at least two, since "
            f"the remaining columns are what the left-out one is fitted "
            f"against.")
    if n_targets is None:
        n_targets = n
    n_targets = int(n_targets)
    if not 1 <= n_targets <= n:
        raise ValueError(
            f"n_targets must be between 1 and M's column count {n}; "
            f"got {n_targets}.")

    G = M.T @ M
    if ridge:
        G = G + float(ridge) * np.eye(n)
    C = G[:, :n_targets]              # M' M[:, j] for every target, for free
    bb = np.diag(G)[:n_targets]       # ||M[:, j]||^2, per column

    mask = np.zeros((n, n_targets), dtype=bool)
    diag = np.arange(n_targets)
    mask[diag, diag] = True           # column j is not its own donor

    lam = float(np.linalg.eigvalsh(G)[-1])
    step = 1.0 / (2.0 * lam + _EPS)
    # A shared step is safe: deleting a column cannot raise the largest
    # eigenvalue, so this bound holds for every subproblem in the family.

    W = np.where(mask, 0.0, 1.0 / (n - 1))
    Z = W.copy()
    t = 1.0
    prev = None
    for it in range(max_iter):
        W_new = project_simplex_cols(Z - step * 2.0 * (G @ Z - C), mask=mask)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = W_new + ((t - 1.0) / t_new) * (W_new - W)
        W, t = W_new, t_new
        if (it + 1) % check_every == 0:
            obj = (((G @ W) * W).sum(axis=0) - 2.0 * (C * W).sum(axis=0) + bb)
            if prev is not None and np.all(
                    prev - obj <= tol * np.maximum(1.0, np.abs(prev))):
                break
            prev = obj
    return W


def mspe(y1: np.ndarray, Y0: np.ndarray, w: np.ndarray) -> float:
    """Mean squared prediction error ``mean((y1 - Y0 w)^2)``."""
    resid = y1 - Y0 @ w
    return float(np.mean(resid ** 2))


def _project_simplex_rows(V: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean projection onto ``{w >= 0, sum(w) = 1}``.

    The row-wise twin of :func:`project_simplex`, and exact in the same sense:
    the same sort-based algorithm, applied along ``axis=1`` with the per-row
    threshold located by one ``argmax`` over the reversed condition array
    instead of a Python loop over rows.
    """
    M, n = V.shape
    if n == 1:
        return np.ones((M, 1))
    U = -np.sort(-V, axis=1)
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, n + 1)
    cond = U - cssv / ind > 0
    # Last True per row, matching project_simplex's ``ind[cond][-1]``.
    rho = n - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = cssv[np.arange(M), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def _lipschitz_rows(AtA: np.ndarray, iters: int = 40) -> np.ndarray:
    """Per-problem ``2 * lambda_max(A'A)``, by power iteration on the stack.

    Mirrors :func:`_lipschitz_constant`, including its seeded start vector, so
    a one-problem stack takes the same step size the scalar solver would.
    """
    M, n, _ = AtA.shape
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=n)
    nx = np.linalg.norm(x0)
    if nx < _EPS:  # pragma: no cover - the seeded start vector is never zero
        return np.ones(M)
    x = np.broadcast_to(x0 / nx, (M, n)).copy()
    lam = np.ones(M)
    dead = np.zeros(M, dtype=bool)
    for _ in range(iters):
        v = np.einsum("mij,mj->mi", AtA, x)
        nv = np.linalg.norm(v, axis=1)
        dead = nv < _EPS
        x = v / np.where(dead, 1.0, nv)[:, None]
        lam = np.einsum("mi,mij,mj->m", x, AtA, x)
    return np.where(dead, _EPS, 2.0 * lam + _EPS)


def simplex_lstsq_gram_many(
    AtA: np.ndarray,
    Atb: np.ndarray,
    *,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> np.ndarray:
    """``min_w ||A_m w - b||^2`` over the simplex, for a stack of Gram systems.

    The many-*design* counterpart to :func:`simplex_lstsq_batch`, which shares
    one design across many targets. Here each problem brings its own
    ``A_m'A_m`` and ``A_m'b``, which is the shape a forward-selection scan
    produces: one candidate design per remaining donor, all against the same
    treated unit.

    Taking Grams rather than designs is what makes the stacking possible --
    FISTA never refers to ``A`` itself -- and it is also what makes it cheap for
    the caller, since the candidate Grams are submatrices of one donor Gram.

    Each problem is frozen at its own stopping iterate, so a one-problem stack
    reproduces :func:`simplex_lstsq` exactly.

    Parameters
    ----------
    AtA : np.ndarray
        Gram stack, shape ``(M, n, n)``. A bare ``(n, n)`` is read as ``M = 1``.
    Atb : np.ndarray
        Target stack, shape ``(M, n)``. A bare ``(n,)`` is read as ``M = 1``.
    max_iter, tol : int, float
        Stopping controls, matching :func:`simplex_lstsq`.

    Returns
    -------
    np.ndarray
        Weights of shape ``(M, n)``, each row on the simplex.

    Raises
    ------
    MlsynthEstimationError
        If the stacks are not conformable or hold non-finite values.
    """
    from mlsynth.exceptions import MlsynthEstimationError

    AtA = np.asarray(AtA, dtype=float)
    Atb = np.asarray(Atb, dtype=float)
    if AtA.ndim == 2:
        AtA = AtA[None]
    if Atb.ndim == 1:
        Atb = Atb[None]
    if AtA.ndim != 3 or Atb.ndim != 2:
        raise MlsynthEstimationError(
            f"AtA must be 3-D and Atb 2-D; got {AtA.ndim}-D and {Atb.ndim}-D."
        )
    if AtA.shape[1] != AtA.shape[2]:
        raise MlsynthEstimationError(
            f"each Gram must be square; got {AtA.shape[1]}x{AtA.shape[2]}."
        )
    if AtA.shape[0] != Atb.shape[0]:
        raise MlsynthEstimationError(
            "AtA and Atb must hold the same number of problems; got "
            f"{AtA.shape[0]} and {Atb.shape[0]}."
        )
    if AtA.shape[1] != Atb.shape[1]:
        raise MlsynthEstimationError(
            f"Atb width {Atb.shape[1]} does not match the Gram width "
            f"{AtA.shape[1]}."
        )
    if not (np.isfinite(AtA).all() and np.isfinite(Atb).all()):
        raise MlsynthEstimationError("AtA and Atb must be finite.")

    M, n, _ = AtA.shape
    if M == 0:
        return np.empty((0, n))
    if n == 1:
        return np.ones((M, 1))

    step = (1.0 / _lipschitz_rows(AtA))[:, None]
    w = np.full((M, n), 1.0 / n)
    z = w.copy()
    out = w.copy()
    t = 1.0
    active = np.ones(M, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        grad = 2.0 * (np.einsum("mij,mj->mi", AtA[idx], z[idx]) - Atb[idx])
        w_new = _project_simplex_rows(z[idx] - step[idx] * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z[idx] = w_new + ((t - 1.0) / t_new) * (w_new - w[idx])
        settled = np.linalg.norm(w_new - w[idx], axis=1) < tol
        out[idx] = w_new
        w[idx] = w_new
        active[idx[settled]] = False
        t = t_new
    return out
