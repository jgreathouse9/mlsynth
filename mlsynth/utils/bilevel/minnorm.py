"""Exact simplex least squares in Gram form, solved for a whole batch at once.

The problem is the ordinary synthetic-control weight program,

    min_w  ||B w - A||^2   s.t.   w >= 0,  sum(w) = 1,

and the observation this module is built on is that the equality constraint
removes the design matrix from it. Since ``sum(w) = 1``,

    B w - A = B w - A (1' w) = (B - A 1') w,

so with ``R = B - A 1'`` and ``G = R' R`` the objective is the homogeneous
quadratic form ``w' G w``: geometrically, the point of least norm in the convex
hull of the columns of ``R`` -- each column being one donor's discrepancy from
the treated unit. Wolfe (1976) solves that by an active set over the vertices,
finite and exact.

Two things follow, and both matter for the MSCMT outer search, which solves the
same program tens of thousands of times over a slowly-moving population of
predictor weightings ``V``:

* The whole problem is carried by ``G``, a ``(J, J)`` matrix. Under a
  ``V``-weighted metric ``G(V) = sum_k V_k r_k r_k'`` is linear in ``V``, so a
  whole generation of Grams is one matrix product against the ``K`` rank-one
  pieces ``r_k r_k'``, formed once. No product with the data appears inside the
  search at all.
* The active set runs the entire generation in lockstep. Each iteration is one
  batched linear solve over the current supports, so a population of two hundred
  candidates costs what the hardest single candidate costs, not two hundred
  times what the average one costs.

Wolfe, P. (1976). Finding the nearest point in a polytope. Mathematical
Programming, 11, 128-149. https://doi.org/10.1007/BF01580381

See Also
--------
mlsynth.utils.bilevel.active_set.solve_simplex_qp : the single-problem,
    design-matrix active set. Same optimum; preferred when there is one problem
    and the design is at hand, since it factors ``B`` directly instead of
    forming ``G``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Ridge on the *scale-normalised* corral system. It exists to keep the batched
# LU factorisation defined when the donors in a support are affinely dependent
# (duplicate or collinear donors, or fewer matching rows than donors), where the
# optimum is a face and the system is singular. Among the optima on that face it
# selects the one of least norm. It must stay well below _KKT_TOL: the
# optimality test is applied to the unperturbed G, and a ridge above the test's
# tolerance would keep the test failing and the active set growing forever.
_RIDGE = 1e-12
# Optimality tolerance, relative to the problem's own magnitude (mean diagonal
# of G, i.e. the mean squared donor discrepancy).
_KKT_TOL = 1e-10
# Weights live on the simplex, so this one is absolute.
_W_TOL = 1e-12


def simplex_gram(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Gram matrix ``G`` with ``w' G w == ||B w - A||^2`` on the simplex.

    Parameters
    ----------
    B : np.ndarray, shape (m, J)
        Donor / design matrix.
    A : np.ndarray, shape (m,)
        Treated unit's target vector.

    Returns
    -------
    np.ndarray, shape (J, J)
        ``R' R`` for ``R = B - A 1'``, symmetric positive semi-definite.

    Notes
    -----
    The identity holds only where ``sum(w) = 1``; off the simplex ``w'Gw`` and
    ``||Bw - A||^2`` differ. Every caller here is solving over the simplex.
    """
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    if B.ndim != 2:
        raise ValueError(f"B must be a 2-D (m, J) matrix; got shape {B.shape}.")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"len(A)={A.shape[0]} must equal B's row count {B.shape[0]}.")
    R = B - A[:, None]
    G = R.T @ R
    return 0.5 * (G + G.T)          # symmetric to the last bit


def _validate(G: np.ndarray, ndim: int) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    if G.ndim != ndim:
        raise ValueError(
            f"G must be {ndim}-D "
            f"{'(J, J)' if ndim == 2 else '(S, J, J)'}; got shape {G.shape}.")
    if G.shape[-1] != G.shape[-2]:
        raise ValueError(f"G must be square in its last two axes; got {G.shape}.")
    if G.shape[-1] == 0:
        raise ValueError("G has no columns: at least one donor is required.")
    if not np.all(np.isfinite(G)):
        raise ValueError("G contains non-finite entries.")
    return G


def solve_simplex_minnorm(
    G: np.ndarray,
    *,
    warm_start: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    return_info: bool = False,
):
    """Minimise ``w' G w`` over ``w >= 0, sum(w) == 1`` for one Gram matrix.

    A thin wrapper over :func:`solve_simplex_minnorm_batch`; see it for the
    algorithm and for the meaning of ``warm_start`` and ``max_iter``.

    Parameters
    ----------
    G : np.ndarray, shape (J, J)
        Symmetric positive semi-definite Gram matrix, e.g. from
        :func:`simplex_gram`.
    warm_start : np.ndarray, shape (J,), optional
        Feasible starting weights.
    max_iter : int, optional
        Cap on active-set iterations.
    return_info : bool
        If ``True`` also return ``{"iterations", "converged"}``.

    Returns
    -------
    np.ndarray, shape (J,)
        The optimal weights, or ``(w, info)`` when ``return_info=True``.
    """
    G = _validate(G, ndim=2)
    warm = None if warm_start is None else np.asarray(
        warm_start, dtype=float).reshape(1, -1)
    out = solve_simplex_minnorm_batch(
        G[None], warm_start=warm, max_iter=max_iter, return_info=return_info)
    if not return_info:
        return out[0]
    W, info = out
    return W[0], {"iterations": info["iterations"],
                  "converged": bool(info["converged"][0])}


def solve_simplex_minnorm_batch(
    G: np.ndarray,
    *,
    warm_start: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    return_info: bool = False,
):
    """Minimise ``w' G_s w`` on the simplex for every ``G_s`` in a stack.

    Wolfe's minimum-norm-point active set, run over the whole stack in lockstep:
    every candidate holds a *corral* (its current support), and each iteration
    solves all the corral systems as one batched LU, takes each candidate's
    step, and drops the candidates whose optimality test has passed. Iterations
    are therefore set by the hardest member of the batch, and the arrays shrink
    as members certify.

    Parameters
    ----------
    G : np.ndarray, shape (S, J, J)
        Symmetric positive semi-definite Gram matrices, one per problem.
    warm_start : np.ndarray, shape (S, J), optional
        Feasible starting weights, one row per problem -- the previous
        generation's solution, when the population moves slowly. A row that is
        not usable (negative, all-zero, non-finite), or a whole array of the
        wrong shape, is discarded in favour of the cold start; a warm start
        changes only how much work the solve costs.
    max_iter : int, optional
        Cap on active-set iterations. Defaults to ``max(50, 10 * J)``.
    return_info : bool
        If ``True`` also return ``{"iterations", "converged"}``, where
        ``converged`` is a length-``S`` boolean array. A candidate that hits
        ``max_iter`` comes back ``False`` with its current feasible iterate, so
        an exhausted budget is visible to the caller instead of passing as an
        optimum.

    Returns
    -------
    np.ndarray, shape (S, J)
        Optimal weights, one row per problem, or ``(W, info)`` when
        ``return_info=True``.

    Notes
    -----
    Where several weight vectors attain the minimum -- collinear donors, or more
    donors than matching rows, so the optimal set is a face and not a point --
    which of them is returned depends on the starting corral, and so on the warm
    start. The objective does not. Compare losses, not weights.
    """
    G = _validate(G, ndim=3)
    S, J = G.shape[0], G.shape[-1]
    if max_iter is None:
        max_iter = max(50, 10 * J)
    max_iter = int(max_iter)

    gdiag = np.einsum("sjj->sj", G)
    scale = gdiag.mean(axis=1)
    if J == 1:
        W = np.ones((S, 1))
        return (W, {"iterations": 0, "converged": np.ones(S, bool)}) \
            if return_info else W

    # Cold start: the single best donor. A vertex is always feasible and is the
    # optimum outright whenever the treated unit is nearest one donor.
    W = np.zeros((S, J))
    W[np.arange(S), np.argmin(gdiag, axis=1)] = 1.0
    if warm_start is not None:
        ws = np.asarray(warm_start, dtype=float)
        if ws.shape == (S, J):
            total = ws.sum(axis=1)
            usable = (np.all(np.isfinite(ws), axis=1) & (ws.min(axis=1) >= -_W_TOL)
                      & (total > _W_TOL))
            if usable.any():
                W[usable] = np.maximum(ws[usable], 0.0) / total[usable, None]
    corral = W > _W_TOL

    # A Gram with a zero diagonal is the zero matrix (it is PSD), so the
    # objective is flat and every simplex point is optimal, including the start.
    # These are held out of the loop: their corral system carries no information
    # and would be singular.
    live = scale > 0.0
    converged = ~live

    rows = np.arange(J)
    iterations = 0
    for _ in range(max_iter):
        a = np.flatnonzero(live)
        if a.size == 0:
            break
        iterations += 1
        m = corral[a]
        counts = m.sum(axis=1)
        c = int(counts.max())
        # Compact the corral: column ``i`` of ``idx`` holds each candidate's
        # i-th corral member, so the systems below are (c, c) with c the largest
        # corral in the batch -- a handful of donors, not J of them.
        idx = np.argsort(~m, axis=1, kind="stable")[:, :c]
        valid = np.arange(c)[None, :] < counts[:, None]

        # Corral system G_SS z = 1, scaled by the problem's magnitude so the
        # ridge below is a fixed relative perturbation. Padding columns get an
        # identity row and a zero right-hand side, so they solve to zero.
        Gc = G[a[:, None, None], idx[:, :, None], idx[:, None, :]]
        Gc = np.where(valid[:, :, None] & valid[:, None, :], Gc, 0.0)
        Gc /= scale[a][:, None, None]
        di = np.arange(c)
        Gc[:, di, di] = np.where(valid, Gc[:, di, di] + _RIDGE, 1.0)
        z = np.linalg.solve(Gc, valid.astype(float)[..., None])[..., 0]
        # The affine minimiser over the corral. z sums to zero only if the
        # corral system is degenerate beyond the ridge's reach; keep the current
        # iterate there and let the optimality test decide.
        zs = z.sum(axis=1)
        alpha = np.where(np.abs(zs)[:, None] < 1e-300, 0.0, z / zs[:, None])

        lam = np.where(valid, W[a[:, None], idx], 0.0)
        interior = np.where(valid, alpha, np.inf).min(axis=1) > _W_TOL

        # Blocked: move from the current iterate toward the affine minimiser
        # until the first corral weight reaches zero, and drop it.
        block = valid & (alpha <= _W_TOL)
        ratios = np.where(block, lam / np.maximum(lam - alpha, 1e-300), np.inf)
        theta = ratios.min(axis=1)
        theta = np.where(np.isfinite(theta), theta, 0.0)[:, None]
        lam_blocked = np.maximum((1.0 - theta) * lam + theta * alpha, 0.0)
        dropped = valid & (lam_blocked <= _W_TOL)
        lam_blocked = np.where(dropped, 0.0, lam_blocked)

        W[a[:, None], idx] = np.where(interior[:, None],
                                      np.maximum(alpha, 0.0), lam_blocked)
        corral[a[:, None], idx] = np.where(interior[:, None], valid,
                                           valid & ~dropped)

        # Optimality, for the candidates that reached their corral's minimum:
        # with nu = w'Gw the multiplier on sum(w) = 1, a donor outside the
        # corral improves the fit exactly when (Gw)_j < nu. Bring in the most
        # improving one, or certify. Only the corral columns of G are touched,
        # since w is zero elsewhere.
        f = np.flatnonzero(interior)
        if f.size:
            af, idf = a[f], idx[f]
            lf = np.maximum(alpha[f], 0.0)
            Gcol = G[af[:, None, None], rows[None, :, None], idf[:, None, :]]
            g = np.einsum("sjc,sc->sj", Gcol, lf)
            nu = np.einsum("sc,sc->s", np.take_along_axis(g, idf, axis=1), lf)
            jstar = np.argmin(g, axis=1)
            improves = g[np.arange(f.size), jstar] < nu - _KKT_TOL * scale[af]
            vi = np.flatnonzero(improves)
            if vi.size:
                corral[af[vi], jstar[vi]] = True
            done = af[~improves]
            live[done] = False
            converged[done] = True

    W = np.maximum(W, 0.0)
    W /= W.sum(axis=1, keepdims=True)
    if return_info:
        return W, {"iterations": iterations, "converged": converged}
    return W
