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

from typing import Callable, Optional

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


def gram_reduction_is_safe(B: np.ndarray, tol: float = 1e-8) -> bool:
    """Whether ``B`` may be replaced by its Gram without changing the answer.

    Forming ``G`` squares the design's condition number, and that is only free
    where the design has full column rank. Two things go wrong when it does not,
    and they are different failures:

    * If the design is rank deficient only because of a small regulariser -- a
      uniqueness ridge, a near-zero penalty at the end of a grid -- then that
      regulariser is the only thing separating the columns, and squaring puts it
      below what float64 resolves. The Gram solve then returns a point that is
      not optimal at all. Measured on mlSC's penalty grid over a 9-period,
      12-disaggregate design, it finished 225 percent above the optimum.
    * If the design is rank deficient and the minimiser is a face and not a
      point, both solvers are correct and they land in different places: on an
      8-row, 39-donor design whose target the donors reproduce exactly, they
      agree on the fitted values to 1e-11 while differing by 0.19 in the weights.
      Nothing is wrong, but anything reading the weights -- a donor table, a
      counterfactual built from post-period donor outcomes -- would change.

    So the reduction is for designs this returns ``True`` for, and it is a
    property of the design alone, decidable before any solving.

    For the second failure it is sound but far from tight, and the slack is not
    small. Rank deficiency is only a precondition for a non-unique minimiser:
    the objective is flat along a direction only if that direction is also
    *feasible* where the solution sits, which needs a support large relative to
    the design's rank. Synthetic-control solutions are sparse, so a donor pool
    wider than the pre-period usually still has a unique minimiser -- on the
    Proposition 99 placebo family all 38 solves do, at 37 donors against 19
    pre-periods, and this test admits none of them. A caller that can afford to
    solve first should ask :func:`simplex_optimum_is_unique` instead, which
    settles the same question exactly, on the support.

    Parameters
    ----------
    B : np.ndarray, shape (m, J)
        The design the reduction would be applied to.
    tol : float
        Smallest acceptable ratio of the smallest to the largest singular value.

    Returns
    -------
    bool
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] < B.shape[1]:
        return False
    sv = np.linalg.svd(B, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0.0:
        return False
    return bool(sv[-1] / sv[0] > tol)


def ridged_gram_reduction_is_safe(
    design: np.ndarray, ridge: float, tol: float = 1e-8
) -> bool:
    """:func:`gram_reduction_is_safe` for ``[design; sqrt(ridge) I]``, usually
    without factorising it.

    A caller that folds an L2 penalty into a least-squares program by stacking
    ``sqrt(ridge) * I`` beneath the design is asking the guard about a matrix it
    can describe rather than one it has to look at. The augmented Gram is
    ``design' design + ridge I``, so

    * ``lambda_min >= ridge``, since ``design' design`` is positive
      semidefinite, and
    * ``lambda_max <= ||design||_F^2 + ridge``, since the trace bounds the
      largest eigenvalue,

    giving ``sv_min / sv_max >= sqrt(ridge / (||design||_F^2 + ridge))``. That
    costs one Frobenius norm and settles the question whenever it clears
    ``tol``.

    The bound is one-sided in the safe direction. Clearing it *proves* the
    answer is ``True``; failing to clear it proves nothing, so the spectrum is
    computed as before. The decisions are therefore identical to asking
    :func:`gram_reduction_is_safe` about the augmented matrix -- identical by
    construction, not by choosing a tolerance that happens to agree.

    Where it pays: SDID's placebo loop poses 1000 of these per fit at ``B=500``,
    and on a 101x120 panel the ridged half of them had a true ratio of 7.43e-2
    against a bound of 7.21e-2 -- tight to three percent, and six orders of
    magnitude clear of ``tol``. The unridged half gets no help, because at
    ``ridge = 0`` the lower bound is vacuous and the design's conditioning
    genuinely has to be measured.

    Parameters
    ----------
    design : np.ndarray, shape (m, J)
        The design *before* augmentation, centred if the caller centres it.
    ridge : float
        Non-negative penalty coefficient. ``0`` defers to
        :func:`gram_reduction_is_safe` on ``design`` itself.
    tol : float
        Smallest acceptable ratio of the smallest to the largest singular value.

    Returns
    -------
    bool
    """
    design = np.asarray(design, dtype=float)
    if design.ndim != 2:
        raise ValueError(
            f"design must be a 2-D (m, J) matrix; got shape {design.shape}.")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError(f"ridge must be finite and non-negative; got {ridge!r}.")
    if ridge == 0.0:
        return gram_reduction_is_safe(design, tol)
    if min(design.shape) == 0:
        return False
    flat = design.ravel()
    frob_sq = float(flat @ flat)
    if not np.isfinite(frob_sq):
        return False
    if np.sqrt(ridge / (frob_sq + ridge)) > tol:
        return True
    J = design.shape[1]
    return gram_reduction_is_safe(
        np.vstack([design, np.sqrt(ridge) * np.eye(J)]), tol)


def simplex_point_is_optimal(
    B: np.ndarray, A: np.ndarray, w: np.ndarray, tol: float = 1e-7
) -> bool:
    """Whether ``w`` solves ``min ||B w - A||^2`` on the simplex, checked on ``B``.

    With ``g = B'(B w - A)`` and ``nu = g' w``, the KKT conditions for this
    program are ``g_j >= nu`` for every donor, with equality wherever ``w_j > 0``.
    Feasibility plus that inequality is a certificate: it is computed from the
    design and the target, so it holds whatever solver produced ``w`` and
    whatever form that solver worked in.

    Which is the point of it here. The Gram reduction squares the condition
    number -- ``G = R'R`` for ``R = B - A 1'`` -- so a design that is merely
    awkward at ``cond(B) ~ 1e7``, as covariates measured in different units are,
    gives a ``G`` that is singular in float64. The batched solver then converges
    on that ``G``, correctly, to a point that does not solve the program ``G``
    came from, and nothing computed from ``G`` can tell. On the covariate
    specification of the Wiltshire panel that is 62 of 76 members of a cohort,
    with a KKT residual of 7e-3 where the design-form solver leaves 6e-10.

    Optimality and uniqueness are separate questions, and standing one solver in
    for another needs both: see :func:`simplex_optimum_is_unique`. A point may be
    optimal on a face, or be a near-miss for a unique optimum.

    Parameters
    ----------
    B : np.ndarray, shape (m, J)
        The design the weights are claimed to solve.
    A : np.ndarray, shape (m,)
        The target.
    w : np.ndarray, shape (J,)
        The candidate weights.
    tol : float
        Relative tolerance, applied to the reduced gradient's own magnitude, and
        absolutely to simplex feasibility.

    Returns
    -------
    bool
        ``True`` only when ``w`` is feasible and satisfies the KKT conditions.
    """
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if B.ndim != 2:
        raise ValueError(f"B must be a 2-D (m, J) matrix; got shape {B.shape}.")
    if w.shape[0] != B.shape[1]:
        raise ValueError(
            f"len(w)={w.shape[0]} must equal B's column count {B.shape[1]}.")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"len(A)={A.shape[0]} must equal B's row count {B.shape[0]}.")
    if not np.all(np.isfinite(w)):
        return False
    if w.min() < -_W_TOL or abs(w.sum() - 1.0) > 1e-9:
        return False
    g = B.T @ (B @ w - A)
    if not np.all(np.isfinite(g)):  # pragma: no cover - overflow on absurd input
        return False
    nu = float(g @ w)
    scale = 1.0 + float(np.max(np.abs(g)))
    return bool(float(g.min()) >= nu - tol * scale)


def simplex_optimum_is_unique(
    B: np.ndarray, A: np.ndarray, w: np.ndarray, tol: float = 1e-7
) -> bool:
    """Whether ``min ||B w - A||^2`` on the simplex has one minimiser, given one.

    Two exact solvers of this program return the same weights when the minimiser
    is a point and different weights when it is a face, so this is the question
    that decides whether one may stand in for the other -- and it is answerable
    after solving, cheaply, on the solution's own support.

    The objective is flat along a direction ``d`` only if ``B d = 0``, and ``d``
    stays feasible only if ``1' d = 0`` and ``d`` does not push a coordinate
    below zero. A donor outside the support with a strictly positive reduced
    gradient cannot be entered at all, so only the weakly-active set can move.
    The minimiser is therefore unique exactly when no such ``d`` exists, i.e.
    when ``B`` restricted to that set, with the sum-to-one row appended, has full
    column rank.

    This is what :func:`gram_reduction_is_safe` approximates from the design
    alone, and it approximates it conservatively: full column rank of the whole
    design implies this, but not the reverse. The gap is the ordinary
    synthetic-control geometry -- more donors than pre-treatment periods -- where
    the fit is imperfect and the solution sparse, so no null direction is
    feasible and the minimiser is unique after all. On the Proposition 99 placebo
    family all 38 solves are unique, at 37 donors against 19 pre-periods, and the
    shape test admits none of them.

    Parameters
    ----------
    B : np.ndarray, shape (m, J)
        The design the weights were solved against.
    A : np.ndarray, shape (m,)
        The target.
    w : np.ndarray, shape (J,)
        A minimiser, from any exact solver.
    tol : float
        Relative tolerance for the weakly-active set and the rank test.

    Returns
    -------
    bool
        ``True`` only when the minimiser is provably the only one. Use it that
        way round: a ``False`` costs a second solve, a wrong ``True`` would
        change an answer.
    """
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if B.ndim != 2 or w.shape[0] != B.shape[1] or A.shape[0] != B.shape[0]:
        raise ValueError(
            f"shapes do not line up: B {B.shape}, A {A.shape}, w {w.shape}.")
    if B.shape[1] == 1:
        return True
    g = B.T @ (B @ w - A)
    nu = float(g @ w)
    scale = 1.0 + float(np.max(np.abs(g)))
    # Weakly active: carrying weight, or able to enter without raising the loss.
    active = (w > tol * 1e-2) | (np.abs(g - nu) <= tol * scale)
    n_active = int(active.sum())
    if n_active == 0:  # pragma: no cover - a simplex point always has support
        return True
    M = np.vstack([B[:, active], np.ones((1, n_active))])
    return bool(np.linalg.matrix_rank(M, tol=tol * 1e-2) == n_active)


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
        #
        # One at a time, which is what makes the iteration count scale with the
        # size of the optimal support: on the Basque placebo panels it runs from
        # 9 iterations where two donors carry the fit to 33 where six do. Adding
        # the top-m violators together to shorten that was measured and does not
        # work -- Wolfe's guarantee that an added vertex takes positive weight
        # covers one addition, and with several the blocked step drops the ones
        # that did not, so the corral oscillates. Every Basque panel then ran to
        # the iteration cap and finished 1-2 percent above the optimum.
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


def _reproduces_the_loop(B: np.ndarray, A: np.ndarray, w: np.ndarray) -> bool:
    """Whether a batched answer may stand in for the one-at-a-time solve.

    Both halves are needed and neither implies the other. ``w`` has to solve the
    program as posed on the design, which the Gram form can fail to do when
    ``cond(B)^2`` exceeds float64; and the program has to have one solution, or
    two exact solvers land in different places on a face.
    """
    return (simplex_point_is_optimal(B, A, w)
            and simplex_optimum_is_unique(B, A, w))


def solve_simplex_loo_exact(
    M: np.ndarray,
    *,
    n_targets: Optional[int] = None,
    fallback: Optional[Callable] = None,
    return_info: bool = False,
):
    """Leave-one-out simplex least squares over the columns of ``M``, exactly.

    For each ``j`` solves ``min ||M[:, k != j] w - M[:, j]||^2`` on the simplex.
    This is the shape of an in-space placebo test, where every donor is cast as
    treated in turn against the others.

    The family is carried by one Gram. Deleting a column deletes a row and a
    column of ``M' M``, and each target is itself a column of ``M``, so the cross
    term is a column of that same Gram: with ``Mg = M' M``, the ``j``-th
    problem's Gram is ``Mg[a,b] - Mg[a,j] - Mg[j,b] + Mg[j,j]`` over the
    remaining indices. No product with the data occurs per problem, and the
    batched active set then certifies the whole family together.

    Every member is then certified against its own design: the returned point
    has to satisfy the KKT conditions there (:func:`simplex_point_is_optimal`,
    which the Gram form can fail when ``cond(M)^2`` exceeds float64) and its
    optimum has to be a point and not a face
    (:func:`simplex_optimum_is_unique`, since on a face two exact solvers land
    in different places). Members failing either are re-solved with the
    single-problem active set of :mod:`~mlsynth.utils.bilevel.active_set`. So
    the result is not merely optimal but identical to solving the family one at
    a time, which matters here: the placebo p-value is a rank statistic over
    these fits, and the library's published ranks came from that solver.

    Parameters
    ----------
    M : np.ndarray, shape (m, J)
        Columns to fit against one another; ``J >= 2``.
    n_targets : int, optional
        How many leading columns are cast as treated. Columns beyond it stay in
        every donor pool without being a target themselves. Defaults to ``J``.
    fallback : callable, optional
        ``(B, A) -> w``, the one-at-a-time solver this batch is standing in for,
        used on members that fail certification. Defaults to
        :func:`~mlsynth.utils.bilevel.active_set.solve_simplex_qp`. Pass the
        caller's own: VanillaSC's engine reaches that solver through a wrapper
        that escalates to CVXPY when it reports failure on itself, and the two
        can disagree exactly where the certification fails.
    return_info : bool
        If ``True`` also return ``{"n_problems", "n_fallback"}``.

    Returns
    -------
    np.ndarray, shape (n_targets, J)
        Row ``j`` holds the weights fitting ``M[:, j]``, with ``W[j, j]`` zero.
    """
    if fallback is None:
        from .active_set import solve_simplex_qp as fallback

    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D (m, J); got shape {M.shape}.")
    J = M.shape[1]
    if J < 2:
        raise ValueError(
            f"M has {J} column(s); leaving one out needs at least two, since the "
            f"remaining columns are what the left-out one is fitted against.")
    n = J if n_targets is None else int(n_targets)
    if not 1 <= n <= J:
        raise ValueError(f"n_targets must be between 1 and {J}; got {n}.")

    keep = ~np.eye(J, dtype=bool)[:n]                      # (n, J)
    idx = np.argsort(~keep, axis=1, kind="stable")[:, :J - 1]   # donors per target
    Mg = M.T @ M
    # G_j from the Gram alone: the deleted column enters only through its own
    # row, column and diagonal entry.
    G = (Mg[idx[:, :, None], idx[:, None, :]]
         - Mg[idx, np.arange(n)[:, None]][:, :, None]
         - Mg[np.arange(n)[:, None], idx][:, None, :]
         + Mg[np.arange(n), np.arange(n)][:, None, None])
    Wc = solve_simplex_minnorm_batch(0.5 * (G + np.swapaxes(G, 1, 2)))

    W = np.zeros((n, J))
    np.put_along_axis(W, idx, Wc, axis=1)
    n_fallback = 0
    for j in range(n):
        cols = idx[j]
        if not _reproduces_the_loop(M[:, cols], M[:, j], W[j, cols]):
            W[j, cols] = np.asarray(fallback(M[:, cols], M[:, j]),
                                    dtype=float).ravel()
            n_fallback += 1
    if return_info:
        return W, {"n_problems": n, "n_fallback": n_fallback}
    return W


def solve_simplex_shared_design(
    A: np.ndarray,
    B: np.ndarray,
    *,
    fallback: Optional[Callable] = None,
    return_info: bool = False,
):
    """Simplex least squares for many targets against one shared design.

    Solves ``min ||A w - b_j||^2`` on the simplex for every column ``b_j`` of
    ``B``. This is the shape a cohort of treated units poses when they are
    fitted against a common donor pool.

    One design means the family costs one Gram plus one cross product. With
    ``c_j = A' b_j`` and ``s_j = b_j' b_j``, the ``j``-th problem's Gram is

        ``G_j = A'A - c_j 1' - 1 c_j' + s_j 1 1'``,

    so ``A'A`` and ``A'B`` formed once carry the whole set, and the batched
    active set then runs the cohort in lockstep.

    Every member is then certified against the design: the returned point has to
    satisfy the KKT conditions there (:func:`simplex_point_is_optimal`) and its
    optimum has to be a point and not a face (:func:`simplex_optimum_is_unique`).
    Members failing either are re-solved one at a time. The weights are reported
    per treated unit and the counterfactuals are built from them, so the batch
    has to reproduce the one-at-a-time solve and not merely tie with it. The
    optimality test is what covariate matching needs: predictors in different
    units push ``cond(A)`` to 1e7 and the Gram past what float64 keeps, and
    there most of the group falls back and the batch buys nothing. On the
    outcome-only design it is 89 members with 6 falling back, at 43ms against
    the loop's 396ms.

    Parameters
    ----------
    A : np.ndarray, shape (m, J)
        Design shared by every target.
    B : np.ndarray, shape (m, k) or (m,)
        Targets, one per column. A 1-D array is one target.
    fallback : callable, optional
        ``(B, A) -> w``, the one-at-a-time solver this batch is standing in for,
        used on members that fail certification. Defaults to
        :func:`~mlsynth.utils.bilevel.active_set.solve_simplex_qp`, which is
        what STACKEDSC calls.
    return_info : bool
        If ``True`` also return ``{"n_problems", "n_fallback"}``.

    Returns
    -------
    np.ndarray, shape (k, J)
        Row ``j`` holds the weights fitting ``B[:, j]``.
    """
    if fallback is None:
        from .active_set import solve_simplex_qp as fallback

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D (m, J); got shape {A.shape}.")
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D (m, k) or 1-D (m,); got shape {B.shape}.")
    if B.shape[0] != A.shape[0]:
        raise ValueError(
            f"B has {B.shape[0]} rows but A has {A.shape[0]}; the targets and "
            f"the design must be measured over the same rows.")

    k = B.shape[1]
    AtA = A.T @ A
    C = (A.T @ B).T                                        # (k, J), row j is c_j
    s = np.einsum("ij,ij->j", B, B)                        # (k,)
    G = AtA[None] - C[:, :, None] - C[:, None, :] + s[:, None, None]
    W = solve_simplex_minnorm_batch(0.5 * (G + np.swapaxes(G, 1, 2)))

    n_fallback = 0
    for j in range(k):
        if not _reproduces_the_loop(A, B[:, j], W[j]):
            W[j] = np.asarray(fallback(A, B[:, j]), dtype=float).ravel()
            n_fallback += 1
    if return_info:
        return W, {"n_problems": int(B.shape[1]), "n_fallback": n_fallback}
    return W
