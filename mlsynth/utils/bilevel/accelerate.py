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

The support arrives before the weights do, and the support is all the active set
is being told, so the loop stops on the support and not on the iterate: see
``SUPPORT_PATIENCE``. :func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`
calls this itself for a wide enough pool, so the seed is a property of the solver
and reaches every caller.

The accelerator is *speed only*. It supplies a feasible warm start; the exact
active-set (with the cvxpy fallback) still determines the returned weights, so
the answer is unchanged for any warm start. Correctness therefore never depends
on FISTA converging, the Lipschitz estimate being tight, or the problem being
well-conditioned -- only the runtime does.
"""
from __future__ import annotations

import numbers
from typing import Optional

import numpy as np

# Engage the accelerator only for donor pools at least this wide. Below it the
# cold active-set already solves in well under a pivot's worth of FISTA overhead
# (measured crossover ~ J = 60-80; typical SC panels -- Prop 99 J = 38, Basque
# J = 16 -- stay comfortably on the untouched fast path). Also skipped whenever
# the caller already supplies a warm start (e.g. the conformal / placebo loops).
ACCEL_MIN_DONORS = 80

# A coordinate counts as in the support above this. It is the tolerance
# ``solve_simplex_qp`` itself reads when it turns a warm start into an active
# set (``active = w <= tol``, ``tol = 1e-9``), so the two agree on what the seed
# is saying.
SUPPORT_TOL = 1e-9
# Consecutive iterations the support must hold before the warm start returns --
# a quarter of the default budget. The seed's job is to name the support, so
# once it stops moving the remaining iterations refine weights the exact solve
# is about to recompute: on the SDID programs of a 101x120 panel the support is
# final by iteration ~150 of 400.
#
# The margin is what the number is for, and it is asymmetric. One saved
# iteration is worth ~44 us; one extra pivot the seed fails to save costs
# ~1.2 ms on a 96x99 free set, or about 27 iterations. A stop that fires while
# the support is still moving therefore loses even when it saves most of the
# loop, and a design whose support never settles -- the outcome-only SC program
# of that same panel -- must simply run to ``max_iter``. At a patience of 25
# that program stopped at iteration 376 with a seed three donors worse and paid
# 12 percent for it; at 100 the stop does not fire there at all, while SDID's
# two programs still stop at 261 and 231.
SUPPORT_PATIENCE = 100
# Iterations between support samples. The test is three numpy calls on a
# J-vector, which is ~11 percent of an iteration at these sizes -- enough that
# checking every iteration cost more than the stop saved on any design where it
# did not fire. Sampling makes the bookkeeping free and costs at most one
# sampling interval of lateness.
SUPPORT_CHECK_EVERY = 10

# Iterations of the batched seed. It has one job -- name the support -- and it
# is paced by the hardest row in the batch, so a budget sized for the iterate
# to converge is spent refining weights the exact solve recomputes. Measured on
# a 500-problem, 99-donor ridged family: 50 iterations leave the exact solver 11
# pivots and the whole solve is 7.3x the cold one; 100 leave it 4 and the solve
# is 6.8x, the seed having cost more than the pivots it saved.
SEED_MAX_ITER = 50
# Power iterations for the seed's Lipschitz estimate. See ``_spectral_bound_batch``.
SEED_SPECTRAL_ITERS = 3
# Donor pools below this are not seeded. The batch amortises one seed over every
# problem in the stack, so the crossover sits well below the single-problem
# solver's ``ACCEL_MIN_DONORS``; measured per shape in the table in #461.
BATCH_ACCEL_MIN_DONORS = 30


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
                     max_iter: int = 400, tol: float = 1e-7,
                     support_patience: Optional[int] = SUPPORT_PATIENCE
                     ) -> np.ndarray:
    """A feasible, near-optimal warm start for ``min ||A - B w||^2`` on the
    simplex, via Gram-collapsed FISTA.

    Deterministic. Returns a point on the simplex; it is intended to seed the
    exact active-set solver, not to be used as the final solution.

    Two rules stop the loop. ``tol`` is the usual one, on how far the iterate
    moved. ``support_patience`` is the one that matters for the job: once the
    support has been unchanged for that many consecutive iterations, the active
    set already has everything this function can tell it, and the remaining
    iterations only refine weights the exact solve is about to recompute. On the
    two SDID programs of a 101x120 panel the support is final by iteration ~150
    while the iterate is still moving at 400; the stop fires at 261 and 231 and
    takes those two programs from 50.0 ms to 36.7 ms end to end. Pass ``None`` to
    stop on ``tol`` and ``max_iter`` alone.

    The support is sampled every ``SUPPORT_CHECK_EVERY`` iterations, and the
    counter arms only after the first coordinate is pinned. Both exist because a
    seed that is *nearly* right is not most of the benefit: an extra pivot costs
    about 27 iterations, so a stop that fires while the support is still moving
    is a loss even when it saves most of the loop. On the outcome-only SC program
    of that same panel the support never settles and the loop correctly runs to
    ``max_iter``.

    Stopping earlier gives a coarser seed, never a wrong answer: the active set
    certifies KKT optimality on the design, whatever point it starts from.

    Parameters
    ----------
    B : numpy.ndarray, shape (m, J)
        Donor matching matrix.
    A : numpy.ndarray, shape (m,)
        Treated unit's matching vector.
    max_iter : int
        Hard cap on iterations.
    tol : float
        Stop when the iterate moves less than this in the sup norm.
    support_patience : int or None
        Stop when the support (``w > SUPPORT_TOL``) has held for this many
        consecutive iterations, sampled every ``SUPPORT_CHECK_EVERY`` of them.
        ``None`` disables the rule.

    Returns
    -------
    numpy.ndarray, shape (J,)
    """
    if support_patience is not None and (
            not isinstance(support_patience, numbers.Integral)
            or support_patience < 1):
        raise ValueError(
            "support_patience must be a positive integer or None, got "
            f"{support_patience!r}."
        )
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
    support = None
    held = 0
    holds_needed = (None if support_patience is None
                    else max(1, -(-support_patience // SUPPORT_CHECK_EVERY)))
    for i in range(max_iter):
        w_new = simplex_project(y - (2.0 * (G @ y - c)) / L)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        if holds_needed is not None and i % SUPPORT_CHECK_EVERY == 0:
            current = w_new > SUPPORT_TOL
            # The counter arms only once FISTA has begun pinning coordinates.
            # It starts at the uniform point, where every coordinate is
            # positive, so an unarmed counter reads that plateau as a support
            # that has settled and returns the whole pool as the seed -- which
            # is the uniform start under another name, and leaves the active set
            # the full cold pivot count. Measured on a 96x100 SC design whose
            # optimum has 16 donors: unarmed, the seed named all 100 and the
            # solve took the cold 84 pivots; armed, it names 49 and takes 33.
            armed = int(current.sum()) < J
            held = held + 1 if (armed and support is not None
                                and np.array_equal(current, support)) else 0
            support = current
            if held >= holds_needed:
                w = w_new
                break
        w = w_new
        t = t_new
    return w


def simplex_project_batch(V: np.ndarray) -> np.ndarray:
    """Row-wise :func:`simplex_project`, vectorised over the whole stack.

    Same construction, same result -- the threshold ``theta`` that makes
    ``(v - theta)_+`` sum to one -- found for every row at once, since the batch
    solver's seed projects a ``(S, J)`` block on every iteration and a Python
    loop over ``S`` would cost more than the gradient step it follows.
    """
    V = np.atleast_2d(np.asarray(V, dtype=float))
    S, J = V.shape
    if J == 1:
        return np.ones((S, 1))
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, J + 1)
    cond = U - css / ind > 0                  # cond[:, 0] is always True
    # The largest index where the condition holds, per row. Reversing turns
    # "last True" into "first True", which argmax answers in one pass.
    rho = J - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(S), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def _spectral_bound_batch(G: np.ndarray, iters: int = SEED_SPECTRAL_ITERS) -> np.ndarray:
    """Per-problem estimate of ``lambda_max``, inflated 10 percent, batched.

    The scalar routine's contract carries over, and it is what lets this run on
    a handful of iterations where that one runs fifty: a loose or invalid bound
    only slows the seed, since the projection keeps every iterate on the
    simplex, and it can never make the certified answer wrong. Each iteration
    here costs a batched ``(S, J, J)`` matvec -- the same price as an iteration
    of the seed it is sizing -- so fifty of them cost more than the seed does.
    Measured on a 500-problem, 99-donor ridged family: at fifty the bound is 73
    percent of the seed's total cost and the whole solve is 3.4x the cold one;
    at three it is 7.3x, and the exact solver certifies in 11 iterations instead
    of 12.
    """
    S, J = G.shape[0], G.shape[-1]
    v = np.random.default_rng(0).standard_normal((S, J))    # fixed seed
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)
    lam = np.zeros(S)
    for _ in range(iters):
        Gv = np.einsum("sjk,sk->sj", G, v)
        nrm = np.linalg.norm(Gv, axis=1, keepdims=True)
        alive = (nrm[:, 0] > 0.0)
        v = np.where(alive[:, None], Gv / np.maximum(nrm, 1e-300), v)
        lam = np.einsum("sj,sjk,sk->s", v, G, v)
    return np.where(lam > 0.0, lam * 1.1, 0.0)


def fista_warm_start_batch(
    G: np.ndarray, *, max_iter: int = SEED_MAX_ITER, tol: float = 1e-7,
    support_patience: Optional[int] = SUPPORT_PATIENCE,
) -> np.ndarray:
    """Feasible near-optimal seeds for ``min w' G_s w`` on the simplex, one row
    per problem in the stack.

    The batched counterpart of :func:`fista_warm_start`, and the same job: name
    the support so the exact solver does not have to discover it. The two differ
    only in what they are handed. The single-problem form takes a design and a
    target and collapses them into a Gram; a batch solver already holds its
    Grams, and its objective is the pure quadratic ``w' G w``, so the gradient
    is ``2 G w`` with no cross term.

    Naming the support is worth more here than it is there. Wolfe's method adds
    one vertex per iteration and solves a corral system that grows with each
    addition, so a ``k``-donor optimum costs at least ``k`` iterations of a
    ``(k, k)`` solve. The ridged programs SDID builds have optima on a dense
    face -- around 70 percent of the pool -- which is the worst case for that
    rule and the best case for being told the answer: on a 500-problem placebo
    family with 99 donors the seed takes 87 iterations to 1.

    Every problem runs the same number of iterations. The stopping rules are
    therefore read across the whole batch -- the loop ends when *every* row has
    settled -- so a batch is paced by its hardest member, as the solve it seeds
    already is.

    The seed is speed only. The active set certifies optimality whatever point
    it starts from, so a coarse, late, or wrong seed costs iterations and cannot
    move the answer.

    Parameters
    ----------
    G : numpy.ndarray, shape (S, J, J)
        Symmetric positive semi-definite Gram matrices, one per problem.
    max_iter : int
        Hard cap on iterations.
    tol : float
        Stop once no row moves more than this in the sup norm.
    support_patience : int or None
        Stop once no row's support has changed for this many consecutive
        iterations, sampled every ``SUPPORT_CHECK_EVERY`` of them. ``None``
        disables the rule.

    Returns
    -------
    numpy.ndarray, shape (S, J)
    """
    if support_patience is not None and (
            not isinstance(support_patience, numbers.Integral)
            or support_patience < 1):
        raise ValueError(
            "support_patience must be a positive integer or None, got "
            f"{support_patience!r}."
        )
    G = np.asarray(G, dtype=float)
    S, J = G.shape[0], G.shape[-1]
    if J == 1:
        return np.ones((S, 1))
    L = 2.0 * _spectral_bound_batch(G)
    # A degenerate problem (a zero Gram, whose objective is flat, or a
    # non-finite one) has no usable step; it keeps the uniform point, which is
    # feasible and is as good as any other for a flat objective.
    usable = np.isfinite(L) & (L > 0.0)
    L = np.where(usable, L, 1.0)[:, None]

    w = np.full((S, J), 1.0 / J)
    y = w.copy()
    t = 1.0
    support = None
    held = 0
    holds_needed = (None if support_patience is None
                    else max(1, -(-support_patience // SUPPORT_CHECK_EVERY)))
    for i in range(max_iter):
        step = np.where(usable[:, None], 2.0 * np.einsum("sjk,sk->sj", G, y) / L, 0.0)
        w_new = simplex_project_batch(y - step)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        if holds_needed is not None and i % SUPPORT_CHECK_EVERY == 0:
            current = w_new > SUPPORT_TOL
            # Armed on the same reasoning as the scalar loop: every row starts
            # at the uniform point where all J coordinates are positive, so an
            # unarmed counter would read that plateau as a settled support and
            # hand back the whole pool -- the uniform start renamed, which
            # leaves the solver its full cold iteration count. Read across the
            # batch, so one row still moving keeps the loop running.
            armed = bool((current.sum(axis=1) < J).all())
            held = held + 1 if (armed and support is not None
                                and np.array_equal(current, support)) else 0
            support = current
            if held >= holds_needed:
                w = w_new
                break
        w = w_new
        t = t_new
    return w
