"""A bounded dual simplex for linear programs whose every variable is boxed.

The programs are

.. math::

    \\min_x\\; c^\\top x \\quad \\text{s.t.}\\quad r_l \\le A x \\le r_u, \\qquad l \\le x \\le u,

with every bound finite. The method is the revised dual simplex in the form of Koberstein ("The Dual Simplex Method,
Techniques for a Fast and Stable Implementation", PhD thesis, Universitaet Paderborn, 2005), Algorithm 7: dual
steepest-edge pricing (section 3.3, rule 3.53, after Forrest and Goldfarb), the bound flipping ratio test
(section 3.2, Algorithms 4 to 6) with Harris' tolerance in the choice of the entering variable (section 6.2.2.4,
Algorithm 18), run on the computational form ``A x - s = 0`` with the row activities ``s`` as logical variables
boxed by ``[r_l, r_u]``.

Two consequences of every variable being boxed shape the module, and they are why it exists beside HiGHS.

* No dual phase I. A basis is dual feasible once each nonbasic variable sits at the bound its reduced cost calls
  for: lower when ``d_j > 0``, upper when ``d_j < 0`` (Koberstein, Table 2.2). Any basis can be made so by flipping
  bounds, so the slack basis, a parent problem's basis restricted to a child, and a basis extended by new rows all
  start the dual simplex directly (Koberstein, Chapter 4 covers the general case, which never arises here).
* A lower bound at every iterate. With multipliers ``y`` on ``A x - s = 0``, weak duality over the boxes gives
  ``L(y) = sum_j min(l_j d_j, u_j d_j) + sum_i min(r_l,i y_i, r_u,i y_i)`` with ``d = c - A'y``, a lower bound on
  the optimum for any ``y`` (Koberstein, section 2.3, the dual (2.11) with every bound finite). The solve can stop
  at an iteration limit and its multipliers still bound the optimum; scipy's HiGHS interface returns no multipliers
  when its iteration limit stops it. A caller whose decision rests on the bound evaluates ``L(y)`` rigorously
  itself; this module computes it in floating point.

The basis inverse is held dense, updated by the product-form rule (Koberstein, equations 3.27-3.28) and refactored
every ``REFACTOR_EVERY`` iterations (the constant frequency phi = 100 of section 6.2.3.1) and whenever the pivot
element computed from the row and from the column disagree (the pivot element test, equations 6.27-6.29). The
bases met in PANGEO's cover LPs have a few hundred rows at most and 0/1 entries, the numerically easy case of
section 6.2.3.2. The multipliers, reduced costs and basic values are recomputed from the inverse at every
iteration (the variant of Koberstein's Algorithm 3), which costs one product with ``A`` and keeps them consistent
with the basis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

try:  # numba is the optional ``fast`` extra; without it the NumPy engine runs
    from numba import njit
    _NUMBA = True
except ImportError:  # pragma: no cover - the suite runs with numba installed
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name.
CANONICAL_NAME = "mlsynth.utils.solvers.dual_simplex"
_CACHE = __name__ == CANONICAL_NAME

#: Primal feasibility tolerance, absolute and relative to the bound's size (Koberstein, Table 6.1 sets 1e-7 and
#: 1e-9 for general LPs; the cover LPs here have 0/1 rows, so the absolute tolerance is tightened to 1e-9).
PRIMAL_TOL = 1e-9
PRIMAL_REL_TOL = 1e-9
#: Dual feasibility tolerance (Koberstein, Table 6.1): Harris' ratio test admits reduced costs this far past zero.
DUAL_TOL = 1e-7
#: The smallest pivot-row entry a ratio test accepts (Koberstein, Table 6.1, within [1e-7, 1e-5]).
PIVOT_TOL = 1e-7
#: Reduced costs this close to zero are zero when the bound statuses are set (Koberstein, Table 6.1).
ZERO_TOL = 1e-12
#: Iterations between refactorizations of the basis (Koberstein, section 6.2.3.1).
REFACTOR_EVERY = 100
#: The pivot element test: the entry of the pivot row and of the pivot column must agree to this, relative to
#: 1 + |entry|, or the basis is refactored before the change (Koberstein, equations 6.27-6.29).
PIVOT_CHECK_TOL = 1e-9
#: Iterations without a rise in the dual objective before the costs are perturbed, and again (after they are
#: restored) before Bland's rule takes over: Koberstein's maxcycle = 3 * phi (section 6.3.1).
STALL_AFTER = 3 * REFACTOR_EVERY
#: The engine when ``solve_boxed_lp`` is not told one: ``"auto"`` is the compiled kernel when numba is installed and
#: the NumPy reference when it is not.
DEFAULT_ENGINE = "auto"
#: A ray proves infeasibility only when its Lagrangian under zero costs exceeds this share of the sum of its terms'
#: magnitudes, a margin for the rounding of that sum; a ray that falls short is not reported as infeasibility.
INFEASIBILITY_MARGIN = 1e-9
#: The cost perturbation's size: a fixed part of 100 times the dual tolerance and a part psi = 1e-5 relative to the
#: cost, randomised by a factor in [0.5, 1) (Koberstein, section 6.3.1, steps 1 and 2).
PERTURB_ABS = 100.0 * DUAL_TOL
PERTURB_REL = 1e-5


@dataclass(frozen=True)
class BoxedBasis:
    """A basis of the computational form: ``basic`` lists the ``m`` basic variables, structural ``j < n`` and logical
    (row ``i``) as ``n + i``; ``at_upper`` marks, for all ``n + m`` variables, those nonbasic at their upper bound
    (entries of basic variables are ignored). ``inverse`` is the basis matrix's inverse when the basis came from a
    solve: an LP that changes only bounds or costs keeps the same basis matrix, so a solve started from it skips the
    factorization once a residual check confirms it belongs to the new LP's matrix. After a change of bounds alone the
    basis also stays dual feasible, which is why branch and bound re-solves its subproblems by the dual simplex
    (Koberstein, "The Dual Simplex Method, Techniques for a fast and stable implementation", PhD thesis, Paderborn,
    2005, chapter 1, p. 2)."""

    basic: np.ndarray
    at_upper: np.ndarray
    inverse: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    def restrict(self, cols: Sequence[int], rows: Sequence[int]) -> "BoxedBasis":
        """The basis carried to the LP over the structural columns ``cols`` and rows ``rows`` of this one (both in
        this LP's numbering, in the child's order). Surviving basic variables keep their place; rows left without one
        get their logical; the result has one basic variable per kept row. A singular result is repaired by the
        solver."""
        m = int(self.basic.size)
        n = int(self.at_upper.size) - m
        cols = np.asarray(cols, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        new_col = -np.ones(n, dtype=np.int64)
        new_col[cols] = np.arange(cols.size)
        new_row = -np.ones(m, dtype=np.int64)
        new_row[rows] = np.arange(rows.size)
        n2, m2 = int(cols.size), int(rows.size)

        def mapped(j: int) -> int:
            return int(new_col[j]) if j < n else (n2 + int(new_row[j - n]) if new_row[j - n] >= 0 else -1)

        basic = [b for b in (mapped(int(j)) for j in self.basic) if b >= 0]
        have = set(basic)
        basic += [n2 + i for i in range(m2) if n2 + i not in have]
        at_upper = np.zeros(n2 + m2, dtype=bool)
        at_upper[:n2] = self.at_upper[cols]
        at_upper[n2:] = self.at_upper[n + rows]
        return BoxedBasis(np.asarray(basic[:m2], dtype=np.int64), at_upper)

    def with_rows(self, k: int) -> "BoxedBasis":
        """The basis of the LP with ``k`` rows appended: their logicals join the basis, which keeps it dual
        feasible (a basic logical has reduced cost 0 and moves no other)."""
        m = int(self.basic.size)
        n = int(self.at_upper.size) - m
        basic = np.concatenate([self.basic, n + m + np.arange(k)]).astype(np.int64)
        return BoxedBasis(basic, np.concatenate([self.at_upper, np.zeros(k, dtype=bool)]))


@dataclass(frozen=True)
class BoxedLPResult:
    """A solve's outcome. ``status`` is ``"optimal"``, ``"iteration_limit"`` (stopped with a dual feasible basis;
    ``bound`` still bounds the optimum) or ``"infeasible"`` (the dual is unbounded, so no point satisfies the rows and
    boxes, and ``bound`` is ``+inf``). ``x`` is the primal basic solution (feasible only when optimal), ``row_activity``
    is ``A x``, ``y`` the row multipliers, ``d = c - A'y`` the structural reduced costs, ``objective = c'x``,
    ``bound`` the Lagrangian value of ``y`` (:func:`lagrangian_bound`), ``iterations`` the basis changes made,
    ``basis`` the final basis (a warm start for a related LP), ``basis_repaired`` whether a singular warm-start
    basis had dependent columns replaced by logicals, and ``degeneracy_stage`` the furthest of the degeneracy
    measures of :func:`solve_boxed_lp` the solve used: ``"none"``, ``"perturbed"`` (perturbed costs, not yet
    restored), ``"restored"`` (true costs back after a perturbed optimum) or ``"bland"`` (Bland's rule). With ``status == "infeasible"``, ``ray`` holds row multipliers whose
    Lagrangian under zero costs is positive, a Farkas certificate for the boxed LP (Bazaraa, Jarvis and Sherali,
    Linear Programming and Network Flows, 4th ed., Lemma 5.1, p. 234): a caller that prunes on it can re-evaluate it
    rigorously. ``"cutoff"`` means the Lagrangian bound under the true costs reached the ``cutoff`` given, and the
    solve stopped there. ``engine`` names the engine that ran, and ``warm_inverse_used`` whether the solve started
    from the warm-start basis's own inverse."""

    status: str
    x: np.ndarray
    row_activity: np.ndarray
    y: np.ndarray
    d: np.ndarray
    objective: float
    bound: float
    iterations: int
    basis: BoxedBasis
    basis_repaired: bool = False
    degeneracy_stage: str = "none"
    ray: Optional[np.ndarray] = None
    engine: str = "numpy"
    warm_inverse_used: bool = False


def _lagrangian_terms(c, A, rl, ru, l, u, y) -> np.ndarray:
    A = sp.csc_matrix(A) if not sp.issparse(A) else A.tocsc()
    y = np.asarray(y, dtype=float)
    d = np.asarray(c, dtype=float) - A.T @ y
    return np.concatenate([np.minimum(np.asarray(l, float) * d, np.asarray(u, float) * d),
                           np.minimum(np.asarray(rl, float) * y, np.asarray(ru, float) * y)])


def lagrangian_bound(c, A, rl, ru, l, u, y) -> float:
    """The Lagrangian lower bound of multipliers ``y`` on the boxed LP: ``sum_j min(l_j d_j, u_j d_j) + sum_i
    min(rl_i y_i, ru_i y_i)`` with ``d = c - A'y``. It bounds the optimum from below for every ``y`` (weak duality with
    every variable boxed; Koberstein 2005, section 2.3), in floating point; it is summed with ``math.fsum``."""
    return math.fsum(_lagrangian_terms(c, A, rl, ru, l, u, y).tolist())


def _bland_entering(cand: np.ndarray, ratio: np.ndarray) -> int:
    """Bland's entering choice: among the candidates whose ratio ties the smallest (within ``ZERO_TOL``), the one of
    smallest variable index. Returns its position in ``cand``."""
    tie = np.flatnonzero(ratio <= np.min(ratio) + ZERO_TOL)
    return int(tie[np.argmin(cand[tie])])


def _vector(v, name: str) -> np.ndarray:
    return np.atleast_1d(np.asarray(v, dtype=float))


def _validate(c, A, rl, ru, l, u, basis, max_iterations):
    c, rl, ru, l, u = (_vector(v, k) for v, k in ((c, "c"), (rl, "rl"), (ru, "ru"), (l, "l"), (u, "u")))
    A = sp.csc_matrix(A, dtype=float) if not sp.issparse(A) else A.tocsc().astype(float)
    if c.ndim != 1 or A.ndim != 2:
        raise MlsynthConfigError(f"c must be a vector and A a matrix; got shapes {c.shape} and {A.shape}.")
    m, n = A.shape
    for v, k, size in ((c, "c", n), (l, "l", n), (u, "u", n), (rl, "rl", m), (ru, "ru", m)):
        if v.shape != (size,):
            raise MlsynthConfigError(f"{k} has shape {v.shape}; A has shape {A.shape}, so {k} needs shape ({size},).")
    for v, k in ((l, "l"), (u, "u"), (rl, "rl"), (ru, "ru")):
        if not np.all(np.isfinite(v)):
            raise MlsynthConfigError(f"every bound must be finite for the boxed dual simplex; {k} has a non-finite "
                                     "entry.")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(A.data)):
        raise MlsynthDataError("the costs c and the matrix A must be finite.")
    if np.any(l > u):
        raise MlsynthDataError(f"a lower bound lies above its upper bound (variable {int(np.argmax(l > u))}).")
    if np.any(rl > ru):
        raise MlsynthDataError(f"a row's lower bound lies above its upper bound (row {int(np.argmax(rl > ru))}).")
    if max_iterations is not None and (isinstance(max_iterations, bool)
                                       or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0):
        raise MlsynthConfigError(f"max_iterations must be a non-negative integer or None; got {max_iterations!r}.")
    if basis is not None:
        b = np.asarray(basis.basic)
        if b.shape != (m,) or np.asarray(basis.at_upper).shape != (n + m,) or b.size and (
                b.min() < 0 or b.max() >= n + m) or np.unique(b).size != b.size:
            raise MlsynthConfigError(f"the warm-start basis must list {m} distinct variables of {n + m} and mark "
                                     f"bounds for all {n + m}; got {b.size} basic and {np.asarray(basis.at_upper).size}.")
    return c, A, rl, ru, l, u


def _repair(F: sp.csc_matrix, basic: np.ndarray, n: int, m: int) -> tuple:
    """A nonsingular basis from ``basic``: its independent columns (QR with column pivoting), completed by logicals
    that raise the rank. Returns the basis and whether anything was replaced."""
    B = F[:, basic].toarray()
    if m == 0:
        return basic, False
    import scipy.linalg as sla

    _, R, piv = sla.qr(B, pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > 1e-9 * max(diag.max(), 1.0)))
    if rank == m:
        return basic, False
    kept = list(basic[piv[:rank]])
    for i in range(m):
        if len(kept) == m:
            break
        if n + i in kept:
            continue
        trial = F[:, kept + [n + i]].toarray()
        if np.linalg.matrix_rank(trial) == len(kept) + 1:
            kept.append(n + i)
    return np.asarray(kept, dtype=np.int64), True


# --- the compiled engine --------------------------------------------------------------------------------------------
# The same algorithm as the NumPy loop of solve_boxed_lp, with Koberstein's incremental updates (Algorithm 2) in place
# of the reference's recomputation each iteration: d <- d - t alpha_r with t = d_q / alpha_q, Algorithm 5's flip pass
# (only reduced costs past the dual tolerance flip), x_B <- x_B - B^-1 a~ for the flips, then the primal step along
# alpha_q. The pivot element test runs before any update, and every refactorization recomputes everything from the
# basis. Logical i's column is -e_i, so its reduced cost is y_i: the multipliers are the logical part of d.


@njit(cache=_CACHE)
def _col_into(Ap, Ai, Ax, n, j, out):  # pragma: no cover - compiled; checked through the engine tests
    out[:] = 0.0
    if j < n:
        for t in range(Ap[j], Ap[j + 1]):
            out[Ai[t]] = Ax[t]
    else:
        out[j - n] = -1.0


@njit(cache=_CACHE)
def _dot_col(Ap, Ai, Ax, n, j, v):  # pragma: no cover - compiled; checked through the engine tests
    if j < n:
        s = 0.0
        for t in range(Ap[j], Ap[j + 1]):
            s += Ax[t] * v[Ai[t]]
        return s
    return -v[j - n]


@njit(cache=_CACHE)
def _factor_nb(Ap, Ai, Ax, n, m, basic):  # pragma: no cover - compiled; checked through the engine tests
    B = np.zeros((m, m))
    col = np.zeros(m)
    for r in range(m):
        _col_into(Ap, Ai, Ax, n, basic[r], col)
        B[:, r] = col
    Binv = np.linalg.inv(B)
    beta = np.empty(m)
    for i in range(m):
        beta[i] = Binv[i] @ Binv[i]
    return Binv, beta


@njit(cache=_CACHE)
def _recompute_nb(Ap, Ai, Ax, n, m, work, lo, up, width, basic, is_basic, at_upper, Binv, d, xB,
                  dual_tol):  # pragma: no cover - compiled; checked through the engine tests
    cB = np.empty(m)
    for r in range(m):
        cB[r] = work[basic[r]]
    y = Binv.T @ cB
    N = n + m
    for j in range(N):
        d[j] = 0.0 if is_basic[j] else work[j] - _dot_col(Ap, Ai, Ax, n, j, y)
    for j in range(N):
        if not is_basic[j] and width[j] > 0:
            if d[j] > dual_tol:
                at_upper[j] = False
            elif d[j] < -dual_tol:
                at_upper[j] = True
    rhs = np.zeros(m)
    for j in range(N):
        if not is_basic[j]:
            v = up[j] if at_upper[j] else lo[j]
            if v != 0.0:
                if j < n:
                    for t in range(Ap[j], Ap[j + 1]):
                        rhs[Ai[t]] += Ax[t] * v
                else:
                    rhs[j - n] -= v
    xB[:] = -(Binv @ rhs)


@njit(cache=_CACHE)
def _ray_lagrangian(Ap, Ai, Ax, n, m, lo, up, direction):  # pragma: no cover - compiled; checked through tests
    s = 0.0
    sa = 0.0
    for j in range(n + m):
        dj = -_dot_col(Ap, Ai, Ax, n, j, direction)
        t = min(lo[j] * dj, up[j] * dj)
        s += t
        sa += abs(t)
    return s, sa


@njit(cache=_CACHE)
def _kernel(Ap, Ai, Ax, n, m, cost, lo, up, basic, at_upper, cap, use_cut, cutoff, Binv0, warm, primal_tol,
            primal_rel_tol, dual_tol, pivot_tol, zero_tol, refactor_every, pivot_check_tol, stall_after, perturb_abs,
            perturb_rel, infeas_margin):  # pragma: no cover - compiled; checked through the engine tests
    N = n + m
    width = up - lo
    is_basic = np.zeros(N, dtype=np.bool_)
    for r in range(m):
        is_basic[basic[r]] = True
    work = cost.copy()
    d = np.zeros(N)
    xB = np.zeros(m)
    if warm:
        Binv = Binv0.copy()
        beta = np.empty(m)
        for i in range(m):
            beta[i] = Binv[i] @ Binv[i]
    else:
        Binv, beta = _factor_nb(Ap, Ai, Ax, n, m, basic)
    _recompute_nb(Ap, Ai, Ax, n, m, work, lo, up, width, basic, is_basic, at_upper, Binv, d, xB, dual_tol)
    np.random.seed(0)
    stage = 0
    best = -np.inf
    last_rise = 0
    it = 0
    since = 0
    fresh = True
    status = 1                                           # 0 optimal, 1 iteration_limit, 2 infeasible, 3 cutoff
    ray = np.zeros(m)
    alpha = np.zeros(N)
    col = np.zeros(m)
    abar = np.zeros(m)
    cidx = np.empty(N, dtype=np.int64)
    while True:
        delta = np.zeros(m)
        anyinf = False
        for r in range(m):
            lb = lo[basic[r]]
            ub = up[basic[r]]
            tol = primal_tol + primal_rel_tol * max(abs(lb), abs(ub))
            if xB[r] < lb - tol:
                delta[r] = xB[r] - lb
                anyinf = True
            elif xB[r] > ub + tol:
                delta[r] = xB[r] - ub
                anyinf = True
        if not anyinf:
            if stage == 1:
                work = cost.copy()
                stage = 2
                best = -np.inf
                last_rise = it
                _recompute_nb(Ap, Ai, Ax, n, m, work, lo, up, width, basic, is_basic, at_upper, Binv, d, xB,
                              dual_tol)
                continue
            status = 0
            break
        if it >= cap:
            break
        z = 0.0
        for j in range(N):
            z += min(lo[j] * d[j], up[j] * d[j])
        if use_cut:
            zt = z
            if stage == 1:
                zt = 0.0
                for j in range(N):
                    dj = (d[j] - (work[j] - cost[j])) if not is_basic[j] else -(work[j] - cost[j])
                    zt += min(lo[j] * dj, up[j] * dj)
            if zt >= cutoff:
                status = 3
                break
        if z > best + 1e-12 * (1.0 + (abs(best) if best > -np.inf else 0.0)):
            best = z
            last_rise = it
        elif it - last_rise >= stall_after:
            if stage == 0:
                pert = (perturb_abs + perturb_rel * np.abs(cost[:n])) * 0.5 * (1.0 + np.random.random(n))
                work = cost.copy()
                for j in range(n):
                    if width[j] > 0:
                        work[j] += pert[j]
                stage = 1
                best = -np.inf
                last_rise = it
                _recompute_nb(Ap, Ai, Ax, n, m, work, lo, up, width, basic, is_basic, at_upper, Binv, d, xB,
                              dual_tol)
                continue
            if stage == 2:
                stage = 3
        bland = stage == 3
        r = -1
        if bland:
            bi = N + 1
            for i in range(m):
                if delta[i] != 0.0 and basic[i] < bi:
                    bi = basic[i]
                    r = i
        else:
            bv = -1.0
            for i in range(m):
                if delta[i] != 0.0:
                    v = delta[i] * delta[i] / beta[i]
                    if v > bv:
                        bv = v
                        r = i
        p = basic[r]
        rho = Binv[r].copy()
        sgn = 1.0 if delta[r] > 0 else -1.0
        nc = 0
        for j in range(N):
            alpha[j] = 0.0
            if is_basic[j]:
                continue
            a = _dot_col(Ap, Ai, Ax, n, j, rho)
            alpha[j] = a
            if width[j] > 0:
                at = sgn * a
                if ((not at_upper[j]) and at > pivot_tol) or (at_upper[j] and at < -pivot_tol):
                    cidx[nc] = j
                    nc += 1
        certify = nc == 0
        pick = -1
        if not certify:
            cand = cidx[:nc]
            mag = np.empty(nc)
            ratio = np.empty(nc)
            for t in range(nc):
                mag[t] = abs(alpha[cand[t]])
                ratio[t] = max(abs(d[cand[t]]) / mag[t], 0.0)
            order = np.argsort(ratio, kind="mergesort")
            if bland:
                mn = ratio[order[0]]
                bi = N + 1
                for t in range(nc):
                    if ratio[t] <= mn + zero_tol and cand[t] < bi:
                        bi = cand[t]
                        pick = t
            else:
                slope = abs(delta[r])
                k = -1
                for kk in range(nc):
                    slope -= width[cand[order[kk]]] * mag[order[kk]]
                    if slope <= primal_tol:
                        k = kk
                        break
                if k < 0:
                    certify = True
                else:
                    tmax = np.inf
                    for kk in range(k, nc):
                        o = order[kk]
                        v = (abs(d[cand[o]]) + dual_tol) / mag[o]
                        if v < tmax:
                            tmax = v
                    bm = -1.0
                    for kk in range(k, nc):
                        o = order[kk]
                        if ratio[o] <= tmax and mag[o] > bm:
                            bm = mag[o]
                            pick = o
        if certify:
            s, sa = _ray_lagrangian(Ap, Ai, Ax, n, m, lo, up, sgn * rho)
            if s > infeas_margin * sa:
                status = 2
                ray = sgn * rho
            break
        q = cidx[pick]
        _col_into(Ap, Ai, Ax, n, q, col)
        aq = Binv @ col
        if abs(aq[r] - alpha[q]) > pivot_check_tol * (1.0 + abs(aq[r])) and not fresh:
            Binv, beta = _factor_nb(Ap, Ai, Ax, n, m, basic)
            _recompute_nb(Ap, Ai, Ax, n, m, work, lo, up, width, basic, is_basic, at_upper, Binv, d, xB, dual_tol)
            since = 0
            fresh = True
            continue
        tstep = d[q] / alpha[q]
        abar[:] = 0.0
        anyflip = False
        for j in range(N):
            if is_basic[j] or j == q:
                continue
            d[j] -= tstep * alpha[j]
            if width[j] > 0 and j != p:
                w = 0.0
                if (not at_upper[j]) and d[j] < -dual_tol:
                    at_upper[j] = True
                    w = width[j]
                elif at_upper[j] and d[j] > dual_tol:
                    at_upper[j] = False
                    w = -width[j]
                if w != 0.0:
                    anyflip = True
                    if j < n:
                        for t in range(Ap[j], Ap[j + 1]):
                            abar[Ai[t]] += w * Ax[t]
                    else:
                        abar[j - n] -= w
        d[p] = -tstep
        d[q] = 0.0
        if anyflip:
            xB -= Binv @ abar
        target = up[p] if delta[r] > 0 else lo[p]
        thp = (xB[r] - target) / aq[r]
        xq = (up[q] if at_upper[q] else lo[q]) + thp
        xB -= thp * aq
        xB[r] = xq
        tau = Binv @ rho
        beta_r = rho @ rho
        rq = aq / aq[r]
        newr = Binv[r] / aq[r]
        for i in range(m):
            if i != r:
                Binv[i] -= aq[i] * newr
        Binv[r] = newr
        for i in range(m):
            beta[i] = max(beta[i] - 2.0 * rq[i] * tau[i] + rq[i] * rq[i] * beta_r, 1e-12)
        beta[r] = max(beta_r / (aq[r] * aq[r]), 1e-12)
        at_upper[p] = delta[r] > 0
        is_basic[p] = False
        is_basic[q] = True
        basic[r] = q
        at_upper[q] = False
        it += 1
        since += 1
        fresh = False
        if since >= refactor_every:
            Binv, beta = _factor_nb(Ap, Ai, Ax, n, m, basic)
            _recompute_nb(Ap, Ai, Ax, n, m, work, lo, up, width, basic, is_basic, at_upper, Binv, d, xB, dual_tol)
            since = 0
            fresh = True
    xfull = np.empty(N)
    for j in range(N):
        xfull[j] = up[j] if at_upper[j] else lo[j]
    for r in range(m):
        xfull[basic[r]] = xB[r]
    y = d[n:].copy()
    return status, xfull, y, it, stage, ray, Binv


def _checked_inverse(F, basic, inverse, m):
    """``inverse`` when it is the inverse of the basis matrix ``F[:, basic]``: one residual ``B (B^-1 v) - v`` on a
    fixed vector, within 1e-9 relative. Otherwise None, and the solve factors the basis itself."""
    if inverse is None or m == 0 or np.shape(inverse) != (m, m):
        return None
    v = 1.0 + np.arange(m) / m
    w = np.asarray(inverse, dtype=float) @ v
    res = np.asarray(F[:, basic] @ w).ravel() - v
    if not np.all(np.isfinite(res)) or np.max(np.abs(res)) > 1e-9 * (1.0 + np.max(np.abs(w))):
        return None
    return np.array(inverse, dtype=float)


def _solve_compiled(c, A, rl, ru, l, u, lo, up, cost, basic, at_upper, warm, cap, cutoff, repaired):
    m, n = A.shape
    A = A.tocsc()
    A.sort_indices()
    status, xfull, y, it, stage, ray, Binv = _kernel(
        A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data.astype(float), n, m, cost, lo, up, basic,
        at_upper, cap, cutoff is not None, float(cutoff) if cutoff is not None else 0.0,
        np.ascontiguousarray(warm if warm is not None else np.zeros((m, m))), warm is not None, PRIMAL_TOL,
        PRIMAL_REL_TOL, DUAL_TOL, PIVOT_TOL, ZERO_TOL, REFACTOR_EVERY, PIVOT_CHECK_TOL, STALL_AFTER, PERTURB_ABS,
        PERTURB_REL, INFEASIBILITY_MARGIN)
    status = ("optimal", "iteration_limit", "infeasible", "cutoff")[status]
    x = xfull[:n].copy()
    bound = np.inf if status == "infeasible" else lagrangian_bound(c, A, rl, ru, l, u, y)
    return BoxedLPResult(status=status, x=x, row_activity=np.asarray(A @ x).ravel(), y=y, d=c - A.T @ y,
                         objective=float(c @ x), bound=bound, iterations=int(it),
                         basis=BoxedBasis(basic.copy(), at_upper.copy(), Binv), basis_repaired=bool(repaired),
                         degeneracy_stage=("none", "perturbed", "restored", "bland")[stage],
                         ray=ray if status == "infeasible" else None, engine="numba",
                         warm_inverse_used=warm is not None)


def solve_boxed_lp(c, A, rl, ru, l, u, *, basis: Optional[BoxedBasis] = None,
                   max_iterations: Optional[int] = None, cutoff: Optional[float] = None,
                   engine: Optional[str] = None) -> BoxedLPResult:
    """Minimise ``c'x`` subject to ``rl <= A x <= ru`` and ``l <= x <= u``, every bound finite, by the dual simplex
    of the module docstring, from ``basis`` when given (else the slack basis) and for at most ``max_iterations``
    basis changes (None: ``100 * (m + n) + 1000``, a cap on a solve that should never reach it). With ``cutoff``,
    the solve stops as soon as the Lagrangian bound under the true costs reaches it (Pan, Linear Programming
    Computation, 2nd ed., Lemma 8.3.4, p. 216: the dual objective does not decrease): a branch whose bound reaches the
    incumbent is pruned without solving its LP to the end.

    ``engine`` (``DEFAULT_ENGINE`` when None) is ``"numba"``, the compiled kernel, ``"numpy"``, the reference, or
    ``"auto"``, the kernel when numba is installed. The two run the same algorithm and are tested against each other
    and HiGHS; the kernel updates its vectors incrementally where the reference recomputes them, so their iteration
    counts can differ on ties. A ``basis`` carrying the inverse of its basis matrix for this LP's matrix (one from a
    solve of an LP with the same rows and columns and other bounds) starts without a factorization.

    Raises :class:`MlsynthConfigError` for a non-finite bound, a shape mismatch, a negative iteration limit, a cutoff
    that is not a number, an unknown engine, ``engine="numba"`` without numba or a malformed basis, and
    :class:`MlsynthDataError` for non-finite costs or coefficients or crossed bounds."""
    engine = DEFAULT_ENGINE if engine is None else engine
    if engine not in ("auto", "numpy", "numba"):
        raise MlsynthConfigError(f"engine must be 'auto', 'numpy' or 'numba'; got {engine!r}.")
    if engine == "numba" and not _NUMBA:
        raise MlsynthConfigError("engine='numba' needs numba (the optional 'fast' extra), which is not installed.")
    if cutoff is not None and (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float, np.floating, np.integer))
                               or math.isnan(float(cutoff))):
        raise MlsynthConfigError(f"cutoff must be a number or None; got {cutoff!r}.")
    c, A, rl, ru, l, u = _validate(c, A, rl, ru, l, u, basis, max_iterations)
    m, n = A.shape
    N = n + m
    F = sp.hstack([A, -sp.identity(m, format="csc")], format="csc")      # A x - s = 0
    FT = F.T.tocsr()
    lo = np.concatenate([l, rl])
    up = np.concatenate([u, ru])
    cost = np.concatenate([c, np.zeros(m)])
    width = up - lo
    cap = 100 * (m + n) + 1000 if max_iterations is None else int(max_iterations)

    repaired = False
    warm = None
    if basis is None:
        basic = n + np.arange(m, dtype=np.int64)
        at_upper = np.zeros(N, dtype=bool)
    else:
        basic = np.asarray(basis.basic, dtype=np.int64).copy()
        warm = _checked_inverse(F, basic, basis.inverse, m)
        if warm is None:
            basic, repaired = _repair(F, basic, n, m)
        at_upper = np.asarray(basis.at_upper, dtype=bool).copy()
    if engine == "numba" or (engine == "auto" and _NUMBA):
        return _solve_compiled(c, A, rl, ru, l, u, lo, up, cost, np.asarray(basic, dtype=np.int64).copy(),
                               at_upper.astype(np.bool_), warm, cap, cutoff, repaired)
    is_basic = np.zeros(N, dtype=bool)
    is_basic[basic] = True

    def factor():
        Binv = np.linalg.inv(F[:, basic].toarray()) if m else np.zeros((0, 0))
        return Binv, np.einsum("ij,ij->i", Binv, Binv)               # exact DSE weights: squared row norms

    if warm is not None:
        Binv, beta = warm, np.einsum("ij,ij->i", warm, warm)
    else:
        Binv, beta = factor()
    since = 0
    iterations = 0
    status = "iteration_limit"
    fresh = True
    # Degeneracy (Koberstein, section 6.3). The working costs are perturbed when the dual objective stalls for
    # STALL_AFTER iterations; on reaching an optimum of the perturbed LP the true costs come back, bound flips restore
    # dual feasibility (every variable is boxed) and the dual simplex goes on. Koberstein also perturbs at the start
    # when fewer than n/4 costs are distinct; measured here on 40 equal-cost cover LPs it took 2,569 iterations
    # against 2,485 without, so the solve starts unperturbed. A second stall after
    # that switches to Bland's rule, smallest index first in both choices with the plain ratio test, which cannot
    # cycle (Bland, "New finite pivoting rules for the simplex method", Math. Oper. Res. 2 (1977) 103-107). The
    # bound is always the Lagrangian value under the true costs.
    work = cost.copy()
    rng = np.random.default_rng(0)

    def perturbed() -> np.ndarray:
        xi = (PERTURB_ABS + PERTURB_REL * np.abs(c)) * 0.5 * (1.0 + rng.random(n))
        out = cost.copy()
        out[:n] += np.where(width[:n] > 0, xi, 0.0)
        return out

    stage = 0                                    # 0: true costs; 1: perturbed; 2: restored; 3: Bland's rule
    best, last_rise = -np.inf, 0
    ray = None

    def certified(direction: np.ndarray) -> bool:
        """Whether the row multipliers ``direction`` prove the LP infeasible: their Lagrangian under zero costs
        exceeds the rounding margin."""
        terms = _lagrangian_terms(np.zeros(n), A, rl, ru, l, u, direction)
        return math.fsum(terms.tolist()) > INFEASIBILITY_MARGIN * math.fsum(np.abs(terms).tolist())

    while True:
        y = Binv.T @ work[basic]
        d = work - FT @ y
        d[is_basic] = 0.0
        # dual feasibility by bound flips: every nonbasic variable is boxed (Koberstein, Table 2.2; Algorithm 5)
        nb = ~is_basic & (width > 0)
        at_upper[nb & (d > DUAL_TOL)] = False
        at_upper[nb & (d < -DUAL_TOL)] = True
        xfull = np.where(at_upper, up, lo)
        xfull[is_basic] = 0.0
        xB = -(Binv @ (F @ xfull)) if m else np.zeros(0)
        xfull[basic] = xB
        lb, ub = lo[basic], up[basic]
        tol = PRIMAL_TOL + PRIMAL_REL_TOL * np.maximum(np.abs(lb), np.abs(ub))
        delta = np.where(xB < lb - tol, xB - lb, np.where(xB > ub + tol, xB - ub, 0.0))
        if not np.any(delta):
            if stage == 1:                                            # the perturbed LP's optimum: true costs back
                work, stage, best, last_rise = cost.copy(), 2, -np.inf, iterations
                continue
            status = "optimal"
            break
        if iterations >= cap:
            break
        if cutoff is not None and lagrangian_bound(c, A, rl, ru, l, u, y) >= cutoff:
            status = "cutoff"
            break
        z = float(np.sum(np.minimum(lo * d, up * d)))
        if z > best + 1e-12 * (1.0 + abs(best if np.isfinite(best) else 0.0)):
            best, last_rise = z, iterations
        elif iterations - last_rise >= STALL_AFTER:
            if stage == 0:
                work, stage, best, last_rise = perturbed(), 1, -np.inf, iterations
                continue
            if stage == 2:
                stage = 3
        bland = stage == 3
        infeasible = np.flatnonzero(delta)
        r = int(infeasible[np.argmin(basic[infeasible])]) if bland else int(np.argmax(delta * delta / beta))
        p = int(basic[r])
        rho = Binv[r]
        alpha = FT @ rho                                              # the transformed pivot row
        alpha[is_basic] = 0.0
        sgn = 1.0 if delta[r] > 0 else -1.0
        at = sgn * alpha                                              # Koberstein's alpha-tilde
        cand = np.flatnonzero((width > 0) & ~is_basic & (((~at_upper) & (at > PIVOT_TOL))
                                                         | (at_upper & (at < -PIVOT_TOL))))
        if cand.size == 0:
            if certified(sgn * rho):
                status, ray = "infeasible", sgn * rho
            break
        mag = np.abs(at[cand])
        ratio = np.maximum(np.abs(d[cand]) / mag, 0.0)
        order = np.argsort(ratio, kind="stable")
        slope = abs(delta[r])
        if bland:                                                     # plain ratio test, smallest index among ties
            pick = _bland_entering(cand, ratio)
        else:
            for k in range(order.size):                               # Algorithm 4: pass breakpoints while improving
                slope -= width[cand[order[k]]] * mag[order[k]]
                if slope <= PRIMAL_TOL:                               # zero slope: the row is met at this breakpoint
                    break
            else:                                                     # every breakpoint passed, slope still positive:
                if certified(sgn * rho):                              # the dual is unbounded along rho
                    status, ray = "infeasible", sgn * rho
                break
            rest = order[k:]
            theta_max = np.min((np.abs(d[cand[rest]]) + DUAL_TOL) / mag[rest])
            window = rest[ratio[rest] <= theta_max]                   # Harris: the largest pivot within tolerance
            pick = window[np.argmax(mag[window])] if window.size else order[k]
        q = int(cand[pick])
        col = F[:, q].toarray().ravel()
        aq = Binv @ col                                               # FTran
        if abs(aq[r] - alpha[q]) > PIVOT_CHECK_TOL * (1.0 + abs(aq[r])) and not fresh:
            Binv, beta = factor()                                     # pivot element test (6.27-6.29)
            since, fresh = 0, True
            continue
        # the breakpoints the long step passed are flipped by the dual feasibility pass that opens the next iteration
        tau = Binv @ rho
        beta_r = float(rho @ rho)
        ratio_q = aq / aq[r]
        new_r = Binv[r] / aq[r]
        Binv -= np.outer(aq, new_r)
        Binv[r] = new_r
        beta = np.maximum(beta - 2.0 * ratio_q * tau + ratio_q * ratio_q * beta_r, 1e-12)
        beta[r] = max(beta_r / (aq[r] * aq[r]), 1e-12)
        at_upper[p] = delta[r] > 0                                    # leaves at the bound it violated
        is_basic[p], is_basic[q] = False, True
        basic[r] = q
        at_upper[q] = False
        iterations += 1
        since += 1
        fresh = False
        if since >= REFACTOR_EVERY:
            Binv, beta = factor()
            since, fresh = 0, True

    x = xfull[:n].copy()
    bound = np.inf if status == "infeasible" else lagrangian_bound(c, A, rl, ru, l, u, y)
    return BoxedLPResult(status=status, x=x, row_activity=np.asarray(A @ x).ravel(), y=y, d=c - A.T @ y,
                         objective=float(c @ x), bound=bound, iterations=iterations,
                         basis=BoxedBasis(basic.copy(), at_upper.copy(), Binv.copy()), basis_repaired=bool(repaired),
                         degeneracy_stage=("none", "perturbed", "restored", "bland")[stage], ray=ray,
                         engine="numpy", warm_inverse_used=warm is not None)
