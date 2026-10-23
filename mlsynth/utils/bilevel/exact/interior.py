"""The interior case (paper, Section 5): exact matches selected by a vanishing penalty.

When X1 lies inside the donor hull, every positive definite V leaves the same exact
matches F = {w in the simplex : X0 w = X1}. The vanishing penalty of Abadie and L'Hour
(2021) selects among them the minimiser of sum_j w_j ||X1 - X_j||_V^2, a linear program
over F whose vertices are the admissible cells: the sets A of K + 1 donors whose simplex
holds X1, each with X1's barycentric weights w_A. Cell A is the selection under V
exactly when every outside donor's lift (X_j, X_j'V X_j) lies above the hyperplane
through the lifted cell, <Q_j^A, V> > 0 with Q_j^A = X_j X_j' - sum_i beta_ij X_i X_i'
and beta_j the barycentric coordinates of X_j in A: A is then the Delaunay cell of the
transformed donors that holds X1 (Lemma C.2). For V = Diag(d) the rows are
rho_j = diag(Q_j^A), and A is attainable when some d >= 0 makes every rho_j'd positive.
The answer is the attainable cell of least validation loss (Proposition 5).

A cell costs one solve with [X0_A; 1']: X1's weights and every outside donor's beta
come from it at once, the rows R = (X_out^2)' - beta'(X_A^2)' are one product, the
neighbours (the cells that trade one donor for an outside one and still hold X1) are
one ratio test per outside donor, and attainability is one least-distance problem,
R d >= 1 with d >= 0, whose verdict carries a point or a Farkas certificate
(lsq.ldp_verdict). The authors' code (synthx, interior_cellwalk.py and
interior_vsearch.py) solves one HiGHS LP per cell and finds each neighbour by trying
the exits in turn.

The attainable cells are connected by these flips. As d moves along a path in the
open positive orthant, the selected cell changes only where K + 2 transformed donors
are cospherical, and there it trades one donor (the bistellar flip of the circuit they
form), so a walk from one attainable cell that expands the attainable cells it meets
reaches them all. mlsynth/tests/test_exact_interior.py checks this against every set of
K + 1 donors on small designs.

Under positive definite V the attainability test is the paper's semidefinite program,
<Q_j^A, V> > 0 for every outside j with V >= 0, and most admissible cells pass it (83
to 97 per cent of a sample on Proposition 99's interior placebos, where under diagonal
V at most 2 per cent do). Walking only the attainable cells would save little, so every
admissible cell is listed, as the authors' code lists them, and the cells are tested in
increasing order of loss: the first attainable one is the answer, and on those placebos
it is the first one tested. The test runs the barrier method (Boyd and Vandenberghe,
ch. 11) from the metric under which the cell is a regular simplex, and stops at the
first evidence that checks: a positive definite V with every slack above the threshold,
or a point y of the simplex with K lambda_max(sum_j y_j Q_j) at or below it, which
bounds the program's value by duality. cvxpy, the authors' solver, decides the rare
test where neither checks.

Predictors are centred on the donors' mean and scaled to unit standard deviation first:
Q_j^A does not depend on the origin (X_j = sum_i beta_ij X_i and sum_i beta_ij = 1), a
diagonal V absorbs the units, and the squares in R stay of one size. The threshold on
the selection slack, 1e-9 on the scaled predictors, is the authors'.
"""
from __future__ import annotations

from collections import deque
from functools import cached_property

import numpy as np
from scipy.optimize import linprog

from ....exceptions import MlsynthDataError, MlsynthEstimationError
from . import lsq, rationalize

THETA_TOL = 1e-9          # the authors' threshold on the selection slack theta*_+, scaled predictors
ADMIT_TOL = 1e-9          # a cell holds X1 when every barycentric weight exceeds this (the authors')
MAX_CELLS = 200_000       # the authors' limit on the cells a walk visits
BARRIER_T = (1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12)   # barrier parameters, each point centred by Newton


def _standardised(X1, X0):
    """Predictors centred on the donors' mean and scaled by the pooled standard deviation."""
    sd = np.column_stack([X1, X0]).std(axis=1, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    c = X0.mean(axis=1)
    return (X1 - c) / sd, (X0 - c[:, None]) / sd[:, None]


def interior_margin(X1, X0):
    """max t such that X1 = X0 w with w on the simplex and every w_j >= t; -inf outside the hull."""
    K, J = X0.shape
    res = linprog(np.r_[np.zeros(J), -1.0], A_ub=np.c_[-np.eye(J), np.ones(J)], b_ub=np.zeros(J),
                  A_eq=np.c_[np.vstack([X0, np.ones(J)]), np.zeros(K + 1)], b_eq=np.append(X1, 1.0),
                  bounds=[(0, None)] * J + [(None, None)], method="highs")
    return -res.fun if res.status == 0 else -np.inf


class Cell:
    """Cell A: X1's barycentric weights w, the outside donors, their barycentric coordinates
    beta (one column each) and the rows R of the diagonal selection inequalities."""

    def __init__(self, A, X1, X0):
        self.A = tuple(sorted(int(i) for i in A))
        K, J = X0.shape
        self.out = [j for j in range(J) if j not in self.A]
        M = np.vstack([X0[:, self.A], np.ones(K + 1)])
        rhs = np.vstack([np.c_[X1, X0[:, self.out]], np.ones(1 + len(self.out))])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = None
        if sol is None or np.abs(M @ sol - rhs).max() > 1e-8 * max(1.0, float(np.abs(rhs).max())):
            raise MlsynthDataError(f"the donors {self.A} are affinely dependent; the interior case needs "
                             "every K + 1 donors affinely independent (Assumption 2(iii))")
        self.w, self.beta = sol[:, 0], sol[:, 1:]
        self.X0 = X0

    @cached_property
    def R(self):
        return (self.X0[:, self.out] ** 2).T - self.beta.T @ (self.X0[:, self.A] ** 2).T

    def neighbours(self):
        """The cells that trade one donor of A for an outside donor j and still hold X1.

        Moving X1's weight along the circuit A + {j}, the donor that leaves is the first to
        reach zero: the least w_i / beta_ij over beta_ij > 0. Some beta_ij is positive, since
        they sum to 1.
        """
        with np.errstate(divide="ignore"):
            ratio = np.where(self.beta > 0, self.w[:, None] / self.beta, np.inf)
        leave = ratio.argmin(axis=0)
        out = []
        for col, j in enumerate(self.out):
            B = list(self.A)
            B[leave[col]] = j
            out.append(tuple(sorted(B)))
        return out


def diag_attainable(R, stats):
    """Whether some d >= 0 makes every rho_j'd positive: (True, d) with d summing to 1, or (False, None).

    R d >= 1 with d >= 0 is one least-distance problem (lsq.ldp_verdict). Its point counts
    when the normalised slack min_j rho_j'd / 1'd clears THETA_TOL, the authors' threshold
    on theta*_+; its Farkas certificate u has R'u_R <= 0 with 1'u_R > 0, so no d >= 0 has
    R d > 0. HiGHS solves the authors' program theta*_+ when neither decides.
    """
    stats["cell_tests"] = stats.get("cell_tests", 0) + 1
    n, K = R.shape
    if n == 0:                                         # no outside donor rules the cell out
        return True, np.full(K, 1.0 / K)
    verdict, evidence = lsq.ldp_verdict(np.vstack([R, np.eye(K)]), np.r_[np.ones(n), np.zeros(K)])
    if verdict is True:
        d = np.maximum(evidence, 0.0)
        if (R @ d).min() > THETA_TOL * d.sum():
            return True, d / d.sum()
    elif verdict is False:
        return False, None
    stats["cell_lp_checks"] = stats.get("cell_lp_checks", 0) + 1
    res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[-R, np.ones(n)], b_ub=np.zeros(n),
                  A_eq=np.r_[np.ones(K), 0.0][None, :], b_eq=[1.0],
                  bounds=[(0, None)] * K + [(None, None)], method="highs")
    if res.status == 0 and -res.fun > THETA_TOL:
        return True, res.x[:K]
    return False, None


def seed_cell(X1, X0):
    """The cell selected under V = I: the vertex of F where sum_j w_j ||X_j||^2 is least (one LP).

    A tie among cospherical donors can leave a vertex with fewer or more than K + 1 donors;
    the metric is then moved off the tie, deterministically, as the authors' code does.
    """
    K, J = X0.shape
    lift = np.sum(X0 ** 2, axis=0)
    golden = np.mod(np.arange(1, J + 1) * (1.0 + 5.0 ** 0.5) / 2.0, 1.0) - 0.5
    for attempt in range(4):
        res = linprog(lift * (1.0 + 1e-6 * attempt * golden), A_eq=np.vstack([X0, np.ones(J)]),
                      b_eq=np.append(X1, 1.0), bounds=(0, None), method="highs")
        support = tuple(int(j) for j in np.flatnonzero(res.x > 1e-9)) if res.status == 0 else ()
        if len(support) == K + 1:
            return support
    raise MlsynthDataError("no cell of K + 1 donors is selected under the identity metric; the treated "
                     "unit or the donors violate the interior general position (Assumption 2)")


def _walk(X1, X0, stats, max_cells, select):
    """A breadth-first walk over the flips from the seed cell: {A: select(cell)}.

    A cell for which ``select`` returns None is neither kept nor expanded.
    """
    seed = seed_cell(X1, X0)
    found, seen, queue, n = {}, {seed}, deque([seed]), 0
    while queue:
        if n >= max_cells:
            raise MlsynthEstimationError(f"the interior walk reached its cell limit of {max_cells} cells before "
                               "visiting every cell it must, so its best cell need not be the optimum")
        A = queue.popleft()
        n += 1
        cell = Cell(A, X1, X0)
        if cell.w.min() <= ADMIT_TOL:         # pragma: no cover - the ratio test keeps X1 in every neighbour,
            continue                          # strictly under Assumption 2(ii); this guards degenerate data
        kept = select(cell)
        if kept is None:
            continue
        found[cell.A] = kept
        for B in cell.neighbours():
            if B not in seen:
                seen.add(B)
                queue.append(B)
    stats["cells"] = stats.get("cells", 0) + n
    return found


def walk_diag(X1, X0, stats, max_cells=MAX_CELLS):
    """The attainable cells, by a walk from the seed that expands only attainable cells.

    Returns {A: (w_A, d)}, d a diagonal metric (summing to 1) under which A is the selection.
    """
    def select(cell):
        ok, d = diag_attainable(cell.R, stats)
        return (cell.w, d) if ok else None
    return _walk(X1, X0, stats, max_cells, select)


def walk_admissible(X1, X0, stats, max_cells=MAX_CELLS):
    """Every admissible cell, {A: w_A}. The flips are the edges of the polytope of exact matches
    and its vertices are the cells, so the walk meets them all: a polytope's graph is connected."""
    return _walk(X1, X0, stats, max_cells, lambda cell: cell.w)


def _prepared(X1, X0, Z1, Z0):
    """Checked inputs and the standardised predictors, refused outside the interior case."""
    X1, X0, Z1, Z0 = (np.asarray(a, float) for a in (X1, X0, Z1, Z0))
    if X0.ndim != 2 or Z0.ndim != 2 or X1.shape != (X0.shape[0],) or Z1.shape != (Z0.shape[0],) \
            or X0.shape[1] != Z0.shape[1]:
        raise MlsynthDataError("expected X1 (K,), X0 (K, J), Z1 (L,), Z0 (L, J)")
    if not all(np.isfinite(a).all() for a in (X1, X0, Z1, Z0)):
        raise MlsynthDataError("inputs must be finite")
    C1, C0 = _standardised(X1, X0)
    if not interior_margin(C1, C0) > ADMIT_TOL:
        raise MlsynthDataError("the treated unit is on the boundary of the donor hull or outside it; the "
                         "interior case needs it strictly inside the donor hull, matched by no fewer "
                         "than K + 1 donors (Assumption 2(ii)), and the exterior search takes a unit outside it")
    return X1, X0, Z1, Z0, C1, C0


def solve_interior_diag(X1, X0, Z1, Z0, stats=None, max_cells=MAX_CELLS):
    """The best attainable exact match under diagonal V: (loss, w) with w on the simplex over the J donors."""
    X1, X0, Z1, Z0, C1, C0 = _prepared(X1, X0, Z1, Z0)
    st = {}
    cells = walk_diag(C1, C0, st, max_cells)
    if not cells:
        raise MlsynthDataError("the seed cell is not attainable, so the walk has nowhere to start; the "
                         "donors violate the interior general position (Assumption 2(iv))")
    losses = {A: float(np.sum((Z1 - Z0[:, list(A)] @ w) ** 2)) for A, (w, _) in cells.items()}
    best = min(losses, key=losses.get)
    w = np.zeros(X0.shape[1])
    w[list(best)] = cells[best][0]
    if stats is not None:
        stats.update(st, attainable_cells=len(cells), cell=best)
    return losses[best], w


def q_matrices(cell, X0):
    """The paper's Q_j^A = X_j X_j' - sum_i beta_ij X_i X_i' for the donors outside the cell, (n, K, K)."""
    XO, XA = X0[:, cell.out], X0[:, list(cell.A)]
    return np.einsum("kj,lj->jkl", XO, XO) - np.einsum("ki,ij,li->jkl", XA, cell.beta, XA)


def _steiner(XA):
    """The metric under which the simplex with vertices XA is regular: the inverse of their scatter."""
    D = XA - XA.mean(axis=1, keepdims=True)
    return np.linalg.inv(D @ D.T)


def _pd_barrier(Q, V0, stats):
    """max theta with <Q_j, V> >= theta, tr V = K and V >= 0, by the barrier method from V0, stopping at
    the first evidence that checks: (True, V), (False, y) or, when neither does, (None, None)."""
    n, K = Q.shape[0], Q.shape[1]
    V0 = V0 * (K / np.trace(V0))
    if np.einsum("jkl,kl->j", Q, V0).min() > THETA_TOL:      # the start already selects the cell
        return True, V0
    B = rationalize._basis(K)
    p = len(B)
    Bm = B.reshape(p, K * K)
    G = Q.reshape(n, K * K) @ Bm.T                            # <Q_j, B_a>
    iu = np.triu_indices(K)
    A = np.r_[np.einsum("akk->a", B), 0.0][None, :]           # tr V = K
    N = rationalize._null(A)

    def mat(v):
        return (v @ Bm).reshape(K, K)

    def in_domain(x):
        if (G @ x[:p] - x[p]).min() <= 0.0:
            return False
        try:
            np.linalg.cholesky(mat(x[:p]))
        except np.linalg.LinAlgError:
            return False
        return True

    s0 = G @ V0[iu]
    x = np.r_[V0[iu], s0.min() - max(1.0, abs(s0.min()))]    # theta strictly below every slack
    for t in BARRIER_T:
        def fun(x, t=t):
            if not in_domain(x):
                return np.inf
            return -t * x[p] - np.log(G @ x[:p] - x[p]).sum() - np.linalg.slogdet(mat(x[:p]))[1]

        def vgh(x, t=t):
            v, th = x[:p], x[p]
            s = G @ v - th
            V = mat(v)
            Vi = np.linalg.inv(V)
            WB = Vi @ B
            g = np.r_[-G.T @ (1.0 / s) - np.einsum("akl,lk->a", B, Vi), -t + (1.0 / s).sum()]
            H = np.zeros((p + 1, p + 1))
            H[:p, :p] = (G.T / s ** 2) @ G + np.einsum("akl,blk->ab", WB, WB)
            H[:p, p] = H[p, :p] = -G.T @ (1.0 / s ** 2)
            H[p, p] = (1.0 / s ** 2).sum()
            return -t * th - np.log(s).sum() - np.linalg.slogdet(V)[1], g, H

        x = rationalize._newton_eq(x, vgh, A, np.array([float(K)]), in_domain, fun=fun, N=N)
        V, slack = mat(x[:p]), G @ x[:p]
        if slack.min() > THETA_TOL and np.linalg.eigvalsh(V).min() > 0.0:
            return True, V
        y = 1.0 / (slack - x[p])
        y /= y.sum()
        if K * np.linalg.eigvalsh(np.einsum("j,jkl->kl", y, Q)).max() <= THETA_TOL:
            return False, y
    return None, None


def pd_attainable(Q, stats, V0=None):
    """Whether some positive definite V makes every <Q_j, V> positive.

    Returns (True, V) with tr V = K and every slack above THETA_TOL, (False, y) with y on the
    simplex and K lambda_max(sum_j y_j Q_j) <= THETA_TOL, which bounds the program's value,
    or (False, None) and (True, None) when cvxpy, the authors' solver, decided.
    """
    stats["pd_tests"] = stats.get("pd_tests", 0) + 1
    n, K = Q.shape[0], Q.shape[1]
    if n == 0:                                         # no outside donor rules the cell out
        return True, np.eye(K)
    ok, evidence = _pd_barrier(Q, np.eye(K) if V0 is None else V0, stats)
    if ok is not None:
        return ok, evidence
    stats["pd_sdp_checks"] = stats.get("pd_sdp_checks", 0) + 1
    import cvxpy as cp
    V, slack = cp.Variable((K, K), PSD=True), cp.Variable()
    prob = cp.Problem(cp.Maximize(slack), [cp.trace(V) == K] + [cp.trace(Qj @ V) >= slack for Qj in Q])
    prob.solve(solver=cp.CLARABEL)
    return (True, None) if prob.value is not None and prob.value > THETA_TOL else (False, None)


def solve_interior_pd(X1, X0, Z1, Z0, stats=None, max_cells=MAX_CELLS):
    """The best attainable exact match under positive definite V: (loss, w) with w on the simplex over the J donors."""
    X1, X0, Z1, Z0, C1, C0 = _prepared(X1, X0, Z1, Z0)
    st = {}
    cells = walk_admissible(C1, C0, st, max_cells)
    losses = {A: float(np.sum((Z1 - Z0[:, list(A)] @ w) ** 2)) for A, w in cells.items()}
    for tested, A in enumerate(sorted(losses, key=losses.get), start=1):
        cell = Cell(A, C1, C0)
        ok, _ = pd_attainable(q_matrices(cell, C0), st, V0=_steiner(C0[:, list(A)]))
        if ok:
            w = np.zeros(X0.shape[1])
            w[list(A)] = cells[A]
            if stats is not None:
                stats.update(st, admissible_cells=len(cells), tested_cells=tested, cell=A)
            return losses[A], w
    raise MlsynthDataError("no admissible cell is attainable under a positive definite V; the donors violate "
                     "the interior general position (Assumption 2(iv))")
