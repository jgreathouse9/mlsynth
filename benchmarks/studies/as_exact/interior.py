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
reaches them all. test_interior.py checks this against every set of K + 1 donors on
small designs.

Predictors are centred on the donors' mean and scaled to unit standard deviation first:
Q_j^A does not depend on the origin (X_j = sum_i beta_ij X_i and sum_i beta_ij = 1), a
diagonal V absorbs the units, and the squares in R stay of one size. The threshold on
the selection slack, 1e-9 on the scaled predictors, is the authors'.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linprog

from . import lsq

THETA_TOL = 1e-9          # the authors' threshold on the selection slack theta*_+, scaled predictors
ADMIT_TOL = 1e-9          # a cell holds X1 when every barycentric weight exceeds this (the authors')
MAX_CELLS = 200_000       # the authors' limit on the cells a walk visits


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
            raise ValueError(f"the donors {self.A} are affinely dependent; the interior case needs "
                             "every K + 1 donors affinely independent (Assumption 2(iii))")
        self.w, self.beta = sol[:, 0], sol[:, 1:]
        self.R = (X0[:, self.out] ** 2).T - self.beta.T @ (X0[:, self.A] ** 2).T

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
    raise ValueError("no cell of K + 1 donors is selected under the identity metric; the treated "
                     "unit or the donors violate the interior general position (Assumption 2)")


def walk_diag(X1, X0, stats, max_cells=MAX_CELLS):
    """The attainable cells, by a walk from the seed that expands only attainable cells.

    Returns {A: (w_A, d)}, d a diagonal metric (summing to 1) under which A is the selection.
    """
    seed = seed_cell(X1, X0)
    found, seen, queue, n = {}, {seed}, deque([seed]), 0
    while queue:
        if n >= max_cells:
            raise RuntimeError(f"the interior walk reached its cell limit of {max_cells} cells before "
                               "visiting every attainable cell, so its best cell need not be the optimum")
        A = queue.popleft()
        n += 1
        cell = Cell(A, X1, X0)
        if cell.w.min() <= ADMIT_TOL:         # pragma: no cover - the ratio test keeps X1 in every neighbour,
            continue                          # strictly under Assumption 2(ii); this guards degenerate data
        ok, d = diag_attainable(cell.R, stats)
        if not ok:
            continue
        found[cell.A] = (cell.w, d)
        for B in cell.neighbours():
            if B not in seen:
                seen.add(B)
                queue.append(B)
    stats["cells"] = stats.get("cells", 0) + n
    return found


def solve_interior_diag(X1, X0, Z1, Z0, stats=None, max_cells=MAX_CELLS):
    """The best attainable exact match under diagonal V: (loss, w) with w on the simplex over the J donors."""
    X1, X0, Z1, Z0 = (np.asarray(a, float) for a in (X1, X0, Z1, Z0))
    if X0.ndim != 2 or Z0.ndim != 2 or X1.shape != (X0.shape[0],) or Z1.shape != (Z0.shape[0],) \
            or X0.shape[1] != Z0.shape[1]:
        raise ValueError("expected X1 (K,), X0 (K, J), Z1 (L,), Z0 (L, J)")
    if not all(np.isfinite(a).all() for a in (X1, X0, Z1, Z0)):
        raise ValueError("inputs must be finite")
    C1, C0 = _standardised(X1, X0)
    if not interior_margin(C1, C0) > ADMIT_TOL:
        raise ValueError("the interior case needs the treated unit strictly inside the donor hull, "
                         "matched by no fewer than K + 1 donors; use exact.solve_diag outside it")
    st = {}
    cells = walk_diag(C1, C0, st, max_cells)
    if not cells:
        raise ValueError("the seed cell is not attainable, so the walk has nowhere to start; the "
                         "donors violate the interior general position (Assumption 2(iv))")
    losses = {A: float(np.sum((Z1 - Z0[:, list(A)] @ w) ** 2)) for A, (w, _) in cells.items()}
    best = min(losses, key=losses.get)
    w = np.zeros(X0.shape[1])
    w[list(best)] = cells[best][0]
    if stats is not None:
        stats.update(st, attainable_cells=len(cells), cell=best)
    return losses[best], w
