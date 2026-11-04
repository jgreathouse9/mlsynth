"""Spike: the shared-slope relaxation of TBR's control-group problem (#965).

For a fixed treatment group T (aggregate y, centred) and candidate controls with
centred series X, the best control group solves
    min_{C != {}, b} || y - b X u_C ||^2,
every selected control carrying the same coefficient b. This is the objective of
time-based regression's design search with T fixed: Kerman, Wang and Vaver
(2017), eq. (1), fit the treatment aggregate on the control aggregate, and Au
(2018), Algorithm 1, ranks designs by the required impact, proportional to the
residual scale sqrt(RSS / (n - 2)).

With w = b u_C, slope b in [lo, hi] and |C| = m, every feasible w lies in
    { 0 <= w <= hi,  m lo <= sum(w) <= m hi }.
Derived here: w_j = b u_j is in [0, b] and sum(w) = b m. For lo = hi the set is
the convex hull of the size-m designs at slope b, the hull of a union of points
(the hull of a disjunction, Balas 1985). The minimum of the convex
QP over that set, over a cover of slopes and every m, is a lower bound on the
best control group's RSS; the gates and the correlation bucket only remove
designs, so the bound stays valid.

Balas, E. (1985). Disjunctive programming and a hierarchy of relaxations for
    discrete optimization problems. SIAM J. Algebr. Discrete Methods 6, 466-486.
Kerman, J., Wang, P. and Vaver, J. (2017). Estimating ad effectiveness using geo
    experiments in a time-based regression framework. Google technical report.
Au, T. C. (2018). A time-based regression matched markets approach for designing
    geo experiments. Google technical report.
"""
import numpy as np
import cvxpy as cp

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.solvers.nnls import nnls


def rss_of(y, X, idx):
    x = X[:, list(idx)].sum(1)
    xx = x @ x
    return float(y @ y - (y @ x) ** 2 / xx) if xx > 0 else float(y @ y)


class ControlBound:
    def __init__(self, y, X):
        self.scale = float(np.linalg.norm(y))
        self.y, self.X = y / self.scale, X / self.scale
        p = X.shape[1]
        self.w = cp.Variable(p)
        self.cap = cp.Parameter(nonneg=True)
        self.lo = cp.Parameter(nonneg=True)
        self.hi = cp.Parameter(nonneg=True)
        self.prob = cp.Problem(cp.Minimize(cp.sum_squares(self.y - self.X @ self.w)),
                               [self.w >= 0, self.w <= self.cap,
                                cp.sum(self.w) >= self.lo, cp.sum(self.w) <= self.hi])
        self.solves = 0

    def piece(self, b_lo, b_hi, m, big=1e9):
        """Lower bound over slopes in [b_lo, b_hi] and |C| = m."""
        self.cap.value = min(b_hi, big)
        self.lo.value = m * b_lo
        self.hi.value = min(m * b_hi, big)
        self.prob.solve(solver=cp.CLARABEL)
        self.solves += 1
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return np.inf        # infeasible piece contributes nothing
        return float(self.prob.value) * self.scale ** 2



class Band:
    """The hull without its per-geo cap: min ||y - X w||^2 over {w >= 0, A <= sum w <= B}.

    It relaxes the capped hull, so it is a weaker bound and a cheaper one: no
    conic solver. One NNLS over w >= 0 (Lawson and Hanson 1974, the library's
    mlsynth.utils.solvers.nnls) gives the minimum when its total s lies in the
    band. Otherwise the minimum lies on the nearer end, s = A or s = B (derived
    here: f is convex, so if the minimiser over w >= 0 has total below A and a
    feasible point with total above A were optimal, the segment between them
    would cross sum w = A at a point no worse, by convexity, and strictly
    better near the optimum), and there it is a simplex QP in v = w / s,
    min ||y - s X v||^2 over the simplex, by the library's
    mlsynth.utils.solvers.active_set.solve_simplex_qp. The ends m * edge recur
    across the pieces on either side of an edge, so each is solved once and
    cached, warm-started from the last.

    Lawson, C. L. and Hanson, R. J. (1974). Solving Least Squares Problems.
        Prentice-Hall.
    """

    def __init__(self, y, X):
        self.scale = float(np.linalg.norm(y))
        self.y, self.X = y / self.scale, X / self.scale
        w, _ = nnls(self.X, self.y)
        self.w_nnls = w
        self.s_nnls = float(w.sum())
        self.f_nnls = float(np.sum((self.y - self.X @ w) ** 2)) * self.scale ** 2
        self._ends = {}
        self._warm = None
        self.solves = 0

    def _at(self, s):
        if s not in self._ends:
            v = solve_simplex_qp(s * self.X, self.y, warm_start=self._warm)
            self._warm = v
            w = s * v
            self._ends[s] = (float(np.sum((self.y - self.X @ w) ** 2)) * self.scale ** 2, w)
            self.solves += 1
        return self._ends[s]

    def value(self, lo, hi, m):
        """The band's minimum for slopes in [lo, hi] and size m, and its minimiser."""
        A, B = m * lo, m * hi
        if A <= self.s_nnls <= B:
            return self.f_nnls, self.w_nnls
        return self._at(A if self.s_nnls < A else B)
