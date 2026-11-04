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

