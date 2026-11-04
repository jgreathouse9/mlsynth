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

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.solvers.nnls import nnls


def rss_of(y, X, idx):
    x = X[:, list(idx)].sum(1)
    xx = x @ x
    return float(y @ y - (y @ x) ** 2 / xx) if xx > 0 else float(y @ y)


def positive_rss(y, X, idx):
    """The best RSS over positive slopes: y'y - (y'x)^2 / x'x when y'x > 0, else y'y.

    A design whose best-fit slope is not positive fails the correlation gate, and
    its infimum over positive slopes is the limit b -> 0+, the total y'y (derived
    here: ||y - b x||^2 is convex in b with its minimum at b* = y'x / x'x <= 0, so
    it increases on b > 0). The incumbent search scores designs this way, so it
    never settles on one the certificate's slope pieces, which cover (0, cap],
    do not cover.
    """
    x = X[:, list(idx)].sum(1)
    xx, xy = float(x @ x), float(y @ x)
    return float(y @ y - xy * xy / xx) if xx > 0 and xy > 0 else float(y @ y)


class ControlBound:
    """The capped hull: min ||y - X w||^2 over {0 <= w <= hi, A <= sum w <= B}.

    One call of the library's capped simplex QP,
    mlsynth.utils.solvers.active_set.solve_simplex_qp(..., upper=...), an
    active-set method with bounds on the variables (Nocedal and Wright 2006,
    Sect. 16.5). Derived here: a slack weight w0 with a zero column, capped at
    B - A, makes the band an equality, sum w + w0 = B, since then sum w lies in
    [A, B] exactly when w0 lies in [0, B - A]; in v = (w, w0) / B the set is the
    capped simplex {0 <= v <= u, sum v = 1} with u = (hi / B, ..., hi / B,
    (B - A) / B), and the objective is ||y - (B X, 0) v||^2. The caps sum to
    (p hi - A) / B + 1, at least one whenever p hi >= A, which holds for every
    size m <= p since A = m lo <= m hi. The solve is exact, so the value is the
    minimum and the point the minimiser, to rounding; the stage still prunes
    only on the first-order certificate at the point (certificate.hull_bound).

    Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, 2nd ed.
        Springer.
    """

    def __init__(self, y, X):
        self.scale = float(np.linalg.norm(y))
        self.y, self.X = y / self.scale, X / self.scale
        self.p = X.shape[1]
        self._Z = np.zeros((X.shape[0], self.p + 1))
        self.point = None
        self._warm = None
        self.solves = 0

    def piece(self, b_lo, b_hi, m):
        """Lower bound over slopes in [b_lo, b_hi] and |C| = m; the minimiser is
        left in ``self.point`` (None for an infeasible piece)."""
        A, B, hi = m * b_lo, m * b_hi, b_hi
        self.solves += 1
        if A > B or self.p * hi < A:
            self.point = None
            return np.inf        # infeasible piece contributes nothing
        if B <= 0:               # hi = 0: the set is the origin
            self.point = np.zeros(self.p)
            return float(self.y @ self.y) * self.scale ** 2
        self._Z[:, :self.p] = B * self.X
        upper = np.full(self.p + 1, hi / B)
        upper[-1] = (B - A) / B
        # p hi >= A makes the caps sum to at least one exactly, but in floating
        # point m = p and lo = hi can round them below (six caps of 0.1 / 0.6 sum
        # to 1 - 1.1e-16). The slack takes the rounding: a relaxation of order eps,
        # which the certificate at the point (exact arithmetic) does not rely on.
        while upper.sum() < 1.0:
            upper[-1] = np.nextafter(upper[-1] + (1.0 - upper.sum()), np.inf)
        v = solve_simplex_qp(self._Z, self.y, upper=upper, warm_start=self._warm)
        self._warm = v
        self.point = B * v[:self.p]
        r = self.y - self.X @ self.point
        return float(r @ r) * self.scale ** 2


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
