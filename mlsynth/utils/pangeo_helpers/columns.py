"""PANGEO's exact design without handing every candidate group to the MIP.

The exact program (:mod:`.mip`) chooses an exact cover of an arm's units by groups of 2 to 2Q units, each group
costing its best split (:func:`.parallelism.best_split`). There are :math:`\\sum_{m=2}^{2Q}\\binom{n}{m}` groups. This
module scores all of them in compiled code, bounds the optimum from below by the root linear program, and gives the
MIP only the groups that bound cannot rule out. Every design whose exact cost is at most the returned design's is
in the last MIP; that MIP's optimum is HiGHS's, on the computed costs, to its tolerances.

Scoring. A split of a group into halves A and B has gap :math:`g = \\bar Y_A - \\bar Y_B = \\sum_i c_i y_i`, with
:math:`c_i = u_i / U_A` on A and :math:`-u_i / U_B` on B (:math:`u` the unit weights, :math:`U` their sums over a half).
Its cost is the residual sum of squares of fitting a level to that gap,
:math:`\\min_\\delta \\|g - \\delta\\mathbf 1\\|^2` (weighted: :math:`\\min_\\delta \\sum_t w_t (g_t - \\delta)^2`), a
least squares problem, and it is computed the way least squares problems are solved. Each unit's level is removed
first (the two-pass algorithm: Bjorck, "Numerical Methods in Matrix Computations", Springer, 2015, eqs. 2.1.28 to
2.1.31; Chan, Golub and LeVeque, "Algorithms for computing the sample variance", Amer. Statist. 37 (1983) 241-247),
the split's gap is formed directly from the centred paths, its level is removed again, and its squares are summed.
The cost is never formed as :math:`c^\\top A^\\top A c` from the cross-product matrix: that is the method of normal
equations, whose error grows with the square of the conditioning (Golub and Van Loan, "Matrix Computations", 4th
ed., sections 5.3.2 and 5.3.7; Bjorck, "Numerical Methods for Least Squares Problems", SIAM, 1996, section 2.2.1).
On panels whose units share a large common factor and run nearly parallel in clusters, it gave the costs that decide
the design relative errors near 0.4, some negative, and a design above the exact optimum on 14 of 16 panels.
Removing the levels is exact for the cost: a constant added to a
unit's path moves the gap by a constant, which the fitted level absorbs, and a constant added to a covariate for every
unit leaves each difference of half means unchanged, since the weights c sum to zero. The covariate term is the
weighted squared difference of the halves' arm-centred covariate means, formed directly; the R^2 denominator is half
A's level-removed residual, formed the same way; the supergeo_total cost is the square of a difference of totals (Chen,
Doudchenko, Jiang, Stein and Ying, arXiv:2301.12044, eq. 3).

Every split's exact cost (in exact arithmetic on the panel's doubles) is also bounded from below and above. The gap's
error is at most :math:`k_1(m) \\sum_p |c_p| \\|A_p\\|_2`, from the error of a dot product, of the weights and of the
centring (Rump, "Verification methods: rigorous results using floating-point arithmetic", Acta Numerica 19 (2010)
287-449, Theorem 2.1); the level fit is an orthogonal projection, which does not lengthen that error; and the sum of
squares and the mean carry their own :math:`\\gamma_{T+6}` and :math:`\\gamma_{2T+4}` terms. That theorem's product
bounds hold provided no underflow occurs; below the normal range a product, quotient or square carries an absolute
error of at most :math:`\\eta / 2`, :math:`\\eta = 2^{-1074}`, and no relative one (Rump 2010, eq. 2.3 on p. 302,
:math:`a \\circ b = \\mathrm{fl}(a \\circ b)(1 + \\varepsilon) + \\delta`, :math:`\\varepsilon\\delta = 0`). Each such
operation on a bound's path is followed by adding :math:`\\eta` to an upper bound or subtracting it from a lower one,
and a sum of :math:`T` such terms by :math:`T\\eta`: exact when the result is subnormal, since addition then does not
round (eq. 2.4), and rounded away when it is normal, where :math:`\\delta = 0`. The bounds therefore hold at every
scale, and at normal scales they are the same doubles as without the term (jgreathouse9/mlsynth#1170). A group's
lower bound is the least over its splits, its upper bound that of
the split chosen. Groups come out in the enumeration's order, by size and then lexicographically, and splits are tried
in :func:`.parallelism.best_split`'s order with the first minimum kept; splits that tie in exact arithmetic are
separated by rounding.

Bound. Let :math:`\\pi` be duals for the cover rows and :math:`\\sigma \\ge 0` one for the row
:math:`\\sum_G x_G \\ge k`. For every design :math:`x`, :math:`c^\\top x = \\pi^\\top \\mathbf 1 + \\sigma k +
\\sum_G \\bar c_G x_G + \\sigma(\\mathbf 1^\\top x - k)` with reduced costs :math:`\\bar c_G = c_G - \\sum_{i \\in G}
\\pi_i - \\sigma`, so

.. math::

   c^\\top x \\;\\ge\\; L + \\max(\\bar c_H, 0) \\quad\\text{for every group } H \\text{ in } x,\\qquad
   L = \\pi^\\top \\mathbf 1 + \\sigma k + \\sum_G \\min(\\bar c_G, 0),

for any duals: the Lagrangian bound of the set-partitioning program with :math:`x \\in [0, 1]` (Geoffrion,
"Lagrangean relaxation for integer programming", Math. Programming Study 2 (1974) 82-114; Lubbecke and Desrosiers,
"Selected topics in column generation", Oper. Res. 53 (2005) 1007-1023, section 2.1). A group with
:math:`\\bar c_H > z - L` is in no design cheaper than :math:`z`, which is reduced-cost fixing (Nemhauser and Wolsey,
"Integer and Combinatorial Optimization", Wiley, 1988, section II.1.4). The duals come from the root LP, solved by
column generation over the scored groups. The LP's duals need not be accurate, since the bound holds for any duals,
provided it is evaluated rigorously (Jansson, "Rigorous lower and upper bounds in linear programming", SIAM J. Optim.
14 (2004) 914-935; Neumaier and Shcherbina, "Safe bounds in linear and mixed-integer linear programming", Math.
Program. 99 (2004) 283-296): the reduced costs are formed from the costs' lower bounds less their own rounding, and
L is summed with its rounding subtracted and rounded down. The search compares the margin z - L exactly, with z the
sum of the design's upper bounds. The row :math:`\\sum_G x_G \\ge k` holds with
:math:`k = \\max(\\kappa_{\\min}, \\lceil n / 2Q \\rceil)`: summing the cover rows with multiplier :math:`1/2Q` gives
:math:`\\sum_G (|G| / 2Q) x_G = n / 2Q`, and rounding up both sides is a Chvatal-Gomory cut (Chvatal, "Edmonds
polytopes and a hierarchy of combinatorial problems", Discrete Math. 4 (1973) 305-337). When 2Q does not divide n
the LP without it covers the arm with :math:`n / 2Q` of the cheap largest groups; on 20 units at Q = 3 the cut
raised the bound from 0.796 to 1.086 against the optimum 1.393, and the MIP proof ran 2.4 times faster with the row
in it.

Search. Before any MIP, the exact-cover search of :mod:`.cover` enumerates designs depth first, pruned by the same
Lagrangian bound, and a search that ends under its node cap proves its design optimal with a certified gap; no MIP
runs. A search stopped by the cap leaves the proof to the MIP loop below, which runs as it would without the search.
The first MIP takes the groups of least reduced cost (``FIRST_PER_UNIT`` per unit, at least ``FIRST_MIN``)
and the groups of a fixed design (:func:`seed_partition`), so it has a design. The best design's cost :math:`z` is the
optimum when every group left out has :math:`\\bar c > z - L`. Otherwise the next MIP takes the groups up to the
smaller of :math:`z - L` and the reduced cost of the ``GROWTH``-times-larger set, so a first design far above the
optimum does not admit almost every group at once: 24 geos at Q = 3 handed 189,950 of 190,026 groups to the second
MIP, 795 s, when the margin went straight to :math:`z - L`. The search ends when the margin reaches :math:`z - L` or
the MIP has every group.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from . import cover, mip

try:  # numba is the optional ``fast`` extra; without it the same code runs as Python
    from numba import njit
except ImportError:  # pragma: no cover - the suite runs with numba installed
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name: a cache written under one
#: import name fails to load under another (see tbr_helpers.design.exact.fastfloor).
CANONICAL_NAME = "mlsynth.utils.pangeo_helpers.columns"
_CACHE = __name__ == CANONICAL_NAME

#: The most groups scored for one arm: 20 million groups of six take about 1 GB.
MAX_COLUMNS = 20_000_000
#: The first MIP's groups: this many per unit, and at least FIRST_MIN.
FIRST_PER_UNIT = 10
FIRST_MIN = 200
#: A MIP that does not prove its design optimal is followed by one over at most GROWTH times as many groups.
GROWTH = 4
#: Column generation starts from the seed design and the cheapest SEED_PER_UNIT * n groups, and adds at most
#: BATCH_PER_UNIT * n groups a round.
SEED_PER_UNIT = 3
BATCH_PER_UNIT = 5
#: Reduced costs within this share of the largest cost count as zero.
_RTOL = 1e-9
_EPS = 1e-12
_U = 2.0 ** -53                      # the unit roundoff of IEEE double precision
#: The underflow unit of IEEE double precision, the smallest positive subnormal (Rump 2010, eq. 2.3).
_ETA = 2.0 ** -1074
_OBJECTIVES = {"ss_res": 0, "r2": 1, "weighted": 2, "supergeo_total": 3}


@dataclass(frozen=True)
class ColumnPool:
    """Every group of an arm with its best split.

    Attributes
    ----------
    unit_indices : np.ndarray
        The arm's units (rows of the panel).
    members : np.ndarray
        ``(N, min(2Q, n))`` positions into ``unit_indices``, ascending, padded with -1.
    cost : np.ndarray
        Length-N best-split costs.
    side_a : np.ndarray
        Length-N bit masks over a row's positions: bit p set when position p is in half A.
    complete : bool
        True when every group's cost is finite; a group with a non-finite cost is dropped, as the enumeration drops
        it, and the pool then no longer holds every group at :func:`column_index`.
    cost_lo, cost_up : np.ndarray
        Rigorous bounds on each group's exact cost (the cost in exact arithmetic on the panel's doubles): the least
        lower bound over its splits, and the upper bound of the split chosen.
    """

    unit_indices: np.ndarray
    members: np.ndarray
    cost: np.ndarray
    side_a: np.ndarray
    complete: bool
    cost_lo: np.ndarray
    cost_up: np.ndarray

    def __len__(self) -> int:
        return int(self.cost.size)

    def candidates(self, rows: Optional[Sequence[int]] = None) -> List[dict]:
        """The groups as :func:`.parallelism.enumerate_candidate_pairs` returns them, all of them or ``rows``."""
        rows = range(len(self)) if rows is None else rows
        out = []
        for j in rows:
            pos = self.members[j][self.members[j] >= 0]
            mask = int(self.side_a[j])
            in_a = np.array([(mask >> p) & 1 for p in range(pos.size)], dtype=bool)
            units = self.unit_indices[pos]
            out.append({"members": units, "score": float(self.cost[j]),
                        "side_a": units[in_a].astype(int), "side_b": units[~in_a].astype(int)})
        return out


@dataclass(frozen=True)
class RootLP:
    """The root LP over every group, solved by column generation.

    Attributes
    ----------
    pi : np.ndarray
        Duals of the cover rows.
    sigma : float
        Dual of the row ``sum(x) >= k``, at least 0.
    reduced_costs : np.ndarray
        Rigorous lower bounds on every group's exact reduced cost at those duals.
    bound : float
        The Lagrangian bound at those duals, a rigorous lower bound on every design's exact cost.
    lp_value : float
        The LP optimum over the groups generated.
    rounds : int
        LPs solved.
    pool_size : int
        Groups in the last LP.
    """

    pi: np.ndarray
    sigma: float
    reduced_costs: np.ndarray
    bound: float
    lp_value: float
    rounds: int
    pool_size: int


def count_columns(n: int, max_size: int) -> int:
    """The number of groups of 2 to min(2Q, n) units out of n."""
    return sum(comb(n, m) for m in range(2, min(2 * max_size, n) + 1))


def column_index(group: Sequence[int], n: int) -> int:
    """The row of a group (ascending positions) in a complete pool of n units: the groups of fewer units first, then
    the lexicographic rank among groups of its size (the combinatorial number system; Knuth, "The Art of Computer
    Programming", vol. 4A, section 7.2.1.3)."""
    m = len(group)
    index = sum(comb(n, k) for k in range(2, m))
    prev = -1
    for i, g in enumerate(group):
        for j in range(prev + 1, g):
            index += comb(n - 1 - j, m - 1 - i)
        prev = g
    return index


def feasible(n: int, max_size: int, min_pairs: int) -> bool:
    """Whether an exact cover with at least ``min_pairs`` groups exists: n >= 2, n even when Q = 1 (every group is a
    pair), and min_pairs <= n // 2, the most groups a design can have."""
    return n >= 2 and not (max_size == 1 and n % 2) and min_pairs <= n // 2


def seed_partition(n: int) -> List[Tuple[int, ...]]:
    """A design with n // 2 groups when one exists: consecutive pairs, the last three units a triple when n is odd."""
    pairs = [(i, i + 1) for i in range(0, n - 3 if n % 2 else n, 2)]
    return pairs + ([(n - 3, n - 2, n - 1)] if n % 2 else [])


def min_groups(n: int, max_size: int, min_pairs: int) -> int:
    """The fewest groups a design can have: ``min_pairs``, and ceil(n / 2Q) since no group exceeds 2Q units."""
    return max(int(min_pairs), -(-n // (2 * max_size)))


def _up(F: Fraction) -> float:
    """The least double at or above the rational F."""
    f = float(F)
    return f if Fraction(f) >= F else math.nextafter(f, math.inf)


def _down(F: Fraction) -> float:
    """The greatest double at or below the rational F."""
    f = float(F)
    return f if Fraction(f) <= F else math.nextafter(f, -math.inf)


def _gamma(k: int) -> float:
    """gamma_k = k u / (1 - k u), rounded up (Rump 2010, Theorem 2.1; Higham, Accuracy and Stability, Lemma 3.1)."""
    U = Fraction(_U)
    return _up(k * U / (1 - k * U))


def _k1(m: int) -> float:
    """The factor bounding a split's residual error by sum_p |c_p| ||A_p||: gamma_m for the dot products, gamma_{m+1}
    for the weights c = u / U, u / (1 - u) for the centred data, inflated by gamma_{2m+4} for evaluating the bound."""
    U = Fraction(_U)
    g = lambda k: k * U / (1 - k * U)
    return _up((g(m) + (g(m + 1) + U / (1 - U)) / (1 - g(m + 1))) * (1 + g(2 * m + 4)))


@njit(cache=_CACHE)
def _traj(pos, m, c, A, Nu, w, use_w, W, sw, k1, g_s, g_mu, r):
    """The level-removed residual sum of squares of the gap r = sum_p c_p A_p, computed from the centred data, and
    rigorous lower and upper bounds on its exact value. See the module docstring."""
    T = A.shape[1]
    for t in range(T):
        r[t] = 0.0
    abs_cn = 0.0
    for p in range(m):
        cp = c[p]
        row = A[pos[p]]
        for t in range(T):
            r[t] += cp * row[t]
        abs_cn += abs(cp) * Nu[pos[p]]
    S = 0.0
    if use_w:
        for t in range(T):
            S += w[t] * r[t]
        mu = S / W
    else:
        for t in range(T):
            S += r[t]
        mu = S / T
    s = 0.0
    dmax = 0.0
    for t in range(T):
        d = r[t] - mu
        if use_w:
            s += w[t] * d * d
            if abs(d) > dmax:
                dmax = abs(d)
        else:
            s += d * d
    # underflow (module docstring): abs_cn has m products, each gap entry m, s's terms two each
    abs_cn = abs_cn + m * _ETA
    e_r = (k1 * abs_cn + _ETA) * sw + (m * T * _ETA) * sw + 2.0 * _ETA
    rbar = abs_cn * (1.0 + 2.0 * k1) + _ETA
    t_mu = (g_mu * rbar + _ETA) * sw + _ETA
    e_mu = t_mu * t_mu + _ETA
    e_s = (T * _ETA) * (2.0 + dmax)                # w d is rounded before the second d multiplies it
    q_up = np.sqrt((s + e_s) / (1.0 - g_s) + _ETA)
    q2 = (s / (1.0 + g_s) - _ETA) - e_mu - e_s
    q_lo = np.sqrt(q2) if q2 > 0.0 else 0.0
    lo = q_lo - e_r
    lo = ((lo * lo - _ETA) * (1.0 - 8.0 * _U) - _ETA) if lo > 0.0 else 0.0
    up = ((q_up + e_r) * (q_up + e_r) + _ETA) * (1.0 + 8.0 * _U) + _ETA
    return s, max(lo, 0.0), up


@njit(cache=_CACHE)
def _covterm(pos, m, c, Cc, sc, om, k1, g_c):
    """The weighted squared standardized difference of the halves' covariate means, computed as a combination of
    the arm-centred covariates, and rigorous bounds on its exact value."""
    M = Cc.shape[1]
    s = 0.0
    lo = 0.0
    up = 0.0
    for j in range(M):
        d = 0.0
        a = 0.0
        for p in range(m):
            v = Cc[pos[p], j]
            d += c[p] * v
            a += abs(c[p]) * abs(v)
        e = k1 * (a + m * _ETA) + (m + 1) * _ETA        # underflow: d's m products, a's m, and k1 a
        v = d / sc[j]
        s += om[j] * v * v
        ad = abs(d)
        if ad > e:
            vl = (ad - e) / sc[j] - _ETA
            if vl > 0.0:
                lo += om[j] * (vl * vl - _ETA) - _ETA
        vu = (ad + e) / sc[j] + _ETA
        up += om[j] * (vu * vu + _ETA) + _ETA
    lo = lo * (1.0 - g_c) - _ETA
    return s, max(lo, 0.0), up * (1.0 + g_c) + _ETA


@njit(cache=_CACHE)
def _score_group(g, m, Q, obj, A, Nu, w, use_w, W, sw, Cc, sc, om, has_cov, u, tot, tot_err, k1t, g_s, g_mu,
                 g_c, gam, c, posA, cA, r, sel):
    best = np.inf
    best_lo = np.inf
    best_up = np.inf
    best_mask = 0
    for size_a in range(1, min(Q, m - 1) + 1):
        if m - size_a > Q:
            continue
        k = size_a - 1                       # half A is position 0 and k of positions 1..m-1, lexicographic
        for j in range(k):
            sel[j] = j + 1
        while True:
            mask = 1
            for j in range(k):
                mask |= 1 << sel[j]
            UA = 0.0
            UB = 0.0
            for p in range(m):
                if (mask >> p) & 1:
                    UA += u[g[p]]
                else:
                    UB += u[g[p]]
            for p in range(m):
                if (mask >> p) & 1:
                    c[p] = u[g[p]] / UA
                else:
                    c[p] = -u[g[p]] / UB
            if obj == 3:
                d = 0.0
                a = 0.0
                e = 0.0
                for p in range(m):
                    if (mask >> p) & 1:
                        d += tot[g[p]]
                    else:
                        d -= tot[g[p]]
                    a += abs(tot[g[p]])
                    e += tot_err[g[p]]
                e = (e + gam[m + 1] * a + _ETA) * (1.0 + 4.0 * _U) + _ETA
                s = d * d
                ad = abs(d)
                lo = max(((ad - e) * (ad - e) - _ETA) * (1.0 - 4.0 * _U) - _ETA, 0.0) if ad > e else 0.0
                up = ((ad + e) * (ad + e) + _ETA) * (1.0 + 4.0 * _U) + _ETA
            else:
                s, lo, up = _traj(g, m, c, A, Nu, w, use_w, W, sw, k1t[m], g_s, g_mu, r)
                if obj == 1:
                    na = 0
                    for p in range(m):
                        if (mask >> p) & 1:
                            posA[na] = g[p]
                            cA[na] = c[p]
                            na += 1
                    st, st_lo, st_up = _traj(posA, na, cA, A, Nu, w, False, W, 1.0, k1t[na], g_s, g_mu, r)
                    if st > 1e-12:
                        s = s / st                  # = 1 - R^2, as split_cost scores it
                    if st_lo > 1e-12:
                        lo = max((lo / st_up - _ETA) * (1.0 - 4.0 * _U) - _ETA, 0.0)
                        up = (up / st_lo + _ETA) * (1.0 + 4.0 * _U) + _ETA
                    elif st_up > 1e-12:            # the exact ss_tot may lie on either side of the threshold
                        lo = min(max((lo / st_up - _ETA) * (1.0 - 4.0 * _U) - _ETA, 0.0), lo)
                        up = max((up / 1e-12 + _ETA) * (1.0 + 4.0 * _U) + _ETA, up)
            if has_cov:
                cs, clo, cup = _covterm(g, m, c, Cc, sc, om, k1t[m], g_c)
                s += cs
                lo = max((lo + clo) * (1.0 - 2.0 * _U) - _ETA, 0.0)
                up = (up + cup) * (1.0 + 2.0 * _U) + _ETA
            if lo < best_lo:
                best_lo = lo
            if s < best:
                best = s
                best_up = up
                best_mask = mask
            j = k - 1
            while j >= 0 and sel[j] == m - k + j:
                j -= 1
            if j < 0:
                break
            sel[j] += 1
            for i in range(j + 1, k):
                sel[i] = sel[i - 1] + 1
    return best, best_lo, best_up, best_mask


@njit(cache=_CACHE)
def _score_all(n, Q, obj, A, Nu, w, use_w, W, sw, Cc, sc, om, has_cov, u, tot, tot_err, k1t, g_s, g_mu, g_c,
               gam, N):
    hi = min(2 * Q, n)
    width = max(hi, 1)
    members = np.full((N, width), -1, np.int32)
    cost = np.empty(N)
    cost_lo = np.empty(N)
    cost_up = np.empty(N)
    side_a = np.zeros(N, np.int32)
    g = np.empty(width, np.int64)
    c = np.empty(width)
    posA = np.empty(width, np.int64)
    cA = np.empty(width)
    r = np.empty(A.shape[1])
    sel = np.empty(width, np.int64)
    row = 0
    for m in range(2, hi + 1):
        for j in range(m):
            g[j] = j
        while True:
            s, lo, up, mk = _score_group(g, m, Q, obj, A, Nu, w, use_w, W, sw, Cc, sc, om, has_cov, u, tot,
                                         tot_err, k1t, g_s, g_mu, g_c, gam, c, posA, cA, r, sel)
            for j in range(m):
                members[row, j] = g[j]
            cost[row] = s
            cost_lo[row] = lo
            cost_up[row] = up
            side_a[row] = mk
            row += 1
            j = m - 1
            while j >= 0 and g[j] == n - m + j:
                j -= 1
            if j < 0:
                break
            g[j] += 1
            for i in range(j + 1, m):
                g[i] = g[i - 1] + 1
    return members, cost, cost_lo, cost_up, side_a


def _prepare(Y, objective, weights, cov, cov_scales, cov_weights, width):
    """The arm's centred paths and the constants of the error bounds (module docstring)."""
    n, T = Y.shape
    use_w = objective == "weighted"
    w = np.ones(T)
    W = float(T)
    if use_w:
        w = np.ascontiguousarray(weights, dtype=float)
        if w.shape != (T,) or not np.isfinite(w).all() or (w < 0).any() or float(w.sum()) <= _EPS:
            raise ValueError("objective='weighted' needs T finite, nonnegative weights summing to more than 1e-12.")
        W = float(w.sum())
        mu = (Y @ w) / W
    else:
        mu = Y.mean(axis=1)
    A = np.ascontiguousarray(Y - mu[:, None])        # the levels removed first: the two-pass algorithm
    Nu = np.sqrt(np.einsum("ij,ij->i", A, A) + T * _ETA) * (1.0 + _gamma(T + 3))   # T squares, each may underflow
    sw = math.sqrt(float(w.max())) * (1.0 + 4.0 * _U) if use_w and T else 1.0
    tot = Y.sum(axis=1)
    tot_err = (_gamma(max(T, 1)) * np.abs(Y).sum(axis=1) + _ETA) * (1.0 + _gamma(T + 1)) + _ETA
    if cov is None:
        Cc, sc, om, has_cov = np.zeros((n, 1)), np.ones(1), np.zeros(1), False
    else:
        C = np.asarray(cov, dtype=float)
        Cc = np.ascontiguousarray(C - C.mean(axis=0))   # a shift common to every unit leaves the difference exact
        sc = np.asarray(cov_scales, dtype=float)
        om = np.ones(C.shape[1]) if cov_weights is None else np.asarray(cov_weights, dtype=float)
        has_cov = True
    k1t = np.array([_k1(m) for m in range(width + 1)])
    gam = np.array([_gamma(k) for k in range(width + 3)])
    return dict(A=A, Nu=Nu, w=w, use_w=use_w, W=W, sw=sw, Cc=Cc, sc=sc, om=om, has_cov=has_cov, tot=tot,
                tot_err=tot_err, k1t=k1t, g_s=_gamma(T + 6), g_mu=_gamma(2 * T + 4), g_c=_gamma(Cc.shape[1] + 8),
                gam=gam)


def score_columns(
    unit_indices, Ypre: np.ndarray, max_size: int, objective: str = "ss_res",
    weights: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None,
    cov_scales: Optional[np.ndarray] = None, cov_weights: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
) -> ColumnPool:
    """Every group of 2 to 2Q of ``unit_indices`` with its best split, its cost, and rigorous lower and upper bounds
    on the exact cost, as :func:`.parallelism.enumerate_candidate_pairs` scores them. The arguments are that
    function's; ``Ypre``, ``cov`` and ``unit_weights`` are indexed by panel row."""
    if objective not in _OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; expected one of {sorted(_OBJECTIVES)}.")
    if objective == "weighted" and weights is None:
        raise ValueError("objective='weighted' requires weights.")
    idx = np.asarray(unit_indices, dtype=int)
    n = idx.size
    N = count_columns(n, max_size)
    if N > MAX_COLUMNS:
        raise MlsynthEstimationError(
            f"PANGEO: {n} units at max_supergeo_size={max_size} make {N:,} candidate groups, more than the "
            f"{MAX_COLUMNS:,} the exact design scores; use fast=True or a smaller max_supergeo_size.")
    Y = np.asarray(Ypre, dtype=float)[idx]
    width = max(min(2 * int(max_size), n), 1)
    k = _prepare(Y, objective, weights, None if cov is None else np.asarray(cov, float)[idx], cov_scales,
                 cov_weights, width)
    u = np.ones(n) if unit_weights is None else np.asarray(unit_weights, dtype=float)[idx]
    members, cost, cost_lo, cost_up, side_a = _score_all(
        n, int(max_size), _OBJECTIVES[objective], k["A"], k["Nu"], k["w"], k["use_w"], k["W"], k["sw"], k["Cc"],
        k["sc"], k["om"], k["has_cov"], u, k["tot"], k["tot_err"], k["k1t"], k["g_s"], k["g_mu"], k["g_c"],
        k["gam"], N)
    finite = np.isfinite(cost)
    if not finite.all():
        members, cost, side_a = members[finite], cost[finite], side_a[finite]
        cost_lo, cost_up = cost_lo[finite], cost_up[finite]
    return ColumnPool(idx, members, cost, side_a, bool(finite.all()), cost_lo, cost_up)


def _reduced_costs(members: np.ndarray, cost: np.ndarray, pi: np.ndarray, sigma: float) -> np.ndarray:
    pad = np.append(np.asarray(pi, dtype=float), 0.0)     # position -1 (padding) reads the 0
    rc = cost - sigma
    for j in range(members.shape[1]):
        rc = rc - pad[members[:, j]]
    return rc


def _rc_lower(pool: ColumnPool, pi: np.ndarray, sigma: float) -> np.ndarray:
    """Rigorous lower bounds on every group's exact reduced cost: cost_lo minus its duals and sigma, less the
    rounding of that sum of width + 1 terms (Rump 2010, Theorem 2.1)."""
    pad = np.append(np.asarray(pi, dtype=float), 0.0)     # position -1 (padding) reads the 0
    apad = np.abs(pad)
    rc = pool.cost_lo - sigma
    mag = pool.cost_lo + sigma
    for j in range(pool.members.shape[1]):
        rc = rc - pad[pool.members[:, j]]
        mag = mag + apad[pool.members[:, j]]
    return rc - (2.0 * _gamma(pool.members.shape[1] + 3) * (mag + np.abs(rc)) + _ETA)


def lagrangian_bound(pool: ColumnPool, pi: np.ndarray, sigma: float, k: int) -> Tuple[float, np.ndarray]:
    """The Lagrangian bound of the module docstring at duals ``pi`` and ``sigma`` (clipped at 0) for the row
    ``sum(x) >= k``, and every group's reduced cost, both as rigorous lower bounds: the reduced costs from the costs'
    lower bounds less their rounding, the duals summed by ``math.fsum`` (correctly rounded), the negative reduced
    costs summed with the error bound of Rump (2010), Theorem 2.1, and the total formed in rationals and rounded
    down. A lower bound on every design's exact cost, for any duals."""
    sigma = max(float(sigma), 0.0)
    rc = _rc_lower(pool, pi, sigma)
    neg = np.minimum(rc, 0.0)
    s2 = float(neg.sum())
    e2 = (_gamma(max(neg.size, 1)) * float(np.abs(neg).sum()) + _ETA) * (1.0 + _gamma(max(neg.size, 1))) + _ETA
    s1 = math.fsum(float(x) for x in pi)
    L = Fraction(s1) - abs(Fraction(s1)) * Fraction(_U) + Fraction(sigma) * k + Fraction(s2) - Fraction(e2)
    return _down(L), rc


def _master(pool: ColumnPool, cols: np.ndarray, cost: np.ndarray, n: int, k: int):
    rows = pool.members[cols]
    r, c = np.nonzero(rows >= 0)
    A = sp.csr_matrix((np.ones(r.size), (rows[r, c], r)), shape=(n, cols.size))
    res = linprog(cost[cols], A_eq=A, b_eq=np.ones(n), A_ub=-np.ones((1, cols.size)), b_ub=[-float(k)],
                  bounds=(0, 1), method="highs")
    if res.status != 0:  # pragma: no cover - the seed design is in every master, so it is feasible and bounded
        raise MlsynthEstimationError(f"PANGEO: the root LP did not solve ({res.message}).")
    return np.asarray(res.eqlin.marginals, dtype=float), -float(res.ineqlin.marginals[0]), float(res.fun)


def root_lp(pool: ColumnPool, k: int) -> RootLP:
    """The LP over every group with the row ``sum(x) >= k``, by column generation from the seed design and the
    cheapest groups, adding the groups of most negative reduced cost each round until none is negative. The pool must
    be complete and admit a design (:func:`feasible`). Costs are scaled by the power of two of
    :func:`.mip._cost_scale`, since HiGHS's tolerances are absolute."""
    n = pool.unit_indices.size
    scale = mip._cost_scale(pool.cost)
    cost = pool.cost * scale
    tol = _RTOL * max(1.0, float(np.abs(cost).max()))
    in_pool = np.zeros(len(pool), dtype=bool)
    in_pool[[column_index(g, n) for g in seed_partition(n)]] = True
    first = min(len(pool), SEED_PER_UNIT * n)
    in_pool[np.argpartition(cost, first - 1)[:first]] = True
    rounds = 0
    while True:
        cols = np.flatnonzero(in_pool)
        pi, sigma, value = _master(pool, cols, cost, n, k)
        rounds += 1
        rc = _reduced_costs(pool.members, cost, pi, max(sigma, 0.0))
        out = np.where(in_pool, np.inf, rc)
        entering = np.flatnonzero(out < -tol)
        if entering.size == 0:
            break
        take = min(entering.size, BATCH_PER_UNIT * n)
        in_pool[entering[np.argsort(out[entering], kind="stable")[:take]]] = True
    pi, sigma = pi / scale, sigma / scale
    bound, rc = lagrangian_bound(pool, pi, sigma, k)
    return RootLP(pi, max(sigma, 0.0), rc, bound, value / scale, rounds, int(cols.size))


def _search(pool: ColumnPool, root: RootLP, k: int, Q: int, node_limit: int) -> dict:
    """The start design, then the proof search over the groups its upper bound z cannot exclude (reduced cost at
    most z - L). Returns the status, the best design's rows, the lower bound when proved, the nodes, the groups
    searched and the seconds."""
    t = time.perf_counter()
    start = cover.start_design(pool, root, k, Q)
    if start.rows.size:
        z_up = sum((Fraction(float(pool.cost_up[j])) for j in start.rows), Fraction(0))
        best = _up(z_up)
        keep = np.flatnonzero(root.reduced_costs <= _up(z_up - Fraction(root.bound)))
    else:
        best, keep = np.inf, np.arange(len(pool))
    res = cover.search(pool, root, keep, k, Q, best, node_limit)
    rows = res.rows if res.rows.size else start.rows
    return dict(search=res.status, rows=rows, lower=res.lower, nodes=start.nodes + res.nodes, kept=keep.size,
                seconds=time.perf_counter() - t)


def solve_exact(
    unit_indices, Ypre: np.ndarray, max_size: int, *, objective: str = "ss_res",
    weights: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None,
    cov_scales: Optional[np.ndarray] = None, cov_weights: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None, min_pairs: int = 1, backend: str = "auto",
    mip_rel_gap: float = 0.0, mip_abs_gap: float = 0.0, solve: Optional[Callable] = None,
    search_node_limit: Optional[int] = None,
):
    """The exact-cover design of one arm: every group scored (:func:`score_columns`), the root bound
    (:func:`root_lp`), and the MIP (``solve``, :func:`.mip.solve_partition` by default) over the groups the bound
    cannot exclude, as in the module docstring. Returns ``(chosen, diagnostics)`` as ``solve_partition`` does, its
    diagnostics with ``n_candidate_pairs`` and ``columns_scored`` (every group), ``columns_solved`` (the groups in the
    last MIP), ``lagrangian_bound``, ``lp_value``, ``lp_rounds``, ``mip_solves``, ``score_seconds`` and
    ``lp_seconds``; ``solve_seconds`` is the MIPs' total. An arm with no design, or with a group of non-finite cost,
    goes to one MIP over every group, which reports it as it reports the full enumeration.

    Before any MIP, the exact-cover search (:mod:`.cover`) looks for the design and a proof that it is optimal,
    placing at most ``search_node_limit`` groups (``cover.SEARCH_NODE_LIMIT`` when None; 0 skips it). It runs only
    with ``backend="auto"``: a backend named explicitly is a request for that MIP, and gets it. A proof returns
    the design with no MIP: ``backend`` is ``"search"``, ``mip_solves`` 0, ``dual_bound`` a rigorous lower bound on
    the optimum, ``objective_upper`` one above the design's exact cost and ``mip_gap`` the gap between them relative
    to the design's cost, as the MIP path reports it (absolute when that cost is 0).
    A search stopped by the cap leaves the MIP loop to run as without it. ``search`` reports which happened
    (``"proved"``, ``"node_limit"``, ``"off"``, or ``"unavailable"`` without numba) and ``search_nodes`` the groups
    the search placed."""
    if search_node_limit is None:
        search_node_limit = cover.SEARCH_NODE_LIMIT
    if isinstance(search_node_limit, bool) or not isinstance(search_node_limit, (int, np.integer)) \
            or search_node_limit < 0:
        raise MlsynthConfigError(
            f"search_node_limit must be a non-negative integer (0 skips the search); got {search_node_limit!r}.")
    solve = mip.solve_partition if solve is None else solve
    kw = dict(return_diagnostics=True, backend=backend, mip_rel_gap=mip_rel_gap, mip_abs_gap=mip_abs_gap)
    t0 = time.perf_counter()
    pool = score_columns(unit_indices, Ypre, max_size, objective=objective, weights=weights, cov=cov,
                         cov_scales=cov_scales, cov_weights=cov_weights, unit_weights=unit_weights)
    idx = pool.unit_indices
    n = idx.size
    extra = dict(n_candidate_pairs=len(pool), columns_scored=len(pool), score_seconds=time.perf_counter() - t0)
    if not (pool.complete and feasible(n, max_size, min_pairs)):
        chosen, d = solve(pool.candidates(), idx, min_pairs=min_pairs, **kw)
        # Not covered: a non-finite group holds a unit with a NaN path, which no design covers, so the
        # MIP above raises; a panel whose groups overflow only in part would reach here.
        d.update(extra, columns_solved=len(pool), lagrangian_bound=None,  # pragma: no cover
                 lp_value=None, lp_rounds=0, mip_solves=1, lp_seconds=0.0)
        return chosen, d  # pragma: no cover
    k = min_groups(n, max_size, min_pairs)
    t1 = time.perf_counter()
    root = root_lp(pool, k)
    lp_seconds = time.perf_counter() - t1
    L, rc = root.bound, root.reduced_costs               # rigorous lower bounds
    seed = np.array([column_index(g, n) for g in seed_partition(n)])
    pos = {int(u): p for p, u in enumerate(idx)}
    info = dict(search="off", search_nodes=0)
    if backend != "auto":
        pass                                          # a MIP backend asked for by name
    elif search_node_limit > 0 and not cover._NUMBA:
        info["search"] = "unavailable"   # pragma: no cover - the suite runs with numba installed
    elif search_node_limit > 0:
        found = _search(pool, root, k, max_size, int(search_node_limit))
        info.update(search=found["search"], search_nodes=found["nodes"])
        if found["search"] == "proved":
            rows = found["rows"]
            chosen = pool.candidates(rows)
            z_up = sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))
            upper = _up(z_up)
            lower = min(found["lower"], upper)
            z = math.fsum(float(pool.cost[j]) for j in rows)
            d = dict(path="exact_search", backend="search", solver_name="exact-cover search", status="optimal",
                     presolve_retry=False, mip_rel_gap_requested=float(mip_rel_gap),
                     mip_abs_gap_requested=float(mip_abs_gap), gap_enforced=True,
                     mip_gap=(upper - lower) / abs(z) if z != 0 else upper - lower, dual_bound=float(lower),
                     node_count=float(found["nodes"]), solve_seconds=found["seconds"])
            d.update({**extra, **info}, columns_solved=int(found["kept"]), lagrangian_bound=float(L),
                     lp_value=root.lp_value, lp_rounds=root.rounds, mip_solves=0, lp_seconds=lp_seconds,
                     objective_value=z, objective_upper=upper,
                     n_selected_pairs=len(chosen))
            return chosen, d

    def upper(chosen) -> Fraction:
        """A rigorous upper bound on a design's exact cost: the sum of its groups' cost_up, in rationals."""
        rows = [column_index(tuple(sorted(pos[int(u)] for u in c["members"])), n) for c in chosen]
        return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))

    count = min(len(pool), max(FIRST_MIN, FIRST_PER_UNIT * n))
    delta = float(np.partition(rc, count - 1)[count - 1])
    best = last = None
    solves, mip_seconds = 0, 0.0
    while True:
        keep = np.union1d(np.flatnonzero(rc <= delta), seed)
        chosen, d = solve(pool.candidates(keep), idx, min_pairs=k, **kw)
        solves += 1
        mip_seconds += float(d["solve_seconds"])
        last = d
        if best is None or d["objective_value"] < best[1]["objective_value"]:
            best = (chosen, d, upper(chosen))
        margin = best[2] - Fraction(L)                 # exact: every group left out has rc_lo > delta
        if margin <= Fraction(delta) or keep.size >= len(pool):
            break
        # at most GROWTH times the groups, and never past z - L: a first design far above the optimum would
        # otherwise admit almost every group at once
        count = min(len(pool), count * GROWTH)
        delta = min(_up(margin), float(np.partition(rc, count - 1)[count - 1]))
    chosen, d, z_up = best
    z = d["objective_value"]
    d = dict(d)
    if d is not last and last.get("dual_bound") is not None:
        # the last MIP held every design cheaper than this one, so its bound bounds the optimum
        d["dual_bound"] = last["dual_bound"]
        d["mip_gap"] = (abs(z - last["dual_bound"]) / abs(z)) if z != 0 else last.get("mip_gap")
    d.update({**extra, **info}, columns_solved=int(keep.size), lagrangian_bound=float(L), lp_value=root.lp_value,
             lp_rounds=root.rounds, mip_solves=solves, solve_seconds=mip_seconds, lp_seconds=lp_seconds,
             objective_value=float(z), objective_upper=_up(z_up), n_selected_pairs=len(chosen))
    return chosen, d
