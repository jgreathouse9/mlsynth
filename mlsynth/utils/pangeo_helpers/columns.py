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
:math:`a \\circ b = \\mathrm{fl}(a \\circ b)(1 + \\varepsilon) + \\delta`, :math:`\\varepsilon\\delta = 0`). Three
squares that form a cost bound carry that error with no relative term to cover it: the residual sum of squares
(its :math:`T` squares, :math:`2T\\eta` in all, the rest covering the two quotients; the final square and scaling add
:math:`\\eta` of their own; the weighted form rounds :math:`w_t d_t` before the second :math:`d_t` multiplies it, an
error of at most :math:`|d_t|\\eta / 2` that the gap's error term :math:`e_r \\ge k_1 |d_t| \\sqrt{w_{\\max}} / 2`
covers whenever it is not below :math:`\\eta`), the square of the
difference of totals, and each covariate's :math:`\\omega_j v_j^2`, whose error the weight multiplies. Each adds
:math:`\\eta` to an upper bound or subtracts it from a lower one. That is exact when the result is subnormal, since
addition then does not round (eq. 2.4), and rounds away when it is normal, where :math:`\\delta = 0`, so at normal
scales the bounds are the same doubles as without it. The other operations need no term. Sums are exact below the
normal range (eq. 2.4), so the error a :math:`\\gamma_k` term bounds is zero when its argument is subnormal; when the
argument :math:`X` is normal, each :math:`\\gamma_k` is taken at least one step past the one the error needs, a
margin of at least :math:`uX \\ge \\eta / 2`. Two
places keep a term all the same: the Lagrangian bound's summation error is two products whose margin falls below
:math:`\\eta` when the sum is under :math:`2\\lambda` (:math:`\\lambda = 2^{-1022}`, the smallest normal number), and
the search's pruning test subtracts :math:`r\\sigma`, a product that is a term of the bound itself, not of its error
(:mod:`.cover`). Measured on panels scaled by :math:`2^s` from :math:`s = -505` to
:math:`-570`, all four objectives, the bounds of 6,343 groups failed to bracket the exact cost without the three
terms and of none with them (jgreathouse9/mlsynth#1170). A group's
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
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import comb
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..solvers.dual_simplex import solve_boxed_lp
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
#: The proof search after the first MIP separates clique cuts over its groups for at most CLIQUE_ROUNDS LPs
#: (0 turns them off), adding the CLIQUE_PER_ROUND most violated a round.
CLIQUE_ROUNDS = 10
CLIQUE_PER_ROUND = 60
#: A clique counts as violated when the LP point exceeds its right-hand side 1 by more than this.
_VIOLATION = 1e-6
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
    for t in range(T):
        d = r[t] - mu
        if use_w:
            s += w[t] * d * d
        else:
            s += d * d
    e_r = k1 * abs_cn * sw
    rbar = abs_cn * (1.0 + 2.0 * k1)
    e_mu = g_mu * g_mu * rbar * rbar * sw * sw
    # the T squares of s may underflow, each by eta / 2; the rest of e_s covers the quotients below (module docstring)
    e_s = (T * _ETA) * 2.0
    q_up = np.sqrt((s + e_s) / (1.0 - g_s))
    q2 = s / (1.0 + g_s) - e_mu - e_s
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
        e = k1 * a
        v = d / sc[j]
        s += om[j] * v * v
        ad = abs(d)
        if ad > e:
            vl = (ad - e) / sc[j] - _ETA
            if vl > 0.0:
                lo += om[j] * (vl * vl - _ETA) - _ETA
        vu = (ad + e) / sc[j] + _ETA
        up += om[j] * (vu * vu + _ETA) + _ETA
    return s, lo * (1.0 - g_c), up * (1.0 + g_c)


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
                e = (e + gam[m + 1] * a) * (1.0 + 4.0 * _U)
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
                        lo = lo / st_up * (1.0 - 4.0 * _U)
                        up = up / st_lo * (1.0 + 4.0 * _U)
                    elif st_up > 1e-12:            # the exact ss_tot may lie on either side of the threshold
                        lo = min(lo / st_up * (1.0 - 4.0 * _U), lo)
                        up = max(up / 1e-12 * (1.0 + 4.0 * _U), up)
            if has_cov:
                cs, clo, cup = _covterm(g, m, c, Cc, sc, om, k1t[m], g_c)
                s += cs
                lo = (lo + clo) * (1.0 - 2.0 * _U)
                up = (up + cup) * (1.0 + 2.0 * _U)
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
    Nu = np.sqrt(np.einsum("ij,ij->i", A, A)) * (1.0 + _gamma(T + 3))
    sw = math.sqrt(float(w.max())) * (1.0 + 4.0 * _U) if use_w and T else 1.0
    tot = Y.sum(axis=1)
    tot_err = _gamma(max(T, 1)) * np.abs(Y).sum(axis=1) * (1.0 + _gamma(T + 1))
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


@dataclass(frozen=True, init=False)
class _Rows:
    """Values on some rows of the pool, read as an array over the whole pool that is -inf off those rows: indexing
    with an array of rows or a mask returns what that array would, and ``numpy.asarray`` builds it. A branch's
    search reads only its own groups, and an array over the 4.6 million groups of a 40-geo Q = 3 arm cost 25 ms
    per branch to fill."""

    rows: np.ndarray
    values: np.ndarray
    size: int

    def __init__(self, rows, values, size):
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        object.__setattr__(self, "rows", rows[order])
        object.__setattr__(self, "values", np.asarray(values, dtype=float)[order])
        object.__setattr__(self, "size", int(size))

    @property
    def shape(self):
        return (self.size,)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64)
        out = np.full(idx.shape, -np.inf)
        if self.rows.size:
            pos = np.minimum(np.searchsorted(self.rows, idx), self.rows.size - 1)
            hit = self.rows[pos] == idx
            out[hit] = self.values[pos[hit]]
        return out

    def __array__(self, dtype=None, copy=None):
        full = np.full(self.size, -np.inf)
        full[self.rows] = self.values
        return full if dtype is None else full.astype(dtype)


def _rc_lower(pool: ColumnPool, pi: np.ndarray, sigma: float, cost_lo: Optional[np.ndarray] = None,
              rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Rigorous lower bounds on every group's exact reduced cost: cost_lo (the pool's, or lower bounds on shifted
    costs) minus its duals and sigma, less the rounding of that sum of width + 1 terms (Rump 2010, Theorem 2.1).
    With ``rows``, only those groups are computed, to the same doubles, and returned as a :class:`_Rows`, -inf on
    every other row, a lower bound on any reduced cost: a branch's search reads only the groups its LP keeps."""
    pad = np.append(np.asarray(pi, dtype=float), 0.0)     # position -1 (padding) reads the 0
    apad = np.abs(pad)
    lo = pool.cost_lo if cost_lo is None else cost_lo
    members = pool.members
    if rows is not None:
        lo, members = lo[rows], members[rows]
    rc = lo - sigma
    mag = np.abs(lo) + sigma
    for j in range(members.shape[1]):
        rc = rc - pad[members[:, j]]
        mag = mag + apad[members[:, j]]
    rc = rc - 2.0 * _gamma(pool.members.shape[1] + 3) * (mag + np.abs(rc))
    if rows is None:
        return rc
    return _Rows(rows, rc, len(pool))


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


def _search(pool: ColumnPool, root: RootLP, k: int, Q: int, node_limit: int, progress=None) -> dict:
    """The start design, then the proof search over the groups its upper bound z cannot exclude (reduced cost at
    most z - L). Returns the status, the best design's rows, the lower bound when proved, the nodes, the groups
    searched and the seconds. Both searches report to ``progress`` (:func:`.cover.new_progress`)."""
    t = time.perf_counter()
    start = cover.start_design(pool, root, k, Q, progress=progress)
    if start.rows.size:
        z_up = sum((Fraction(float(pool.cost_up[j])) for j in start.rows), Fraction(0))
        best = _up(z_up)
        keep = np.flatnonzero(root.reduced_costs <= _up(z_up - Fraction(root.bound)))
    else:
        best, keep = np.inf, np.arange(len(pool))
    res = cover.search(pool, root, keep, k, Q, best, node_limit, progress=progress)
    rows = res.rows if res.rows.size else start.rows
    return dict(search=res.status, rows=rows, lower=res.lower, nodes=start.nodes + res.nodes, kept=keep.size,
                seconds=time.perf_counter() - t)


def _incidence(members: np.ndarray, n: int) -> np.ndarray:
    """The groups-by-units incidence of ``members`` (rows padded with -1), as booleans."""
    inc = np.zeros((members.shape[0], n), dtype=bool)
    for t in range(members.shape[1]):
        ok = members[:, t] >= 0
        inc[np.flatnonzero(ok), members[ok, t]] = True
    return inc


def _clique_rows(x: np.ndarray, members: np.ndarray, cap: int = CLIQUE_PER_ROUND) -> List[Tuple[int, ...]]:
    """Cliques of the conflict graph on the support of the LP point ``x`` that it violates: groups pairwise sharing a
    unit, at least three of them, no unit common to all (a cover row already holds those), with ``sum x > 1``.

    Finding a violated clique is a maximum-weight clique problem, strongly NP-hard, and a graph on m vertices can
    have 3^(m/3) maximal cliques (Moon and Moser, Israel J. Math. 3 (1965) 23-28, Theorem 1), so the support's cliques are not
    listed (Atamturk, Nemhauser and Savelsbergh, "Conflict graphs in solving integer programming problems", EJOR 121
    (2000) 40-55, p. 50). The separation is the root-vertex scheme of Hoffman and Padberg as Marzi, Rossi and Smriglio
    state it ("Computational study of separation algorithms for clique inequalities", Soft Comput. 23 (2019)
    3013-3027, section 2, Algorithms 1 and 2): the roots in order of nondecreasing degree; a root whose weight plus
    its remaining neighbours' weights cannot pass 1 is skipped (Algorithm 1, step 5); otherwise its clique is grown
    greedily, the heaviest remaining neighbour first (Algorithm 2 with f = x, ties to the lower index); then the root
    leaves the graph. Each root costs at most its degree times the clique's size, so a round is polynomial in the
    support; the cliques it returns are valid cuts whether or not they are the most violated. Returns at most ``cap``
    of them, distinct, most violated first, as row indices of ``members``."""
    supp = np.flatnonzero(x > _EPS)
    sets = [frozenset(int(u) for u in members[j][members[j] >= 0]) for j in supp]
    w = [float(x[j]) for j in supp]
    m = supp.size
    adj = [set() for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            if sets[a] & sets[b]:
                adj[a].add(b)
                adj[b].add(a)
    alive = set(range(m))
    out = []
    for i in sorted(range(m), key=lambda a: (len(adj[a]), a)):
        near = adj[i] & alive
        if w[i] + math.fsum(w[j] for j in near) - 1.0 > _VIOLATION:
            R, cand = [i], set(near)
            while cand:
                j = min(cand, key=lambda v: (-w[v], v))
                R.append(j)
                cand &= adj[j]
            v = math.fsum(w[a] for a in R) - 1.0
            key = tuple(sorted(int(supp[a]) for a in R))
            if len(R) >= 3 and v > _VIOLATION and not frozenset.intersection(*(sets[a] for a in R)):
                out.append((-v, key))        # distinct: every clique holds its root, which then leaves the graph
        alive.discard(i)
    out.sort()
    return [K for _, K in out[:cap]]


def _lift(K: Sequence[int], members: np.ndarray, order: np.ndarray, inc: Optional[np.ndarray] = None) -> np.ndarray:
    """The clique ``K`` (row indices of ``members``) grown greedily over ``order``: a group joins only when it meets
    every member so far, those added before it included. A group that meets only the members of ``K`` can miss
    another added group, and two such groups can be in one design. Returns the members, ``K`` first."""
    if inc is None:
        inc = _incidence(members, int(members.max()) + 1)
    out = [int(j) for j in K]
    ok = np.ones(inc.shape[0], dtype=bool)
    for j in out:
        ok &= inc[:, inc[j]].any(axis=1)
    ok[out] = False
    for j in order:
        j = int(j)
        if ok[j]:
            out.append(j)
            ok &= inc[:, inc[j]].any(axis=1)
            ok[j] = False
    return np.asarray(out, dtype=np.int64)


def _lp(pool: ColumnPool, cols: np.ndarray, units: np.ndarray, kk: int, cliques):
    """The LP over the groups ``cols`` covering ``units`` once each, with ``sum(x) >= kk`` when kk > 0 and a row
    ``sum x <= 1`` for each clique's members among ``cols`` (cliques with fewer than two are left out). Costs are
    scaled by the power of two of :func:`.mip._cost_scale`, as :func:`root_lp` does. Returns the result (None when
    it has no solution), the cliques kept as pool rows, and the scale."""
    n = int(pool.unit_indices.size)
    rows = pool.members[cols]
    r, c = np.nonzero(rows >= 0)
    relabel = -np.ones(n, dtype=np.int64)
    relabel[units] = np.arange(units.size)
    A = sp.csr_matrix((np.ones(r.size), (relabel[rows[r, c]], r)), shape=(units.size, cols.size))
    ub = [-np.ones(cols.size)] if kk > 0 else []
    rhs = [-float(kk)] if kk > 0 else []
    kept = []
    for K in cliques:
        row = np.isin(cols, K)
        if row.sum() >= 2:
            ub.append(row.astype(float))
            rhs.append(1.0)
            kept.append(cols[row])
    scale = mip._cost_scale(pool.cost[cols])
    kw = dict(A_ub=sp.csr_matrix(np.array(ub)), b_ub=rhs) if ub else {}
    res = linprog(pool.cost[cols] * scale, A_eq=A, b_eq=np.ones(units.size), bounds=(0, 1), method="highs", **kw)
    return (res if res.status == 0 else None), kept, scale


def _shifted(pool: ColumnPool, cliques, mu: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Rigorous lower bounds on every group's shifted cost, cost_lo plus the duals of the cliques it is in: the
    float sum of at most m + 1 terms (m the most cliques a group is in) less twice its rounding (Higham 2002,
    Lemma 3.1).

    With ``rows``, only those groups are computed, to the same doubles (m still counts every group's cliques), and
    returned as a :class:`_Rows`, -inf on every other row, a lower bound on any shifted cost. A branch's search
    reads only its groups and its fixed group, and the whole pool cost each branch 89 ms at the 4.6 million groups
    of a 40-geo Q = 3 arm."""
    pairs = list(zip(cliques, mu))
    shift = np.zeros(len(pool))
    for K, m in pairs:
        shift[K] += float(m)
    members = np.concatenate([np.asarray(K, dtype=np.int64) for K, _ in pairs]) if pairs else np.zeros(0, np.int64)
    most = int(np.unique(members, return_counts=True)[1].max()) if members.size else 0
    gam = 2.0 * _gamma(most + 2)
    if rows is None:
        return (pool.cost_lo + shift) - gam * (np.abs(pool.cost_lo) + shift)
    rows = np.asarray(rows, dtype=np.int64)
    c, d = pool.cost_lo[rows], shift[rows]
    return _Rows(rows, (c + d) - gam * (np.abs(c) + d), len(pool))


@dataclass(frozen=True)
class _Duals:
    """A residual LP's duals, in the form cover.search reads: ``pi`` over every unit (0 on covered ones), ``sigma``,
    rigorous lower bounds on every pool row's reduced cost, and the LP's cliques (pool rows) with their duals
    ``mu >= 0`` and rigorous lower bounds on every row's shifted cost (None without cliques). A branch's bounds are
    :class:`_Rows` over its own groups, -inf elsewhere."""

    pi: np.ndarray
    sigma: float
    reduced_costs: np.ndarray
    cliques: tuple = ()
    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cost_lo: Optional[np.ndarray] = None


def _residual_lp(pool: ColumnPool, cols: np.ndarray, units: np.ndarray, kk: int, cliques=()):
    """The LP of :func:`_lp` and its Lagrangian bound. With cliques ``K`` and their duals ``mu >= 0``, every design
    ``x`` over ``cols`` costs

    .. math::

       c^\\top x = \\sum_G \\tilde c_G x_G - \\sum_K \\mu_K + \\sum_K \\mu_K s_K, \\qquad
       \\tilde c_G = c_G + \\sum_{K \\ni G} \\mu_K,

    with slack :math:`s_K = 1 - \\sum_{G \\in K} x_G \\ge 0`, so the bound of the module docstring on the shifted
    costs, less :math:`\\sum \\mu`, bounds the cost (Geoffrion 1974; the cuts are those of Marzi, Rossi and
    Smriglio 2019). It is evaluated as :func:`lagrangian_bound` does, the shifted costs and reduced costs from the
    costs' lower bounds less their rounding, and the duals' sum exact. Returns ``(duals, L, value)``: the
    :class:`_Duals`, the bound as a rational, and the LP's value; None when the LP has no solution."""
    cols = np.asarray(cols, dtype=np.int64)
    units = np.asarray(units, dtype=np.int64)
    if cols.size == 0:
        return None
    res, kept, scale = _lp(pool, cols, units, kk, cliques)
    if res is None:
        return None
    n = int(pool.unit_indices.size)
    pi = np.zeros(n)
    pi[units] = np.asarray(res.eqlin.marginals, dtype=float) / scale
    marg = -np.asarray(res.ineqlin.marginals, dtype=float) / scale if (kk > 0 or kept) else np.zeros(0)
    sigma = max(float(marg[0]), 0.0) if kk > 0 else 0.0
    mu = np.maximum(marg[1 if kk > 0 else 0:], 0.0)
    duals, L = _bound_from(pool, cols, units, kk, kept, pi, sigma, mu)
    return duals, L, float(res.fun) / scale


def _bound_from(pool: ColumnPool, cols: np.ndarray, units: np.ndarray, kk: int, kept, pi: np.ndarray,
                sigma: float, mu: np.ndarray, restrict: bool = False, shift_rows: Optional[np.ndarray] = None):
    """The :class:`_Duals` and rigorous Lagrangian bound of :func:`_residual_lp` for multipliers ``pi`` (on
    ``units``), ``sigma >= 0`` (on ``sum(x) >= kk``) and ``mu >= 0`` (one per clique in ``kept``) over the groups
    ``cols``. Any multipliers of those signs give a valid bound (Geoffrion 1974); optimal ones give the LP's value.
    With ``restrict`` the reduced costs are computed over ``cols`` only (:func:`_rc_lower`), and with ``shift_rows``
    the shifted costs over those rows only (:func:`_shifted`), which must hold ``cols``. Returns ``(duals, L)``,
    L a rational."""
    lo = _shifted(pool, kept, mu, rows=shift_rows) if kept else None
    rc = _rc_lower(pool, pi, sigma, lo, rows=cols if restrict else None)
    neg = np.minimum(rc[cols], 0.0)
    s2 = float(neg.sum())
    g = _gamma(max(neg.size, 1))
    e2 = (g * float(np.abs(neg).sum()) + _ETA) * (1.0 + g) + _ETA     # as lagrangian_bound: two products
    s1 = math.fsum(float(x) for x in pi[units])
    L = Fraction(s1) - abs(Fraction(s1)) * Fraction(_U) + Fraction(sigma) * kk + Fraction(s2) - Fraction(e2) \
        - sum((Fraction(float(m)) for m in mu), Fraction(0))
    return _Duals(pi, sigma, rc, tuple(kept), mu if len(kept) else np.zeros(0), lo), L


def _separate_cliques(pool: ColumnPool, keep: np.ndarray, k: int, rounds: Optional[int] = None) -> tuple:
    """Clique cuts for the LP over the groups ``keep`` (:func:`_lp` with ``sum(x) >= k``): each round solves the LP
    with the cuts so far, takes the violated cliques of its support (:func:`_clique_rows`) and lifts each over the
    kept groups in order of LP value, then cost (:func:`_lift`). At most ``rounds`` LPs (``CLIQUE_ROUNDS`` when
    None); it stops when a round adds no new clique. Returns the cliques as arrays of pool rows; empty when the
    groups cannot cover the arm."""
    rounds = CLIQUE_ROUNDS if rounds is None else int(rounds)
    keep = np.asarray(keep, dtype=np.int64)
    n = int(pool.unit_indices.size)
    if rounds <= 0 or keep.size == 0:
        return ()
    mem = pool.members[keep]
    inc = _incidence(mem, n)
    if not inc.any(axis=0).all():
        return ()
    cliques, seen = [], set()
    for _ in range(rounds):
        res, _, _ = _lp(pool, keep, np.arange(n), k, [keep[K] for K in cliques])
        if res is None:
            break
        order = np.lexsort((pool.cost_lo[keep], -res.x))
        new = []
        for K in _clique_rows(res.x, mem):
            lifted = np.sort(_lift(K, mem, order, inc))
            key = tuple(lifted.tolist())
            if key not in seen:
                seen.add(key)
                new.append(lifted)
        if not new:
            break
        cliques += new
    return tuple(keep[K] for K in cliques)


def _parent_lp(pool: ColumnPool, keep: np.ndarray, k: int, cliques):
    """The LP over the groups ``keep`` in the boxed form of :func:`solve_boxed_lp`: a row per unit (= 1), the
    cardinality row ``k <= sum(x) <= max(|keep|, k)`` (its upper bound never binds, since every x is at most 1), and a
    row ``0 <= sum x <= 1`` for each clique with at least two members among ``keep``. Returns ``(A, rl, ru,
    cliques)``, the cliques those rows hold, as pool rows."""
    n = int(pool.unit_indices.size)
    mem = pool.members[keep]
    r, c = np.nonzero(mem >= 0)
    blocks = [sp.csr_matrix((np.ones(r.size), (mem[r, c], r)), shape=(n, keep.size)),
              sp.csr_matrix(np.ones((1, keep.size)))]
    rl, ru, kept = [np.ones(n), [float(k)]], [np.ones(n), [float(max(keep.size, k))]], []
    for K in cliques:
        row = np.isin(keep, K)
        if row.sum() >= 2:
            blocks.append(sp.csr_matrix(row[None, :].astype(float)))
            rl.append([0.0])
            ru.append([1.0])
            kept.append(keep[row])
    return sp.vstack(blocks, format="csc"), np.concatenate(rl), np.concatenate(ru), kept


def _branches(pool: ColumnPool, root: RootLP, keep: np.ndarray, k: int, cliques=(), best_up: float = np.inf):
    """Every branch of :func:`_iter_branches`, as a list."""
    return list(_iter_branches(pool, root, keep, k, cliques, best_up))


def _iter_branches(pool: ColumnPool, root: RootLP, keep: np.ndarray, k: int, cliques=(),
                   best_up: float = np.inf):
    """The first level of the search, each branch with its residual LP. The search branches on the unit with the
    fewest kept groups (ties to the first unit); a branch fixes one of them, and the cover problem left over (the
    other units, the kept groups disjoint from it, at least k - 1 groups) is an LP relaxation of its own, with the
    ``cliques`` restricted to the groups left (a clique holding the fixed group has none left).

    That LP is the parent LP (:func:`_parent_lp`) with bounds changed: the fixed group at 1 and every group meeting
    it at 0. A child changes bounds only, so its basis matrix is the parent's and the parent's optimal basis stays
    dual feasible (Koberstein, "The Dual Simplex Method", PhD thesis, Paderborn, 2005, chapter 1, p. 2): the parent is
    solved once by :func:`solve_boxed_lp`, and each child starts from its
    optimal basis and inverse, with no factorization. A child stops as soon as its bound reaches ``best_up``
    (Pan, Linear Programming Computation, 2nd ed., Springer, 2023, Lemma 8.3.4, p. 216: the dual objective does not
    decrease), so a branch the incumbent prunes costs no more LP
    work than that. The duals at the point it stops, restricted to the units and cliques left, give the Lagrangian
    bound of :func:`_residual_lp` (:func:`_bound_from`): valid at any iterate, the residual LP's value at the
    child's optimum. Costs are scaled by the power of two of :func:`.mip._cost_scale`, as :func:`root_lp` does.

    Yields ``(row, bound, rows_left, duals)`` per branch, the bound covering the fixed group's cost lower bound, in
    the order of that cost; a branch with no feasible residual has bound infinity and no duals. Each branch's LP
    is solved when the branch is drawn, so a caller that stops early solves no more, and holds only the branches
    it keeps: a branch's duals carry two arrays over the whole pool, 74 MB at the 4.6 million groups of a 40-geo
    Q = 3 arm, and that arm's 1,519 branches held at once passed 14 GB."""
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0:
        return
    n = int(pool.unit_indices.size)
    inc = _incidence(pool.members[keep], n)
    g0 = int(np.argmin(inc.sum(axis=0)))
    order = np.flatnonzero(inc[:, g0])[np.argsort(pool.cost_lo[keep[inc[:, g0]]], kind="stable")]
    if order.size == 0:
        return
    A, rl, ru, kept = _parent_lp(pool, keep, k, cliques)
    scale = mip._cost_scale(pool.cost[keep])
    c = pool.cost[keep] * scale
    parent = solve_boxed_lp(c, A, rl, ru, np.zeros(keep.size), np.ones(keep.size))
    cutoff = float(best_up) * scale if np.isfinite(best_up) else None
    kk = max(int(k) - 1, 0)
    in_left = np.zeros(len(pool), dtype=bool)          # a child's groups, as a mask over the pool
    for a in order:
        j = int(keep[a])
        G = inc[a]
        left_mask = ~inc[:, G].any(axis=1)
        left = keep[left_mask]
        units = np.flatnonzero(~G)
        lo_j = Fraction(float(pool.cost_lo[j]))
        if units.size == 0:
            yield (j, _down(lo_j) if kk == 0 else np.inf, left, None)
            continue
        if left.size == 0 or parent.status == "infeasible":
            yield (j, np.inf, left, None)
            continue
        lo = np.zeros(keep.size)
        up = left_mask.astype(float)
        lo[a] = up[a] = 1.0
        res = solve_boxed_lp(c, A, rl, ru, lo, up, basis=parent.basis, cutoff=cutoff)
        if res.status == "infeasible":
            yield (j, np.inf, left, None)
            continue
        y = res.y / scale
        pi = np.zeros(n)
        pi[units] = y[units]
        sigma = max(float(y[n]), 0.0) if kk > 0 else 0.0
        use, mu = [], []
        in_left[left] = True
        for K, m in zip(kept, np.maximum(-y[n + 1:], 0.0)):
            K_left = K[in_left[K]]
            if K_left.size >= 2:
                use.append(K_left)
                mu.append(float(m))
        in_left[left] = False
        duals, L = _bound_from(pool, left, units, kk, use, pi, sigma, np.asarray(mu, dtype=float), restrict=True,
                               shift_rows=np.append(left, j))     # the search reads the fixed group's too
        yield (j, _down(L + lo_j), left, duals)


def _reopt_budget(pool: ColumnPool, keep: np.ndarray, cap: Optional[int] = None) -> int:
    """The plain stage's node budget in :func:`_search_gated`: the children's columns, summed over the first level
    of :func:`_branches` (each child's LP has one column per group it keeps), so the plain search gets about as many
    nodes as re-optimisation would solve LP columns. 0 when no group is kept.

    With ``cap`` the count stops once it reaches the cap and returns ``min(count, cap)``: in full it costs the
    branches times the kept groups, and over a nearly whole pool (2.39 million groups at 36 geos and Q = 3, with
    the tens of thousands of branches that brings) that ran past half an hour, for a budget the gate then cut to
    its node limit."""
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0:
        return 0
    inc = _incidence(pool.members[keep], int(pool.unit_indices.size))
    g0 = int(np.argmin(inc.sum(axis=0)))
    total = 0
    for a in np.flatnonzero(inc[:, g0]):
        total += int((~inc[:, inc[a]].any(axis=1)).sum())
        if cap is not None and total >= cap:
            return int(cap)
    return total


def _branch_bounds(pool: ColumnPool, root: RootLP, keep: np.ndarray, k: int, Q: int):
    """Each first-level branch's fixed row and rigorous lower bound on every design holding it (:func:`_branches`)."""
    return [(j, b) for j, b, _, _ in _branches(pool, root, keep, k)]


def _search_reopt(pool: ColumnPool, root: RootLP, keep, k: int, Q: int, best_up: float,
                  node_limit: int, progress=None) -> "cover.CoverSearch":
    """The proof search with duals re-optimised one level down: a branch whose residual bound reaches the
    incumbent is pruned, and each other branch is searched with its residual LP's duals (:func:`_branches`), the
    incumbent and the node budget carried from branch to branch. Measured from the optimum at Q = 2, it placed
    0.88 million nodes against 2.89 million at 40 geos and 6.15 million against 10.35 million at 50; a second level
    halved the nodes again but cost 1,900 LPs and lost. Returns a :class:`.cover.CoverSearch`: when proved, its
    lower bound holds for every design over ``keep``. Each branch's search reports to ``progress``, whose total
    also counts each branch's fixed group, as the node count does."""
    keep = np.asarray(keep, dtype=np.int64)
    if keep.size == 0:
        return cover.CoverSearch("proved", keep, float(best_up), float(best_up), 0)
    n = int(pool.unit_indices.size)
    held = np.zeros(n, dtype=bool)
    mem = pool.members[keep]
    held[mem[mem >= 0]] = True
    if not held.all():                                # a unit no kept group covers: no design over keep
        return cover.CoverSearch("proved", np.zeros(0, np.int64), float(best_up), float(best_up), 0)
    best, rows_best, lowest, nodes, status = float(best_up), np.zeros(0, np.int64), np.inf, 0, "proved"
    cliques = _separate_cliques(pool, keep, k)
    for j, bound, left, duals in _iter_branches(pool, root, keep, k, cliques, best_up=best):
        if bound >= best:
            continue
        if nodes + 1 > node_limit:                    # the fixed group is a node of its own
            status = "node_limit"
            break
        res = cover.search(pool, duals, left, k, Q, best, node_limit - nodes - 1, fixed=(j,), progress=progress)
        nodes += res.nodes + 1
        if progress is not None:
            progress[cover.P_TOTAL] += 1.0
        if res.rows.size and res.upper < best:
            best, rows_best = res.upper, res.rows
        lowest = min(lowest, res.lower)
        if res.status == "node_limit":
            status = "node_limit"
            break
    return cover.CoverSearch(status, rows_best, best, float(min(best, lowest)), int(nodes))


def _search_gated(pool: ColumnPool, root: RootLP, keep, k: int, Q: int, best_up: float,
                  node_limit: int, progress=None) -> "cover.CoverSearch":
    """The second proof search: the plain search (:func:`.cover.search` with the root's duals) for at most
    :func:`_reopt_budget` nodes, then, if that does not prove, the re-optimised search (:func:`_search_reopt`) with
    the rest of the node limit and the better incumbent. Re-optimisation solves one LP per branch, so it pays when
    the plain search is long and the branches few; when the incumbent is far above the optimum the branches are
    many and the plain search proves first. Measured on one thread on a 20-market arm at Q = 3 with the incumbent
    at, 20% above and twice the optimum, the plain stage proved in 0.01, 0.05 and 0.52 s, where re-optimisation
    alone took 0.74, 4.3 and 77 s; on three 40-market arms at Q = 2, started from the first MIP's design, the gated
    search proved in 2.8, 3.4 and 1.6 s, where the plain search alone took 2.9, 3.8 and 2.1 s. Returns a
    :class:`.cover.CoverSearch` whose ``stage`` says which stage ended it. Both stages report to ``progress``."""
    keep = np.asarray(keep, dtype=np.int64)
    budget = min(_reopt_budget(pool, keep, cap=int(node_limit)), int(node_limit))
    best, rows, nodes = float(best_up), np.zeros(0, np.int64), 0
    if budget > 0:
        first = cover.search(pool, root, keep, k, Q, best, budget, progress=progress)
        if first.status == "proved" or budget >= node_limit:
            return replace(first, stage="plain")
        nodes = first.nodes
        if first.rows.size and first.upper < best:
            best, rows = first.upper, first.rows
    rest = _search_reopt(pool, root, keep, k, Q, best, max(int(node_limit) - nodes, 0), progress=progress)
    if rest.rows.size:
        rows = rest.rows
    return cover.CoverSearch(rest.status, rows, rest.upper, rest.lower, nodes + rest.nodes, stage="reoptimised")


def _incumbent(pool: ColumnPool, designs) -> Tuple[Optional[Fraction], Optional[np.ndarray]]:
    """The design of least rigorous upper bound among ``designs`` (arrays of pool rows), its bound the sum of its
    groups' cost_up in rationals: ``(upper, rows)``. Empty designs are skipped and a tie keeps the earlier;
    ``(None, None)`` when every design is empty."""
    best_z, best_rows = None, None
    for rows in designs:
        if len(rows) == 0:
            continue
        z = sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))
        if best_z is None or z < best_z:
            best_z, best_rows = z, rows
    return best_z, best_rows


def _search_from(pool: ColumnPool, root: RootLP, k: int, Q: int, node_limit: int, z_up: Fraction,
                 rows: Sequence[int], progress=None) -> dict:
    """The proof search over the groups a design's upper bound z_up cannot exclude, with that design as the
    incumbent (:func:`_search_gated`): returned as the best design when the search finds none cheaper. Its
    searches report to ``progress``."""
    t = time.perf_counter()
    keep = np.flatnonzero(root.reduced_costs <= _up(z_up - Fraction(root.bound)))
    res = _search_gated(pool, root, keep, k, Q, _up(z_up), node_limit, progress=progress)
    found = res.rows if res.rows.size else np.asarray(rows, dtype=np.int64)
    return dict(search=res.status, rows=found, lower=res.lower, nodes=res.nodes, kept=keep.size, stage=res.stage,
                seconds=time.perf_counter() - t)


def solve_exact(
    unit_indices, Ypre: np.ndarray, max_size: int, *, objective: str = "ss_res",
    weights: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None,
    cov_scales: Optional[np.ndarray] = None, cov_weights: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None, min_pairs: int = 1, backend: str = "auto",
    mip_rel_gap: float = 0.0, mip_abs_gap: float = 0.0, solve: Optional[Callable] = None,
    search_node_limit: Optional[int] = None, progress: Optional[np.ndarray] = None,
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
    the search placed.

    ``progress`` (:func:`.cover.new_progress`) receives every search's counts while it runs; its total ends equal to
    ``search_nodes``. Nothing is printed: a caller that wants to watch reads the array from another thread."""
    if search_node_limit is None:
        search_node_limit = cover.SEARCH_NODE_LIMIT
    if isinstance(search_node_limit, bool) or not isinstance(search_node_limit, (int, np.integer)) \
            or search_node_limit < 0:
        raise MlsynthConfigError(
            f"search_node_limit must be a non-negative integer (0 skips the search); got {search_node_limit!r}.")
    cover._check_progress(progress)
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
    info = dict(search="off", search_nodes=0, search_start=None)
    quick_rows = np.zeros(0, dtype=np.int64)          # the quick search's best design, when it found one

    def proved(found: dict, solves: int, mip_secs: float):
        """The design a finished search proved, reported as the MIP path reports one (its gap relative to the
        design's cost, absolute when that cost is 0)."""
        rows = found["rows"]
        chosen = pool.candidates(rows)
        upper = _up(sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0)))
        lower = min(found["lower"], upper)
        z = math.fsum(float(pool.cost[j]) for j in rows)
        d = dict(path="exact_search", backend="search", solver_name="exact-cover search", status="optimal",
                 presolve_retry=False, mip_rel_gap_requested=float(mip_rel_gap),
                 mip_abs_gap_requested=float(mip_abs_gap), gap_enforced=True,
                 mip_gap=(upper - lower) / abs(z) if z != 0 else upper - lower, dual_bound=float(lower),
                 node_count=float(info["search_nodes"]), solve_seconds=mip_secs + found["seconds"])
        d.update({**extra, **info}, columns_solved=int(found["kept"]), lagrangian_bound=float(L),
                 lp_value=root.lp_value, lp_rounds=root.rounds, mip_solves=solves, lp_seconds=lp_seconds,
                 objective_value=z, objective_upper=upper, n_selected_pairs=len(chosen))
        return chosen, d
    if backend != "auto":
        pass                                          # a MIP backend asked for by name
    elif search_node_limit > 0 and not cover._NUMBA:
        info["search"] = "unavailable"   # pragma: no cover - the suite runs with numba installed
    elif search_node_limit > 0:
        quick = min(int(cover.SEARCH_QUICK_LIMIT), int(search_node_limit))
        found = _search(pool, root, k, max_size, quick, progress=progress)
        info.update(search=found["search"], search_nodes=found["nodes"])
        quick_rows = np.asarray(found["rows"], dtype=np.int64)
        if found["search"] == "proved":
            info["search_start"] = "staged"
            return proved(found, 0, 0.0)

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
        if solves == 1 and info["search"] == "node_limit":
            # the proof search again, from the better of the first MIP's design and the quick search's: a first
            # MIP held to too few groups can return a design far above the quick search's (920 against 1.10 on a
            # 36-geo Q = 3 arm), and its margin would keep nearly every group
            mip_rows = np.asarray([column_index(tuple(sorted(pos[int(u)] for u in c["members"])), n)
                                   for c in best[0]], dtype=np.int64)
            z_from, rows_from = _incumbent(pool, [mip_rows, quick_rows])
            found = _search_from(pool, root, k, max_size, int(search_node_limit), z_from, rows_from,
                                 progress=progress)
            info["search_nodes"] += found["nodes"]
            info["second_search"] = found["stage"]
            if found["search"] == "proved":
                info.update(search="proved", search_start="first_mip")
                return proved(found, solves, mip_seconds)
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
