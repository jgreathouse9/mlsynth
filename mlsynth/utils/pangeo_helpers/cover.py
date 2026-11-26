"""An exact-cover search that proves PANGEO's exact design without a MIP.

The design is an exact cover of an arm's units by groups (:mod:`.columns`), the cheapest cover with at least ``k``
groups. The search enumerates covers depth first, in the order of Knuth's Algorithm X ("Dancing links", in
*Millennial Perspectives in Computer Science*, Palgrave, 2000, 187-214): at each node it branches on the uncovered
unit with the fewest groups that still fit, and tries those groups in order of cost.

Pruning uses the root duals. For duals :math:`\\pi` on the cover rows and :math:`\\sigma \\ge 0` on the cardinality
row, every group's cost is :math:`c_G = \\bar c_G + \\sum_{i \\in G} \\pi_i + \\sigma`, so a partial design with
chosen groups :math:`P` and uncovered units :math:`U`, completed by :math:`r` more groups, costs at least

.. math::

   \\sum_{G \\in P} c_G + \\sum_{i \\in U} \\pi_i + r\\,(\\sigma + \\min_G \\bar c_G),

and :math:`r` lies between :math:`\\max(\\lceil |U| / 2Q \\rceil, k - |P|)` and :math:`\\lfloor |U| / 2 \\rfloor`,
so the bound takes the smaller of the two ends. That is the Lagrangian bound of the remaining set-partitioning
problem at the root duals (Geoffrion, "Lagrangean relaxation for integer programming", Math. Programming Study 2
(1974) 82-114). A partial design whose bound reaches the best design's upper bound is pruned.

Rigour. Groups are pruned on the lower bounds of their exact costs and designs are kept on their upper bounds
(:class:`.columns.ColumnPool`), and each bound is compared with its own floating-point error subtracted: a sum of
at most 2n + 8 terms carries at most :math:`\\gamma_{2n+8}` relative error (Higham, "Accuracy and Stability of
Numerical Algorithms", 2nd ed., SIAM, 2002, Lemma 3.1; Rump, Acta Numerica 19 (2010) 287-449, Theorem 2.1). A search
that ends under its node cap has seen every design over the groups searched whose bound is below the best upper
bound, so the least of that bound and the lowest leaf's lower bound is a rigorous lower bound on the optimum over
those groups. The groups left out have reduced cost above :math:`z - L` for the start design's upper bound
:math:`z` (reduced-cost fixing, :mod:`.columns`), so they are in no cheaper design.

Start. A first search over the ``START_PER_UNIT`` n groups of least reduced cost, capped at ``START_NODE_LIMIT``
nodes, gives a design whose upper bound sets the reduced-cost margin for the proof search.

Measured on simulated seasonal panels (one arm, 42 estimation periods, single-threaded), the search proved the optimum
in 1.6 s against 21.3 s for the MIP loop at 24 markets and Q = 3, 2.9 s against 47.7 s at 28 and 10.9 s against
76.6 s at 32; at Q = 2, 0.10 s against 5.4 s at 30 markets. At Q = 2 with 50 markets the proof needed 10 million
nodes even from the optimum, and lost to the MIP loop (32.8 s against 24.4 s): the root bound was 8.4% below the
optimum and the search's bound does not tighten with depth, where HiGHS re-solves the LP and adds cuts at every node.
The node cap, ``SEARCH_NODE_LIMIT``, bounds what such a panel costs before the MIP loop takes over: at 1.5 million
nodes, 8.2 s at 40 markets and 8.6 s at 50, against MIP loops of 18.7 s and 27.0 s, while Q = 3 at 32 markets
still proves under it (1.38 million nodes). A search stopped by the cap does not hand its design to the MIP loop:
on the 50-market arm, adding it to the first MIP's groups made HiGHS take 58 to 64 s over the same groups, against
26 to 29 s without it, and at 40 markets it changed nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # numba is the optional ``fast`` extra; without it the search is skipped and the MIP loop runs
    from numba import njit
    _NUMBA = True
except ImportError:  # pragma: no cover - the suite runs with numba installed
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name (see :mod:`.columns`).
CANONICAL_NAME = "mlsynth.utils.pangeo_helpers.cover"
_CACHE = __name__ == CANONICAL_NAME

#: The proof search stops after this many nodes and the MIP loop takes over; about 7 s at Q = 2 and 50 markets.
SEARCH_NODE_LIMIT = 1_500_000
#: The start searches the START_PER_UNIT * n groups of least reduced cost, for at most START_NODE_LIMIT nodes.
START_PER_UNIT = 40
START_NODE_LIMIT = 200_000

_U = 2.0 ** -53
#: The underflow unit, the smallest positive subnormal: a product's absolute error below the normal range (Rump 2010,
#: eq. 2.3; see :mod:`.columns`).
_ETA = 2.0 ** -1074


def _gamma(k: int) -> float:
    """gamma_k = k u / (1 - k u), inflated past its own rounding."""
    return k * _U / (1.0 - k * _U) * (1.0 + 4.0 * _U)


@dataclass(frozen=True)
class CoverSearch:
    """The outcome of one search.

    Attributes
    ----------
    status : str
        ``"proved"`` when the search ended under its node cap, ``"node_limit"`` when the cap stopped it.
    rows : np.ndarray
        Pool rows of the best design found below the incumbent bound passed in; empty when none was.
    upper : float
        A rigorous upper bound on that design's exact cost, or the incumbent bound when no design was found.
    lower : float
        When proved, a rigorous lower bound on every design over the groups searched: the least of ``upper`` and
        the lowest leaf's lower bound. Not a bound when the cap stopped the search.
    nodes : int
        Groups placed.
    """

    status: str
    rows: np.ndarray
    upper: float
    lower: float
    nodes: int


@njit(cache=_CACHE)
def _fits(rows, col, Ucur):  # pragma: no cover - compiled; checked through search
    for t in range(rows.shape[1]):
        p = rows[col, t]
        if p < 0:
            break
        if (Ucur[p >> 6] >> np.uint64(p & 63)) & np.uint64(1) == np.uint64(0):
            return False
    return True


@njit(cache=_CACHE)
def _cover(n, W, rows, size, lo, up, pisum, gptr, gcols, S0, absA0, sm, k, Q, best_up, g2,
           node_limit):  # pragma: no cover - compiled; checked through search
    """Depth-first search over exact covers. Returns (status, nodes, best_up, lowest, best_path)."""
    maxd = n // 2 + 1
    Ud = np.zeros((maxd + 1, W), np.uint64)
    Pd = np.zeros(maxd + 1)
    Rd = np.zeros(maxd + 1)
    Sd = np.zeros(maxd + 1)
    Nd = np.zeros(maxd + 1, np.int64)
    Gd = np.zeros(maxd + 1, np.int64)
    Kd = np.zeros(maxd + 1, np.int64)
    path = np.zeros(maxd + 1, np.int64)
    best_path = np.full(maxd + 1, -1, np.int64)
    for i in range(n):
        Ud[0, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    Sd[0] = S0
    Nd[0] = n
    lowest = np.inf
    nodes = 0
    status = 0
    bc = 1 << 60
    g0 = -1
    for i in range(n):                       # the root branches on the unit with the fewest groups
        c = gptr[i + 1] - gptr[i]
        if c < bc:
            bc = c
            g0 = i
    if bc == 0:
        return 0, 0, best_up, lowest, best_path
    Gd[0] = g0
    Kd[0] = gptr[g0]
    depth = 0
    while depth >= 0:
        g = Gd[depth]
        if Kd[depth] >= gptr[g + 1]:
            depth -= 1
            continue
        col = gcols[Kd[depth]]
        Kd[depth] += 1
        if not _fits(rows, col, Ud[depth]):
            continue
        nodes += 1
        if nodes > node_limit:
            status = 1
            break
        c1 = depth + 1
        P1 = Pd[depth] + lo[col]
        R1 = Rd[depth] + up[col]
        S1 = Sd[depth] - pisum[col]
        u1 = Nd[depth] - size[col]
        path[depth] = col
        if u1 == 0:                          # a design
            if c1 < k:
                continue
            leaf = P1 - g2 * (P1 + absA0)
            if leaf < lowest:
                lowest = leaf
            r_up = R1 + g2 * R1
            if r_up < best_up:
                best_up = r_up
                for t in range(c1):
                    best_path[t] = path[t]
                for t in range(c1, maxd + 1):
                    best_path[t] = -1
            continue
        r_lo = (u1 + 2 * Q - 1) // (2 * Q)   # groups still to place: at least ceil(u / 2Q) and k - c1
        if k - c1 > r_lo:
            r_lo = k - c1
        r_hi = u1 // 2
        if r_lo > r_hi:
            continue
        rt = r_lo * sm if r_lo * sm < r_hi * sm else r_hi * sm
        lb = P1 + S1 + rt
        if lb - (g2 * (P1 + absA0 + r_hi * abs(sm)) + _ETA) >= best_up:   # rt's product may underflow
            continue
        for w in range(W):
            Ud[depth + 1, w] = Ud[depth, w]
        for t in range(rows.shape[1]):
            p = rows[col, t]
            if p < 0:
                break
            Ud[depth + 1, p >> 6] &= ~(np.uint64(1) << np.uint64(p & 63))
        bc = 1 << 60                         # branch on the uncovered unit with the fewest groups that fit
        gb = -1
        for w in range(W):
            x = Ud[depth + 1, w]
            while x != np.uint64(0):
                b = 0
                y = x
                while (y & np.uint64(1)) == np.uint64(0):
                    y >>= np.uint64(1)
                    b += 1
                x &= x - np.uint64(1)
                i = w * 64 + b
                c = 0
                for q in range(gptr[i], gptr[i + 1]):
                    if _fits(rows, gcols[q], Ud[depth + 1]):
                        c += 1
                        if c >= bc:
                            break
                if c < bc:
                    bc = c
                    gb = i
                    if c == 0:
                        break
            if bc == 0:
                break
        if bc == 0:                          # a unit nothing fits ends the branch
            continue
        depth += 1
        Pd[depth] = P1
        Rd[depth] = R1
        Sd[depth] = S1
        Nd[depth] = u1
        Gd[depth] = gb
        Kd[depth] = gptr[gb]
    return status, nodes, best_up, lowest, best_path


def search(pool, root, keep, k: int, Q: int, best_up: float, node_limit: int) -> CoverSearch:
    """Search the exact covers of the arm by the pool rows ``keep`` for one whose upper bound is below ``best_up``,
    with at least ``k`` groups, placing at most ``node_limit`` groups. ``root`` is :func:`.columns.root_lp`'s."""
    keep = np.asarray(keep, dtype=np.int64)
    n = int(pool.unit_indices.size)
    if keep.size == 0:
        return CoverSearch("proved", keep, float(best_up), float(best_up), 0)
    rows = np.ascontiguousarray(pool.members[keep], dtype=np.int64)
    lo = np.ascontiguousarray(pool.cost_lo[keep], dtype=float)
    up = np.ascontiguousarray(pool.cost_up[keep], dtype=float)
    rc = np.asarray(root.reduced_costs, dtype=float)[keep]
    C = rows.shape[0]
    size = (rows >= 0).sum(axis=1).astype(np.int64)
    pad = np.append(np.asarray(root.pi, dtype=float), 0.0)
    pisum = pad[rows].sum(axis=1)
    unit = rows.ravel()
    col = np.repeat(np.arange(C), rows.shape[1])
    ok = unit >= 0
    unit, col = unit[ok], col[ok]
    order = np.lexsort((lo[col], unit))      # each unit's groups, cheapest first
    gcols = col[order].astype(np.int64)
    gptr = np.zeros(n + 1, dtype=np.int64)
    gptr[1:] = np.cumsum(np.bincount(unit, minlength=n))
    sm = float(root.sigma) + float(rc.min())
    S0 = math.fsum(float(x) for x in root.pi)
    absA0 = float(np.abs(root.pi).sum()) * (1.0 + _gamma(n))
    g2 = 2.0 * _gamma(2 * n + 8) + 4.0 * _U
    W = -(-n // 64)
    status, nodes, upper, lowest, path = _cover(
        n, W, rows, size, lo, up, pisum, gptr, gcols, S0, absA0, sm, int(k), int(Q), float(best_up), g2,
        int(node_limit))
    found = keep[path[path >= 0]]
    return CoverSearch("node_limit" if status else "proved", found, float(upper),
                       float(min(upper, lowest)), int(nodes))


def start_design(pool, root, k: int, Q: int) -> CoverSearch:
    """A first design: a search capped at ``START_NODE_LIMIT`` nodes over the ``START_PER_UNIT`` n groups of least
    reduced cost."""
    rc = np.asarray(root.reduced_costs, dtype=float)
    count = min(rc.size, START_PER_UNIT * int(pool.unit_indices.size))
    small = np.flatnonzero(rc <= np.partition(rc, count - 1)[count - 1])
    return search(pool, root, small, k, Q, np.inf, START_NODE_LIMIT)
