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
nodes, gives a design whose upper bound sets the reduced-cost margin for the proof search. A window that holds no
design is widened ``START_GROWTH``-fold and searched again, up to the whole pool: at Q = 4 the first window can cover
every unit and hold no exact cover (a 24-geo seasonal arm's 960 groups, proved empty in 614 nodes), and a proof
search with no incumbent keeps every group (1.27 million there, 666 s).

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
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ...exceptions import MlsynthConfigError

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
#: The quick search from the staged start places at most this many nodes (and at most the limit above). A search
#: it stops runs again, up to SEARCH_NODE_LIMIT, from the design of the MIP loop's first MIP.
SEARCH_QUICK_LIMIT = 1_500_000
#: The start searches the START_PER_UNIT * n groups of least reduced cost, for at most START_NODE_LIMIT nodes, and
#: widens the window START_GROWTH-fold while it holds no design.
START_PER_UNIT = 40
START_NODE_LIMIT = 200_000
START_GROWTH = 4

_U = 2.0 ** -53
#: The underflow unit, the smallest positive subnormal: a product's absolute error below the normal range (Rump 2010,
#: eq. 2.3; see :mod:`.columns`).
_ETA = 2.0 ** -1074


#: The progress array (:func:`new_progress`) a caller may pass to :func:`search` to watch it run: its slots, by
#: index. The search writes the per-search slots every ``PROGRESS_EVERY`` nodes, from compiled code that releases
#: the GIL, so another thread can read them while it runs; it writes them all when it ends. Nothing is printed, and
#: the result does not depend on the array.
P_SEARCHES = 0      # searches started on this array
P_NODES = 1         # groups placed by the current (or last) search
P_TOTAL = 2         # groups placed by every search on this array
P_INCUMBENT = 3     # the current search's incumbent upper bound (inf before a design)
P_DEPTH = 4         # the depth of the node last written
P_ROOT_DONE = 5     # the root's branches started
P_ROOT_TOTAL = 6    # the root's branches
P_LOWEST = 7        # the lowest leaf lower bound seen (when the search ends, its ``lower``)
P_DONE = 8          # 1 once the current search has ended, 0 while it runs
PROGRESS_SLOTS = 9
PROGRESS_EVERY = 4096


def new_progress() -> np.ndarray:
    """A progress array for :func:`search`: counts at 0, no incumbent and no leaf yet."""
    prog = np.zeros(PROGRESS_SLOTS, dtype=np.float64)
    prog[P_INCUMBENT] = np.inf
    prog[P_LOWEST] = np.inf
    return prog


def _check_progress(progress) -> None:
    """Refuse anything but a writable, contiguous, one-dimensional float64 array of PROGRESS_SLOTS doubles."""
    if progress is None:
        return
    if not (isinstance(progress, np.ndarray) and progress.dtype == np.float64 and progress.ndim == 1
            and progress.size == PROGRESS_SLOTS and progress.flags.c_contiguous and progress.flags.writeable):
        raise MlsynthConfigError(
            "progress must be a writable, contiguous one-dimensional float64 array of "
            f"{PROGRESS_SLOTS} doubles (cover.new_progress()); got {type(progress).__name__}"
            + (f" of dtype {progress.dtype}, shape {progress.shape}" if isinstance(progress, np.ndarray) else ""))


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
    stage : str or None
        For the second proof search (:func:`.columns._search_gated`), the stage that ended it: ``"plain"`` when
        the plain search proved within its budget, ``"reoptimised"`` when the re-optimised search ran. None for
        any other search.
    """

    status: str
    rows: np.ndarray
    upper: float
    lower: float
    nodes: int
    stage: Optional[str] = None


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
def _threshold(best_up, muS):  # pragma: no cover - compiled; checked through search
    """An upper bound on best_up + muS in exact arithmetic (best_up itself when muS is 0)."""
    if muS == 0.0:
        return best_up
    t = best_up + muS
    return t + 2.0 * _U * abs(t)


@njit(cache=_CACHE)
def _report(prog, nodes, base, best_up, depth, root_done, root_total):  # pragma: no cover - compiled
    prog[P_NODES] = nodes
    prog[P_TOTAL] = base + nodes
    prog[P_INCUMBENT] = best_up
    prog[P_DEPTH] = depth
    prog[P_ROOT_DONE] = root_done
    prog[P_ROOT_TOTAL] = root_total


@njit(cache=_CACHE, nogil=True)
def _cover(n, W, rows, size, lo, up, tlo, pisum, gptr, gcols, S0, absA0, sm, k, Q, best_up, g2,
           node_limit, unc, P0, R0, T0, kptr, kmem, mu, gkptr, gk, muS, gK, shifted,
           prog):  # pragma: no cover - compiled
    """Depth-first search over exact covers. Returns (status, nodes, best_up, lowest, best_path).

    ``lo`` are the (shifted) cost lower bounds the node bound sums, ``tlo`` the true ones the leaves' lower bounds
    sum, ``up`` the upper bounds the incumbent sums. Without a shift (``shifted`` False) ``lo`` is ``tlo`` and
    ``P0`` is ``T0``, so the running sum at a leaf is the true-cost sum, the same numbers added in the same order,
    and the leaf reads it; summing the path again at every leaf cost 22% of the search's time on a 32-geo arm. Cliques ``kptr``/``kmem`` (local rows) with duals ``mu`` are
    credited at a node with no chosen member (``gkptr``/``gk`` list each row's cliques) and no member that fits;
    nodes are pruned at the incumbent plus ``muS``. ``prog`` receives the counts every PROGRESS_EVERY nodes
    (:func:`new_progress`)."""
    maxd = n // 2 + 1
    base = prog[P_TOTAL]
    Ud = np.zeros((maxd + 1, W), np.uint64)
    Pd = np.zeros(maxd + 1)
    Rd = np.zeros(maxd + 1)
    nK = kptr.size - 1
    hit = np.zeros(max(nK, 1), np.int64)     # chosen members of each clique on the current path
    last = np.full(maxd + 1, -1, np.int64)   # the row whose cliques are counted at each depth
    Sd = np.zeros(maxd + 1)
    Nd = np.zeros(maxd + 1, np.int64)
    Gd = np.zeros(maxd + 1, np.int64)
    Kd = np.zeros(maxd + 1, np.int64)
    path = np.zeros(maxd + 1, np.int64)
    best_path = np.full(maxd + 1, -1, np.int64)
    n0 = 0
    for i in range(n):
        if unc[i]:
            Ud[0, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
            n0 += 1
    Sd[0] = S0
    Nd[0] = n0
    Pd[0] = P0
    Rd[0] = R0
    thr = _threshold(best_up, muS)
    lowest = np.inf
    nodes = 0
    status = 0
    bc = 1 << 60
    g0 = -1
    for i in range(n):                       # the root branches on the uncovered unit with the fewest groups
        if not unc[i]:
            continue
        c = gptr[i + 1] - gptr[i]
        if c < bc:
            bc = c
            g0 = i
    if bc == 0:
        return 0, 0, best_up, lowest, best_path
    Gd[0] = g0
    Kd[0] = gptr[g0]
    root_total = gptr[g0 + 1] - gptr[g0]
    depth = 0
    while depth >= 0:
        if nK > 0 and last[depth] >= 0:      # the previous sibling's cliques are no longer chosen
            c0 = last[depth]
            for q in range(gkptr[c0], gkptr[c0 + 1]):
                hit[gk[q]] -= 1
            last[depth] = -1
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
        if nodes % PROGRESS_EVERY == 0:
            _report(prog, nodes, base, best_up, depth, Kd[0] - gptr[g0], root_total)
        if nK > 0:
            for q in range(gkptr[col], gkptr[col + 1]):
                hit[gk[q]] += 1
            last[depth] = col
        c1 = depth + 1
        P1 = Pd[depth] + lo[col]
        R1 = Rd[depth] + up[col]
        S1 = Sd[depth] - pisum[col]
        u1 = Nd[depth] - size[col]
        path[depth] = col
        if u1 == 0:                          # a design
            if c1 < k:
                continue
            if shifted:
                T1 = T0                      # the true cost bounds, summed at leaves only
                for t in range(c1):
                    T1 += tlo[path[t]]
            else:
                T1 = P1                      # the same sum: lo is tlo and P0 is T0
            leaf = T1 - g2 * (T1 + absA0)
            if leaf < lowest:
                lowest = leaf
            r_up = R1 + g2 * R1
            if r_up < best_up:
                best_up = r_up
                thr = _threshold(best_up, muS)
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
        margin = g2 * (P1 + absA0 + r_hi * abs(sm)) + _ETA             # rt's product may underflow
        if lb - margin >= thr:
            continue
        for w in range(W):
            Ud[depth + 1, w] = Ud[depth, w]
        for t in range(rows.shape[1]):
            p = rows[col, t]
            if p < 0:
                break
            Ud[depth + 1, p >> 6] &= ~(np.uint64(1) << np.uint64(p & 63))
        if nK > 0:                           # credit each clique with slack 1 in every completion
            cr = 0.0
            for K in range(nK):
                if hit[K] > 0 or mu[K] <= 0.0:
                    continue
                alive = False
                for q in range(kptr[K], kptr[K + 1]):
                    if _fits(rows, kmem[q], Ud[depth + 1]):
                        alive = True
                        break
                if not alive:
                    cr += mu[K]
            if cr > 0.0 and lb + cr - margin - gK * (abs(lb) + cr) >= thr:
                continue
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


def _clique_arrays(keep, fixed, cliques, mu):
    """The cliques the kernel credits, in its index: those with no fixed member (a fixed member hits its cliques
    in every completion), each over its members in ``keep`` as positions in ``keep``, with its dual, and each kept
    row's list of those cliques. A clique with no kept member is credited at every node: its slack is 1 in every
    design over ``keep``."""
    pos = {int(j): t for t, j in enumerate(keep)}
    fx = set(int(j) for j in fixed)
    kmem, kptr, mus = [], [0], []
    per = [[] for _ in range(keep.size)]
    for K, m in zip(cliques, mu):
        if fx & set(int(j) for j in K):
            continue
        local = [pos[int(j)] for j in K if int(j) in pos]
        for t in local:
            per[t].append(len(mus))
        kmem += local
        kptr.append(len(kmem))
        mus.append(float(m))
    gkptr = np.zeros(keep.size + 1, dtype=np.int64)
    gkptr[1:] = np.cumsum([len(p) for p in per])
    gk = np.array([K for p in per for K in p], dtype=np.int64)
    return (np.asarray(kptr, dtype=np.int64), np.asarray(kmem, dtype=np.int64), np.asarray(mus, dtype=float),
            gkptr, gk)


def search(pool, root, keep, k: int, Q: int, best_up: float, node_limit: int, fixed=(),
           progress: Optional[np.ndarray] = None) -> CoverSearch:
    """Search the exact covers of the arm by the pool rows ``keep`` for one whose upper bound is below ``best_up``,
    with at least ``k`` groups, placing at most ``node_limit`` groups. ``root`` holds the duals: ``pi``, ``sigma``
    and rigorous lower bounds on every pool row's ``reduced_costs`` (:func:`.columns.root_lp`'s, or a residual
    LP's). ``fixed`` are pool rows already chosen: their units start covered, their cost bounds start the sums,
    they count toward ``k``, and they are part of the design returned. The rows in ``keep`` are disjoint from
    them.

    Duals from an LP with clique rows (:func:`.columns._residual_lp`) also carry ``cliques``, their duals ``mu``
    and lower bounds ``cost_lo`` on the shifted costs. The node bound then sums the shifted costs, a node is pruned
    when the bound reaches ``best_up`` plus the sum of ``mu``, and every clique with no chosen member and no member
    that still fits adds its dual, since its slack is 1 in every completion. Leaves' lower bounds and the incumbent
    use the true cost bounds.

    ``progress`` (:func:`new_progress`) receives the search's counts while it runs and when it ends, and adds its
    nodes to the array's total; the result does not depend on it."""
    _check_progress(progress)
    prog = new_progress() if progress is None else progress
    base = float(prog[P_TOTAL])
    prog[P_SEARCHES] += 1.0
    prog[P_DONE] = 0.0
    prog[P_NODES] = 0.0
    prog[P_INCUMBENT] = float(best_up)
    prog[P_LOWEST] = np.inf
    prog[P_DEPTH] = prog[P_ROOT_DONE] = prog[P_ROOT_TOTAL] = 0.0
    res = _search(pool, root, keep, k, Q, best_up, node_limit, fixed, prog)
    prog[P_NODES] = float(res.nodes)
    prog[P_TOTAL] = base + float(res.nodes)
    prog[P_INCUMBENT] = res.upper
    prog[P_LOWEST] = res.lower
    prog[P_DONE] = 1.0
    return res


def _search(pool, root, keep, k, Q, best_up, node_limit, fixed, prog) -> CoverSearch:
    """:func:`search`'s work, with ``prog`` the array the kernel writes."""
    keep = np.asarray(keep, dtype=np.int64)
    fixed = np.asarray(fixed, dtype=np.int64)
    n = int(pool.unit_indices.size)
    cliques = tuple(getattr(root, "cliques", ()))
    mu_all = np.asarray(getattr(root, "mu", np.zeros(0)), dtype=float)
    shifted = getattr(root, "cost_lo", None)
    has_shift = shifted is not None
    shifted = pool.cost_lo if shifted is None else np.asarray(shifted, dtype=float)
    unc = np.ones(n, dtype=np.bool_)
    for j in fixed:
        unc[pool.members[j][pool.members[j] >= 0]] = False
    P0 = float(shifted[fixed].sum()) if fixed.size else 0.0
    R0 = float(pool.cost_up[fixed].sum()) if fixed.size else 0.0
    T0 = float(pool.cost_lo[fixed].sum()) if fixed.size else 0.0
    k_rem = max(int(k) - int(fixed.size), 0)
    g2 = 2.0 * _gamma(2 * n + 8) + 4.0 * _U
    if not unc.any():                        # the fixed groups are the design
        upper = R0 + g2 * R0
        if k_rem > 0 or upper >= best_up:
            return CoverSearch("proved", np.zeros(0, np.int64), float(best_up), float(best_up), 0)
        return CoverSearch("proved", fixed, float(upper), float(min(upper, T0 - g2 * T0)), 0)
    if keep.size == 0:
        return CoverSearch("proved", keep, float(best_up), float(best_up), 0)
    rows = np.ascontiguousarray(pool.members[keep], dtype=np.int64)
    lo = np.ascontiguousarray(shifted[keep], dtype=float)
    tlo = np.ascontiguousarray(pool.cost_lo[keep], dtype=float)
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
    pi_unc = np.asarray(root.pi, dtype=float)[unc]
    S0 = math.fsum(float(x) for x in pi_unc)
    absA0 = float(np.abs(pi_unc).sum()) * (1.0 + _gamma(n))
    kptr, kmem, mu, gkptr, gk = _clique_arrays(keep, fixed, cliques, mu_all)
    muS = math.fsum(float(m) for m in mu_all)
    muS = muS * (1.0 + 2.0 * _U) if muS > 0 else 0.0
    gK = _gamma(int(mu.size) + 4)
    W = -(-n // 64)
    status, nodes, upper, lowest, path = _cover(
        n, W, rows, size, lo, up, tlo, pisum, gptr, gcols, S0, absA0, sm, k_rem, int(Q), float(best_up), g2,
        int(node_limit), unc, P0, R0, T0, kptr, kmem, mu, gkptr, gk, float(muS), gK, has_shift, prog)
    found = keep[path[path >= 0]]
    if found.size and fixed.size:
        found = np.concatenate([fixed, found])
    return CoverSearch("node_limit" if status else "proved", found, float(upper),
                       float(min(upper, lowest)), int(nodes))


def start_design(pool, root, k: int, Q: int, progress: Optional[np.ndarray] = None) -> CoverSearch:
    """A first design: a search capped at ``START_NODE_LIMIT`` nodes over the ``START_PER_UNIT`` n groups of least
    reduced cost, repeated over ``START_GROWTH`` times as many groups while a window holds no design, up to the whole
    pool. Each search reports to ``progress`` as :func:`search` does; the result's nodes are all the searches'."""
    rc = np.asarray(root.reduced_costs, dtype=float)
    count = min(rc.size, START_PER_UNIT * int(pool.unit_indices.size))
    nodes = 0
    while True:
        small = np.flatnonzero(rc <= np.partition(rc, count - 1)[count - 1])
        res = search(pool, root, small, k, Q, np.inf, START_NODE_LIMIT, progress=progress)
        nodes += res.nodes
        if res.rows.size or count >= rc.size:
            return replace(res, nodes=nodes)
        count = min(rc.size, count * START_GROWTH)
