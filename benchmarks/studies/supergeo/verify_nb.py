"""Exact optimum over the surviving columns: numba exact-cover search with
floating-point pruning under a proven rounding margin, exact leaves.

Search (Knuth's Algorithm X order): branch on the uncovered geo with the fewest
surviving columns that fit; every cover covers it with exactly one column, so
each exact cover is reached once. A geo with no fitting column ends the branch.

Node bound, valid for any duals (the Lagrangian bound of the certificate,
restricted to the uncovered geos U):
    partial + sum_U pi + (|U| / lo) * m + tau [N odd and no odd column yet],
m = exact min(0, min reduced cost) <= 0. A node is pruned when the bound, evaluated
in floating point, minus a margin that bounds its rounding, reaches the
incumbent's upper value. Margin: every bound is a sum of at most 2N + 8 terms
whose magnitudes sum to at most A = partial + sum|pi| + (N/lo)|m| + tau, so its
floating-point error is at most gamma_{2N+8} A (Rump 2010, eq. 2.3); the margin
is twice that, plus the error of converting the exact inputs to doubles.

Leaves: a cover whose floating-point cost is within the margin of the incumbent
is recorded and re-costed exactly by the caller. A cover certainly cheaper than
the incumbent replaces the incumbent's upper value by its own (cost + margin),
which is an upper bound on its exact cost, so later pruning stays valid.
"""
from fractions import Fraction
import math
import numpy as np
from numba import njit

U = 2.0 ** -53


def gamma(n):
    return n * U / (1 - n * U)


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - 1; c += 1
    return c


@njit(cache=True)
def _search(n, lo, masks, cost, pisum, odd, gptr, gcols, pi, m, tau, opt_up, g2, absA0, node_limit, cand_cap, plen_cap):
    full = (np.int64(1) << n) - 1
    maxd = n // 2 + 2
    Ud = np.zeros(maxd, np.int64); Pd = np.zeros(maxd); Sd = np.zeros(maxd); Od = np.zeros(maxd, np.bool_)
    Gd = np.zeros(maxd, np.int64); Kd = np.zeros(maxd, np.int64); path = np.zeros(maxd, np.int64)
    cands = np.full((cand_cap, plen_cap), -1, np.int64); nc = 0
    best = opt_up
    S0 = 0.0
    for i in range(n): S0 += pi[i]
    nodes = 0; status = 0                                   # 0 done, 1 node limit, 2 candidate cap
    # choose the branching geo at the root
    def_g = -1
    # depth 0 frame
    Ud[0] = full; Pd[0] = 0.0; Sd[0] = S0; Od[0] = False
    # MRV at root
    bestc = 1 << 30; g0 = -1
    for i in range(n):
        c = 0
        for p in range(gptr[i], gptr[i + 1]):
            if masks[gcols[p]] & full == masks[gcols[p]]: c += 1
        if c < bestc: bestc = c; g0 = i
    if bestc == 0:
        return status, nodes, best, cands[:0]
    Gd[0] = g0; Kd[0] = gptr[g0]
    depth = 0
    while depth >= 0:
        g = Gd[depth]
        if Kd[depth] >= gptr[g + 1]:
            depth -= 1
            continue
        col = gcols[Kd[depth]]; Kd[depth] += 1
        U0 = Ud[depth]
        mk = masks[col]
        if mk & U0 != mk:
            continue
        nodes += 1
        if nodes > node_limit:
            status = 1; break
        U1 = U0 & ~mk
        P1 = Pd[depth] + cost[col]
        S1 = Sd[depth] - pisum[col]
        O1 = Od[depth] or odd[col]
        path[depth] = col
        A = P1 + absA0
        err = g2 * A
        if U1 == 0:
            if P1 - err < best:                             # may beat the incumbent: record for the exact check
                if nc >= cand_cap:
                    status = 2; break
                for t in range(depth + 1):
                    cands[nc, t] = path[t]
                nc += 1
                if P1 + err < best:
                    best = P1 + err                         # an upper bound on this cover's exact cost
            continue
        lb = P1 + S1 + (_popcount(U1) / lo) * m
        if (n % 2 == 1) and not O1:
            lb += tau
        if lb - err >= best:
            continue
        # MRV over the uncovered geos; a geo with no fitting column ends the branch
        bestc = 1 << 30; gb = -1
        x = U1
        while x:
            low = x & -x
            i = 0
            y = low
            while y > 1:
                y >>= 1; i += 1
            x ^= low
            c = 0
            for p in range(gptr[i], gptr[i + 1]):
                q = masks[gcols[p]]
                if q & U1 == q:
                    c += 1
                    if c >= bestc: break
            if c < bestc:
                bestc = c; gb = i
                if c == 0: break
        if bestc == 0:
            continue
        depth += 1
        Ud[depth] = U1; Pd[depth] = P1; Sd[depth] = S1; Od[depth] = O1
        Gd[depth] = gb; Kd[depth] = gptr[gb]
    return status, nodes, best, cands[:nc]


@njit(cache=True)
def _fits(masks, col, U, W):
    """Whether column col (W words) lies inside the uncovered set U."""
    for w in range(W):
        q = masks[col, w]
        if q & U[w] != q:
            return False
    return True


@njit(cache=True)
def _search_w(n, B, W, lo, masks, cost, pisum, odd, gptr, gcols, pi, m, tau, opt_up, g2, absA0, node_limit, cand_cap,
              plen_cap):
    """_search with the uncovered set and every column held as W words of B bits (market i in word i // B, bit
    i % B; B <= 63 keeps the sign bit clear). Markets are scanned in increasing order across the words and ties
    break as in _search, so the two walk the same tree."""
    maxd = n // 2 + 2
    Ud = np.zeros((maxd, W), np.int64); Pd = np.zeros(maxd); Sd = np.zeros(maxd); Od = np.zeros(maxd, np.bool_)
    Gd = np.zeros(maxd, np.int64); Kd = np.zeros(maxd, np.int64); path = np.zeros(maxd, np.int64)
    cands = np.full((cand_cap, plen_cap), -1, np.int64); nc = 0
    best = opt_up
    S0 = 0.0
    for i in range(n): S0 += pi[i]
    nodes = 0; status = 0
    for w in range(W):
        Ud[0, w] = (np.int64(1) << min(B, n - w * B)) - 1
    Pd[0] = 0.0; Sd[0] = S0; Od[0] = False
    bestc = 1 << 30; g0 = -1
    for i in range(n):
        c = 0
        for p in range(gptr[i], gptr[i + 1]):
            if _fits(masks, gcols[p], Ud[0], W): c += 1
        if c < bestc: bestc = c; g0 = i
    if bestc == 0:
        return status, nodes, best, cands[:0]
    Gd[0] = g0; Kd[0] = gptr[g0]
    depth = 0
    U1 = np.zeros(W, np.int64)
    while depth >= 0:
        g = Gd[depth]
        if Kd[depth] >= gptr[g + 1]:
            depth -= 1
            continue
        col = gcols[Kd[depth]]; Kd[depth] += 1
        if not _fits(masks, col, Ud[depth], W):
            continue
        nodes += 1
        if nodes > node_limit:
            status = 1; break
        empty = True; cnt = 0
        for w in range(W):
            U1[w] = Ud[depth, w] & ~masks[col, w]
            if U1[w] != 0: empty = False
            cnt += _popcount(U1[w])
        P1 = Pd[depth] + cost[col]
        S1 = Sd[depth] - pisum[col]
        O1 = Od[depth] or odd[col]
        path[depth] = col
        A = P1 + absA0
        err = g2 * A
        if empty:
            if P1 - err < best:
                if nc >= cand_cap:
                    status = 2; break
                for t in range(depth + 1):
                    cands[nc, t] = path[t]
                nc += 1
                if P1 + err < best:
                    best = P1 + err
            continue
        lb = P1 + S1 + (cnt / lo) * m
        if (n % 2 == 1) and not O1:
            lb += tau
        if lb - err >= best:
            continue
        bestc = 1 << 30; gb = -1
        for w in range(W):
            x = U1[w]
            while x:
                low = x & -x
                ib = 0
                y = low
                while y > 1:
                    y >>= 1; ib += 1
                x ^= low
                i = w * B + ib
                c = 0
                for p in range(gptr[i], gptr[i + 1]):
                    if _fits(masks, gcols[p], U1, W):
                        c += 1
                        if c >= bestc: break
                if c < bestc:
                    bestc = c; gb = i
                    if c == 0: break
            if bestc == 0: break
        if bestc == 0:
            continue
        depth += 1
        for w in range(W): Ud[depth, w] = U1[w]
        Pd[depth] = P1; Sd[depth] = S1; Od[depth] = O1
        Gd[depth] = gb; Kd[depth] = gptr[gb]
    return status, nodes, best, cands[:nc]


def _prepare(n, lo, cand, exact_cost, pi, m, tau, OPT, bits):
    """The arrays of the search: column masks (one word, or several words of `bits` markets when bits is given or
    n > 62), costs, dual sums, parity, the per-market column lists in cost order, the margin and the incumbent's
    upper value."""
    C = len(cand)
    wide = n > 62 or bits is not None
    B = 63 if bits is None else int(bits)
    if wide and not 1 <= B <= 63:
        raise ValueError(f"bits must be 1 to 63, not {bits}")
    W = -(-n // B) if wide else 1
    masks = np.zeros((C, W) if wide else C, np.int64); cost = np.zeros(C); pisum = np.zeros(C); odd = np.zeros(C, np.bool_)
    for j, c in enumerate(cand):
        if wide:
            for g in c: masks[j, g // B] |= np.int64(1) << (g % B)
        else:
            mk = 0
            for g in c: mk |= 1 << g
            masks[j] = mk
        cost[j] = float(exact_cost[c]); pisum[j] = sum(float(pi[g]) for g in c); odd[j] = len(c) % 2 == 1
    lists = [[] for _ in range(n)]
    for j, c in enumerate(cand):
        for g in c: lists[g].append(j)
    for g in range(n): lists[g].sort(key=lambda j: cost[j])
    gptr = np.zeros(n + 1, np.int64); gptr[1:] = np.cumsum([len(l) for l in lists])
    gcols = np.array([j for l in lists for j in l], np.int64)
    pi = np.asarray(pi, float)
    absA0 = float(np.sum(np.abs(pi))) + (n / lo) * abs(float(m)) + abs(tau)
    # margin: rounding of the sums (gamma_{2n+8}), doubled, plus the conversion of exact inputs to doubles
    g2 = 2.0 * gamma(2 * n + 8) + 4 * U
    of = float(OPT); opt_up = of if Fraction(of) >= OPT else math.nextafter(of, math.inf)
    opt_up = opt_up + g2 * (opt_up + absA0)                 # the incumbent's value carries the same margin
    S0 = 0.0
    for i in range(n): S0 += pi[i]
    return dict(wide=wide, B=B, W=W, masks=masks, cost=cost, pisum=pisum, odd=odd, gptr=gptr, gcols=gcols, pi=pi,
                absA0=absA0, g2=g2, opt_up=opt_up, S0=S0,
                args=(n, B, W, float(lo), masks, cost, pisum, odd, gptr, gcols, float(m), float(tau), g2, absA0))


def _best_of(cand, exact_cost, OPT, rows):
    """The exact cost of each recorded cover; the cheapest below OPT, or (OPT, None)."""
    best_val, best_design = OPT, None
    for row in rows:
        design = tuple(cand[j] for j in row if j >= 0)
        val = sum(exact_cost[c] for c in design)
        if val < best_val:
            best_val, best_design = val, design
    return best_val, best_design


def verify(n, lo, cand, exact_cost, pi, m, tau, OPT, node_limit=2_000_000_000, cand_cap=200_000, bits=None,
           workers=None, split="auto"):
    """Exact optimum over exact covers by the columns `cand` (tuples of geos).
    exact_cost: dict tuple -> Fraction. pi: doubles. m <= 0, OPT: Fractions; tau: double.
    Returns (status, nodes, exact optimum, design). Up to 62 markets a column is one 64-bit mask; above that, or
    when `bits` is given (the tests force small words), it is ceil(n / bits) words of `bits` <= 63 markets.

    workers: None searches on this thread; an integer k >= 1 cuts the tree into at least `split` subtrees ("auto":
    64 k) and searches them on k threads (:func:`_parallel`). node_limit then bounds the total over the threads,
    checked between subtrees and inside each one against what is left."""
    if workers is not None:
        if int(workers) < 1:
            raise ValueError(f"workers must be a positive integer or None, not {workers}")
        if split != "auto" and int(split) < 1:
            raise ValueError(f"split must be a positive integer or 'auto', not {split}")
        return _parallel(n, lo, cand, exact_cost, pi, m, tau, OPT, node_limit, cand_cap, bits, int(workers),
                         64 * int(workers) if split == "auto" else int(split))
    p = _prepare(n, lo, cand, exact_cost, pi, m, tau, OPT, bits)
    masks, cost, pisum, odd, gptr, gcols = p["masks"], p["cost"], p["pisum"], p["odd"], p["gptr"], p["gcols"]
    if p["wide"]:
        status, nodes, best, cands = _search_w(n, p["B"], p["W"], float(lo), masks, cost, pisum, odd, gptr, gcols,
                                               p["pi"], float(m), float(tau), p["opt_up"], p["g2"], p["absA0"],
                                               node_limit, cand_cap, n // 2 + 2)
    else:
        status, nodes, best, cands = _search(n, float(lo), masks, cost, pisum, odd, gptr, gcols, p["pi"],
                                             float(m), float(tau), p["opt_up"], p["g2"], p["absA0"], node_limit,
                                             cand_cap, n // 2 + 2)
    best_val, best_design = _best_of(cand, exact_cost, OPT, cands)
    return status, int(nodes), best_val, best_design


# --- the search on several threads ---------------------------------------------------------------------------------
#
# The tree is cut at the top into subtrees (tasks), each a node of the serial tree given by its uncovered set, partial
# cost, dual sum, parity and the columns on its path. Threads take tasks in order of their node bound and search them
# with _search_from, which releases the GIL. The incumbent's upper value is one shared double: every task prunes on
# it and writes to it each upper value it finds below it. Two threads can race on the write and leave the larger of
# two values; every value written is the upper value of a recorded cover, so the shared value is always an upper
# bound on the exact cost of the incumbent or of a recorded cover, and pruning on it removes no cover cheaper than
# every recorded one. A race costs pruning, never a cover. The tree split is the parallel depth-first search of
# Rao and Kumar (1987), "Parallel depth first search. Part I", Int. J. Parallel Programming 16(6), with static
# splitting at the top and a shared bound as in parallel branch and bound (Gendron and Crainic 1994, Oper. Res.
# 42(6), section 3).

@njit(cache=True, nogil=True)
def _branch_geo(B, W, masks, gptr, gcols, U1):
    """The uncovered market with the fewest fitting columns (the first in market order on ties), and that count;
    count 0 means a dead end. The rule of _search_w's inner loop."""
    bestc = 1 << 30; gb = -1
    for w in range(W):
        x = U1[w]
        while x:
            low = x & -x
            ib = 0
            y = low
            while y > 1:
                y >>= 1; ib += 1
            x ^= low
            i = w * B + ib
            c = 0
            for p in range(gptr[i], gptr[i + 1]):
                if _fits(masks, gcols[p], U1, W):
                    c += 1
                    if c >= bestc: break
            if c < bestc:
                bestc = c; gb = i
                if c == 0: break
        if bestc == 0: break
    return bestc, gb


@njit(cache=True, nogil=True)
def _children(n, B, W, lo, masks, cost, pisum, odd, gptr, gcols, m, tau, g2, absA0, best, U, P, S, O):
    """The children of one node that survive the prune against `best`: their columns, uncovered sets, partial
    costs, dual sums, parity, node bounds, and whether each is a complete cover (recorded when within the margin).
    Also the number of children visited, counted as in _search_w."""
    bestc, g = _branch_geo(B, W, masks, gptr, gcols, U)
    k = 0 if bestc == 0 else gptr[g + 1] - gptr[g]
    cols = np.empty(k, np.int64); Us = np.empty((k, W), np.int64); Ps = np.empty(k); Ss = np.empty(k)
    Os = np.empty(k, np.bool_); lbs = np.empty(k); leaf = np.empty(k, np.bool_)
    nodes = 0; c = 0
    U1 = np.empty(W, np.int64)
    for q in range(k):
        col = gcols[gptr[g] + q]
        if not _fits(masks, col, U, W):
            continue
        nodes += 1
        empty = True; cnt = 0
        for w in range(W):
            U1[w] = U[w] & ~masks[col, w]
            if U1[w] != 0: empty = False
            cnt += _popcount(U1[w])
        P1 = P + cost[col]; S1 = S - pisum[col]; O1 = O or odd[col]
        err = g2 * (P1 + absA0)
        if empty:
            if not P1 - err < best:
                continue
            lb = P1
        else:
            lb = P1 + S1 + (cnt / lo) * m
            if (n % 2 == 1) and not O1:
                lb += tau
            if lb - err >= best:
                continue
        cols[c] = col; Ps[c] = P1; Ss[c] = S1; Os[c] = O1; lbs[c] = lb; leaf[c] = empty
        for w in range(W): Us[c, w] = U1[w]
        c += 1
    return nodes, cols[:c], Us[:c], Ps[:c], Ss[:c], Os[:c], lbs[:c], leaf[:c]


@njit(cache=True, nogil=True)
def _search_from(n, B, W, lo, masks, cost, pisum, odd, gptr, gcols, m, tau, g2, absA0, node_limit, counts, slot,
                 U0, P0, S0, O0, prefix, shared, cands, nc):
    """_search_w on the subtree below one node (uncovered set U0, partial cost P0, dual sum S0, parity O0, the
    columns `prefix` on its path), pruning on and lowering the shared incumbent shared[0]. A recorded cover is the
    prefix and the path, written to row nc of `cands` (the rest of the row -1).

    Node limit: counts[r, 0] holds thread r's nodes so far and counts[r, 1] its stop flag (rows a cache line apart,
    so the threads do not share one). Every 1,024 nodes this thread writes its count and stops when the sum over the
    threads passes node_limit (status 1) or another thread has stopped (status 3); it sets its flag when it stops
    for either limit. Each thread writes only its own row, so no update is lost and the total passes the limit by
    at most 1,024 nodes a thread. Returns (status, nodes, nc)."""
    base = counts[slot, 0]
    d0 = prefix.shape[0]
    maxd = n // 2 + 2
    plen = cands.shape[1]
    Ud = np.zeros((maxd, W), np.int64); Pd = np.zeros(maxd); Sd = np.zeros(maxd); Od = np.zeros(maxd, np.bool_)
    Gd = np.zeros(maxd, np.int64); Kd = np.zeros(maxd, np.int64); path = np.full(plen, -1, np.int64)
    for t in range(d0): path[t] = prefix[t]
    nodes = 0; status = 0
    for w in range(W): Ud[0, w] = U0[w]
    Pd[0] = P0; Sd[0] = S0; Od[0] = O0
    bestc, g0 = _branch_geo(B, W, masks, gptr, gcols, U0)
    if bestc == 0:
        return status, nodes, nc
    Gd[0] = g0; Kd[0] = gptr[g0]
    depth = 0
    U1 = np.zeros(W, np.int64)
    while depth >= 0:
        g = Gd[depth]
        if Kd[depth] >= gptr[g + 1]:
            depth -= 1
            continue
        col = gcols[Kd[depth]]; Kd[depth] += 1
        if not _fits(masks, col, Ud[depth], W):
            continue
        nodes += 1
        if nodes & 1023 == 0:
            counts[slot, 0] = base + nodes
            tot = 0; stop = False
            for r in range(counts.shape[0]):
                tot += counts[r, 0]
                if counts[r, 1] != 0: stop = True
            if tot > node_limit:
                status = 1; break
            if stop:
                status = 3; break
        empty = True; cnt = 0
        for w in range(W):
            U1[w] = Ud[depth, w] & ~masks[col, w]
            if U1[w] != 0: empty = False
            cnt += _popcount(U1[w])
        P1 = Pd[depth] + cost[col]
        S1 = Sd[depth] - pisum[col]
        O1 = Od[depth] or odd[col]
        path[d0 + depth] = col
        err = g2 * (P1 + absA0)
        if empty:
            if P1 - err < shared[0]:
                if nc >= cands.shape[0]:
                    status = 2; break
                for t in range(plen):
                    cands[nc, t] = path[t] if t <= d0 + depth else -1
                nc += 1
                if P1 + err < shared[0]:
                    shared[0] = P1 + err
            continue
        lb = P1 + S1 + (cnt / lo) * m
        if (n % 2 == 1) and not O1:
            lb += tau
        if lb - err >= shared[0]:
            continue
        bestc, gb = _branch_geo(B, W, masks, gptr, gcols, U1)
        if bestc == 0:
            continue
        depth += 1
        for w in range(W): Ud[depth, w] = U1[w]
        Pd[depth] = P1; Sd[depth] = S1; Od[depth] = O1
        Gd[depth] = gb; Kd[depth] = gptr[gb]
    counts[slot, 0] = base + nodes
    if status == 1 or status == 2:
        counts[slot, 1] = 1
    return status, nodes, nc


def _parallel(n, lo, cand, exact_cost, pi, m, tau, OPT, node_limit, cand_cap, bits, workers, split):
    """verify on `workers` threads (:func:`_parallel_core`), its covers re-costed exactly."""
    p = _prepare(n, lo, cand, exact_cost, pi, m, tau, OPT, 63 if bits is None else bits)
    status, total, rows = _parallel_core(p, n, node_limit, cand_cap, workers, split)
    return status, total, *_best_of(cand, exact_cost, OPT, rows)


def _parallel_core(p, n, node_limit, cand_cap, workers, split):
    """The search on `workers` threads over the prepared arrays `p` (several words of B bits). The frontier starts at
    the root and expands its oldest node until it holds at least `split` nodes or none is left to expand; complete
    covers met on the way are recorded as in the search. The frontier is then searched in increasing order of node
    bound. Returns (status, nodes, the recorded covers as sequences of column indices, -1 entries to be skipped)."""
    import collections, threading
    args = p["args"]; W = p["W"]; g2 = p["g2"]; absA0 = p["absA0"]
    plen = n // 2 + 2
    shared = np.array([p["opt_up"]])
    U0 = np.zeros(W, np.int64)
    for w in range(W):
        U0[w] = (1 << min(p["B"], n - w * p["B"])) - 1
    queue = collections.deque([(U0, 0.0, p["S0"], False, (), -math.inf)])
    leaves, nodes = [], 0
    while queue and len(queue) < split:
        U, P, S, O, pre, lb = queue.popleft()
        if lb - g2 * (P + absA0) >= shared[0]:            # pruned by an incumbent found since it was made
            continue
        k, cols, Us, Ps, Ss, Os, lbs, leaf = _children(*args, shared[0], U, P, S, O)
        nodes += k
        if nodes > node_limit:
            return 1, nodes, leaves
        for q in range(len(cols)):
            path = pre + (int(cols[q]),)
            if leaf[q]:
                if len(leaves) >= cand_cap:
                    return 2, nodes, leaves
                leaves.append(path)
                err = g2 * (Ps[q] + absA0)
                if Ps[q] + err < shared[0]:
                    shared[0] = Ps[q] + err
            else:
                queue.append((Us[q].copy(), float(Ps[q]), float(Ss[q]), bool(Os[q]), path, float(lbs[q])))
    tasks = sorted(queue, key=lambda t: t[5])
    nxt = iter(range(len(tasks))); lock = threading.Lock()
    status = [0]; bufs = [None] * workers
    counts = np.zeros((workers, 8), np.int64)        # row r: thread r's nodes, its stop flag; 64 bytes a row
    left = node_limit - nodes                         # the frontier's nodes count against the limit

    def work(r):
        buf = np.empty((cand_cap, plen), np.int64); nc = 0
        while True:
            with lock:
                if status[0] or counts[:, 0].sum() > left or counts[:, 1].any():
                    break
                i = next(nxt, None)
            if i is None:
                break
            U, P, S, O, pre, lb = tasks[i]
            st, k, nc = _search_from(*args, left, counts, r, U, P, S, O, np.array(pre, np.int64), shared, buf, nc)
            if st in (1, 2):
                with lock:
                    if not status[0]:
                        status[0] = st
        bufs[r] = buf[:nc]

    threads = [threading.Thread(target=work, args=(r,)) for r in range(min(workers, max(1, len(tasks))))]
    for t in threads: t.start()
    for t in threads: t.join()
    rows = list(leaves) + [r for b in bufs if b is not None for r in b]
    total = nodes + int(counts[:, 0].sum())
    if not status[0] and total > node_limit and next(nxt, None) is not None:
        status[0] = 1                                 # stopped between subtrees with subtrees left
    return status[0], total, rows


@njit(cache=True)
def _row_masks(rows, n, B, W, wide):
    C = rows.shape[0]
    m1 = np.zeros(C, np.int64); mw = np.zeros((C, W), np.int64)
    for j in range(C):
        for q in range(4):
            g = rows[j, q]
            if g < 0:
                continue
            if wide:
                mw[j, g // B] |= np.int64(1) << (g % B)
            else:
                m1[j] |= np.int64(1) << g
    return m1, mw


def verify_rows(n, lo, rows, cost, pi, m, tau, opt, node_limit=2_000_000_000, cand_cap=200_000, workers=None,
                split="auto"):
    """The search of :func:`verify` on columns given as an index array (rows padded with -1) and double costs,
    set up with numpy: the capped stages of certify_exact.certify_staged. The margin assumes doubles within the
    rounding of exact costs; a stage on computed costs finds incumbents and proves nothing, and its covers are
    re-costed exactly by the caller. workers and split as in :func:`verify` (at 120 markets the first capped stage,
    563,453 rows and 300,000 nodes, took 21.0 s on this thread and 6.1 s on four). Returns (status, nodes, the
    recorded covers as arrays of row indices)."""
    if workers is not None:
        if int(workers) < 1:
            raise ValueError(f"workers must be a positive integer or None, not {workers}")
        if split != "auto" and int(split) < 1:
            raise ValueError(f"split must be a positive integer or 'auto', not {split}")
    rows = np.ascontiguousarray(rows, np.int64); C = rows.shape[0]
    cost = np.ascontiguousarray(cost, float); pi = np.asarray(pi, float)
    wide = n > 62 or workers is not None; B = 63; W = -(-n // B) if wide else 1
    m1, mw = _row_masks(rows, n, B, W, wide)
    k = (rows >= 0).sum(1)
    pisum = np.where(rows >= 0, pi[np.maximum(rows, 0)], 0.0).sum(1); odd = (k % 2 == 1)
    mk = rows.ravel(); col = np.repeat(np.arange(C), 4); keep = mk >= 0; mk = mk[keep]; col = col[keep]
    order = np.lexsort((cost[col], mk))                # by market, then by cost
    gcols = col[order].astype(np.int64)
    gptr = np.zeros(n + 1, np.int64); gptr[1:] = np.cumsum(np.bincount(mk, minlength=n))
    absA0 = float(np.sum(np.abs(pi))) + (n / lo) * abs(float(m)) + abs(tau)
    g2 = 2.0 * gamma(2 * n + 8) + 4 * U
    opt_up = math.nextafter(float(opt), math.inf); opt_up = opt_up + g2 * (opt_up + absA0)
    if workers is not None:
        S0 = 0.0
        for i in range(n): S0 += pi[i]
        p = dict(B=B, W=W, g2=g2, absA0=absA0, opt_up=opt_up, S0=S0,
                 args=(n, B, W, float(lo), mw, cost, pisum, odd, gptr, gcols, float(m), float(tau), g2, absA0))
        status, nodes, found = _parallel_core(p, n, node_limit, cand_cap, int(workers),
                                              64 * int(workers) if split == "auto" else int(split))
        return status, int(nodes), [np.array([j for j in r if j >= 0], np.int64) for r in found]
    args = (cost, pisum, odd, gptr, gcols, pi, float(m), float(tau), opt_up, g2, absA0, node_limit, cand_cap, n // 2 + 2)
    if wide:
        status, nodes, best, cands = _search_w(n, B, W, float(lo), mw, *args)
    else:
        status, nodes, best, cands = _search(n, float(lo), m1, *args)
    return status, int(nodes), [r[r >= 0] for r in cands]
