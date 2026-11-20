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


def verify(n, lo, cand, exact_cost, pi, m, tau, OPT, node_limit=2_000_000_000, cand_cap=200_000):
    """Exact optimum over exact covers by the columns `cand` (tuples of geos).
    exact_cost: dict tuple -> Fraction. pi: doubles. m <= 0, OPT: Fractions; tau: double.
    Returns (status, nodes, exact optimum, design)."""
    assert n <= 62
    C = len(cand)
    masks = np.zeros(C, np.int64); cost = np.zeros(C); pisum = np.zeros(C); odd = np.zeros(C, np.bool_)
    for j, c in enumerate(cand):
        mk = 0
        for g in c: mk |= 1 << g
        masks[j] = mk; cost[j] = float(exact_cost[c]); pisum[j] = sum(float(pi[g]) for g in c); odd[j] = len(c) % 2 == 1
    lists = [[] for _ in range(n)]
    for j, c in enumerate(cand):
        for g in c: lists[g].append(j)
    for g in range(n): lists[g].sort(key=lambda j: cost[j])
    gptr = np.zeros(n + 1, np.int64); gptr[1:] = np.cumsum([len(l) for l in lists])
    gcols = np.array([j for l in lists for j in l], np.int64)
    absA0 = float(np.sum(np.abs(pi))) + (n / lo) * abs(float(m)) + abs(tau)
    # margin: rounding of the sums (gamma_{2n+8}), doubled, plus the conversion of exact inputs to doubles
    g2 = 2.0 * gamma(2 * n + 8) + 4 * U
    of = float(OPT); opt_up = of if Fraction(of) >= OPT else math.nextafter(of, math.inf)
    opt_up = opt_up + g2 * (opt_up + absA0)                 # the incumbent's value carries the same margin
    status, nodes, best, cands = _search(n, float(lo), masks, cost, pisum, odd, gptr, gcols, np.asarray(pi, float),
                                         float(m), float(tau), opt_up, g2, absA0, node_limit, cand_cap, n // 2 + 2)
    best_val, best_design = OPT, None
    for row in cands:
        design = tuple(cand[j] for j in row if j >= 0)
        val = sum(exact_cost[c] for c in design)
        if val < best_val:
            best_val, best_design = val, design
    return status, int(nodes), best_val, best_design
