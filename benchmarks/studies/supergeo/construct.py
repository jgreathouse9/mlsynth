"""Design construction for large panels: annealing over pairings with a closed-form coupling, then repair of many
groups at once by an exact set-partitioning solve.

A group of four balances when it splits into two pairs with equal differences: a + d = b + c is a - b = c - d. So a
design of fours and pairs is a pairing of the markets plus a coupling of the pairs, and for a fixed pairing the best
coupling has a closed form (:func:`couple`). The construction searches pairings by simulated annealing, scoring each
exactly. Searches that improve a few groups at a time stall: on a simulated 235-market panel whose optimum lies
within about 1e-5 of the bound (a first-moment count, Mertens, "The Easiest Hard Problem: Number Partitioning", 2003,
Sec. V), the study's repair of 18 markets at a time (lns.lns_dp) stopped near 1.8 above it. Near-optimal designs of
number-partitioning type differ from one another in many places at once (Mertens, Sec. V: the low-cost
configurations are nearly independent), so the second stage repairs 10 to 15 groups at a time, 40 to 60 markets,
by an exact set-partitioning solve over every group of them that could be in a cheaper repair (:func:`repair`;
Maher and Ronnberg, "Integer programming column generation: accelerating branch-and-price using a novel pricing
scheme for finding high-quality solutions in set covering, packing, and partitioning problems", Mathematical
Programming Computation 15, 2023, keep every column over the freed rows and enlarge the freed set when a repair
fails). On that panel: annealing 1.86 to 1.93 against the bound 1.78035, and the repair 1.8046 after 30 minutes.
"""
import math

import numpy as np
from numba import njit
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

from . import dive, sg


def couple(d):
    """The best split of pairs, with absolute differences d, into couples (a four of the two pairs, cost
    (d_i - d_j)^2: one pair oriented against the other) and lone pairs (cost d_i^2). Closed form: in sorted order an
    optimal split couples only neighbours. Two couples that cross or nest cost no less than the two adjacent ones
    (the exchange argument for squared differences), and a lone pair a inside a couple b <= c (a <= b) or b inside
    a <= c never beats coupling the two nearer values: (c - b)^2 + a^2 <= (c - a)^2 + b^2 for 0 <= a <= b <= c. So
    a DP over the sorted differences, f(i) = min(f(i-1) + d_i^2, f(i-2) + (d_i - d_{i-1})^2), is optimal. Returns
    (cost, groups as tuples of indices into d)."""
    d = np.abs(np.asarray(d, float)); m = len(d)
    if m == 0:
        return 0.0, []
    o = np.argsort(d, kind="stable"); s = d[o]
    f = np.zeros(m + 1); take = np.zeros(m + 1, np.int64)
    f[1] = s[0] ** 2; take[1] = 1
    for i in range(2, m + 1):
        a = f[i - 1] + s[i - 1] ** 2; b = f[i - 2] + (s[i - 1] - s[i - 2]) ** 2
        f[i], take[i] = (a, 1) if a <= b else (b, 2)
    groups, i = [], m
    while i > 0:
        if take[i] == 1:
            groups.append((int(o[i - 1]),)); i -= 1
        else:
            groups.append((int(o[i - 2]), int(o[i - 1]))); i -= 2
    return float(f[m]), groups[::-1]


@njit(cache=True)
def _score(s, P):
    """:func:`couple`'s cost for the pairing P (rows of two indices into s)."""
    m = P.shape[0]; d = np.empty(m)
    for i in range(m):
        d[i] = abs(s[P[i, 0]] - s[P[i, 1]])
    d.sort()
    f0 = 0.0; f1 = d[0] * d[0]
    for i in range(2, m + 1):
        a = f1 + d[i - 1] * d[i - 1]; b = f0 + (d[i - 1] - d[i - 2]) ** 2
        f0, f1 = f1, min(a, b)
    return f1


@njit(cache=True)
def _anneal(s, P, steps, T0, T1, seed):
    """Simulated annealing over pairings (Kirkpatrick, Gelatt and Vecchi, "Optimization by simulated annealing",
    Science 220, 1983): each move re-pairs two pairs one of the two other ways and is accepted by the Metropolis rule
    at a temperature falling geometrically from T0 to T1. Returns the best pairing seen and its score."""
    np.random.seed(seed); m = P.shape[0]
    cur = _score(s, P); best = cur; bestP = P.copy()
    if m < 2 or steps == 0:
        return best, bestP
    lg = math.log(T1 / T0)
    for t in range(steps):
        T = T0 * math.exp(lg * t / steps)
        p = np.random.randint(m); q = np.random.randint(m - 1)
        if q >= p:
            q += 1
        a, b, c, d = P[p, 0], P[p, 1], P[q, 0], P[q, 1]
        if np.random.random() < 0.5:
            P[p, 1] = c; P[q, 0] = b
        else:
            P[p, 1] = d; P[q, 1] = b
        new = _score(s, P)
        if new <= cur or np.random.random() < math.exp((cur - new) / T):
            cur = new
            if cur < best:
                best = cur; bestP[:, :] = P
        else:
            P[p, 0] = a; P[p, 1] = b; P[q, 0] = c; P[q, 1] = d
    return best, bestP


@njit(cache=True)
def _triple_scan(z):
    best = np.inf; bi = 0; bj = 1; bk = 2; n = z.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                S = z[i] + z[j] + z[k]
                c = min((S - 2 * z[i]) ** 2, (S - 2 * z[j]) ** 2, (S - 2 * z[k]) ** 2)
                if c < best:
                    best = c; bi = i; bj = j; bk = k
    return best, bi, bj, bk


def cheapest_triple(z):
    """The triple of least cost (its best split of one market against two), by a scan of every triple, O(n^3).
    Returns (cost, (i, j, k))."""
    z = np.asarray(z, float)
    c, i, j, k = _triple_scan(z)
    return float(c), (int(i), int(j), int(k))


def _check_int(name, v, least):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < least:
        raise ValueError(f"{name} must be an integer >= {least}, not {v!r}")


def _cost(z, design):
    return sum(dive._sc(z, tuple(sorted(g))) for g in design)


def pairing_design(z, steps=1_000_000, seed=0, T0=None, T1=None):
    """A design of fours and pairs (and, for an odd count, the cheapest triple) by annealing over pairings of the
    markets, each scored by :func:`couple`. The temperature falls from T0 to T1, by default 0.005 g and 1e-12 g with
    g the squared median spacing of the sorted totals (certify_exact._spacing2), from the sorted neighbours'
    pairing. Returns (design, its cost, the start's cost), the design being the cheaper of the start and the annealed
    one by the groups' true cost, which counts a four's one-against-three splits as well."""
    from .certify_exact import _spacing2
    _check_int("steps", steps, 0); _check_int("seed", seed, 0)
    z = np.asarray(z, float); n = len(z)
    if n < 2:
        raise ValueError(f"a design needs at least 2 markets, not {n}")
    if n == 2:
        return [(0, 1)], dive._sc(z, (0, 1)), dive._sc(z, (0, 1))
    if n == 3:
        c = dive._sc(z, (0, 1, 2))
        return [(0, 1, 2)], c, c
    triple = []
    rest = np.arange(n)
    if n % 2:
        _, t = cheapest_triple(z)
        triple = [tuple(sorted(t))]
        rest = np.array([i for i in range(n) if i not in t])
    o = rest[np.argsort(z[rest], kind="stable")]; s = z[o]; m = len(o) // 2
    g = _spacing2(z)
    T0 = 0.005 * g if T0 is None else float(T0); T1 = 1e-12 * g if T1 is None else float(T1)
    P0 = np.arange(len(o), dtype=np.int64).reshape(m, 2)

    def build(P):
        pairs = [(int(o[a]), int(o[b])) for a, b in P]
        _, groups = couple(np.array([z[a] - z[b] for a, b in pairs]))
        out = [tuple(sorted(v for i in grp for v in pairs[i])) for grp in groups] + triple
        return sorted(out)

    start = build(P0); cs = _cost(z, start)
    _, BP = _anneal(s, P0.copy(), int(steps), T0, T1, int(seed))
    design = build(BP); cd = _cost(z, design)
    if cs <= cd:
        return start, cs, cs
    return design, cd, cs


@njit(cache=True)
def _groups_below(z, F, lo, hi, cur):
    """Every group of lo..hi of the markets F with cost below cur, as rows padded with -1, and their costs."""
    m = F.shape[0]; out = []; cost = []
    idx = np.zeros(4, np.int64)
    for a in range(m):
        for b in range(a + 1, m):
            if lo <= 2:
                idx[0] = F[a]; idx[1] = F[b]
                c = sg._score(z, idx, 2)
                if c < cur:
                    out.append((F[a], F[b], -1, -1)); cost.append(c)
            if hi < 3:
                continue
            for c3 in range(b + 1, m):
                if lo <= 3:
                    idx[0] = F[a]; idx[1] = F[b]; idx[2] = F[c3]
                    c = sg._score(z, idx, 3)
                    if c < cur:
                        out.append((F[a], F[b], F[c3], -1)); cost.append(c)
                if hi < 4:
                    continue
                for d in range(c3 + 1, m):
                    idx[0] = F[a]; idx[1] = F[b]; idx[2] = F[c3]; idx[3] = F[d]
                    c = sg._score(z, idx, 4)
                    if c < cur:
                        out.append((F[a], F[b], F[c3], F[d])); cost.append(c)
    return out, cost


def repair(z, design, steps, seed=0, k0=10, k_max=15, patience=5, node_limit=300, lo=2, hi=4, with_k=False):
    """Repair by exact set partitioning over many groups at once. Each step frees k groups, the dearest one drawn
    with weight cost * u^(1/4), u uniform, and k - 1 others uniformly; every group of lo..hi of the freed markets
    whose cost is below the freed groups' total is a column (a dearer one cannot be in a cheaper repair), and HiGHS
    solves the partitioning MIP with a node limit, so the result depends on the inputs and the seed only. A cheaper
    repair replaces the freed groups; after `patience` steps in a row without one, k grows by one up to k_max.
    Returns (design, trace of costs, the start first and one entry per step[, the k of each step])."""
    _check_int("steps", steps, 0); _check_int("seed", seed, 0); _check_int("k0", k0, 1)
    _check_int("k_max", k_max, k0); _check_int("patience", patience, 1); _check_int("node_limit", node_limit, 1)
    z = np.asarray(z, float); n = len(z)
    design = [tuple(sorted(int(v) for v in g)) for g in design]
    if sorted(v for g in design for v in g) != list(range(n)) or any(not lo <= len(g) <= hi for g in design):
        raise ValueError("the design must be a partition of the markets into groups of lo to hi")
    rng = np.random.default_rng(seed)
    gc = [dive._sc(z, g) for g in design]
    trace = [float(sum(gc))]; ks = []; k = k0; fails = 0
    for _ in range(steps):
        kk = min(k, len(design)); ks.append(k)
        w = np.array(gc) * rng.random(len(design)) ** 0.25
        first = int(np.argmax(w))
        others = [i for i in rng.permutation(len(design)).tolist() if i != first][:kk - 1]
        pick = sorted([first] + others)
        cur = sum(gc[i] for i in pick)
        F = np.array(sorted(v for i in pick for v in design[i]), np.int64)
        rows, costs = _groups_below(z, F, lo, hi, cur)
        better = None
        if rows:
            R = np.array(rows, np.int64); C = np.array(costs)
            pos = {int(v): i for i, v in enumerate(F)}
            ri = [pos[int(v)] for r in R for v in r if v >= 0]
            ci = [j for j, r in enumerate(R) for v in r if v >= 0]
            A = csc_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(F), len(R)))
            res = milp(C, constraints=LinearConstraint(A, 1, 1), integrality=np.ones(len(R)), bounds=Bounds(0, 1),
                       options={"node_limit": int(node_limit), "mip_rel_gap": 0})
            if res.x is not None:
                new = [tuple(int(v) for v in R[j] if v >= 0) for j in np.where(res.x > 0.5)[0]]
                nc = [dive._sc(z, g) for g in new]
                if sum(nc) < cur:
                    better = (new, nc)
        if better is not None:
            keep = [i for i in range(len(design)) if i not in set(pick)]
            design = [design[i] for i in keep] + better[0]; gc = [gc[i] for i in keep] + better[1]
            fails = 0
        else:
            fails += 1
            if fails >= patience and k < k_max:
                k += 1; fails = 0
        trace.append(float(sum(gc)))
    design = sorted(design)
    if with_k:
        return design, trace, ks
    return design, trace


def construct(z, anneal_steps=1_000_000, repair_steps=100, seed=0, **repair_kw):
    """:func:`pairing_design`, then :func:`repair` from it. Returns (design, cost)."""
    design, cost, _ = pairing_design(z, steps=anneal_steps, seed=seed)
    if repair_steps:
        design, trace = repair(z, design, steps=repair_steps, seed=seed, **repair_kw)
        cost = trace[-1]
    return design, cost
