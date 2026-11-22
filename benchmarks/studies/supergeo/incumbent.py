"""Starting designs for the supergeo search.

greedy. Largest first: the largest unassigned market gets the group of least
loss among the unassigned markets, of a size that leaves no single market
behind; repeat. The idea of Karmarkar and Karp's differencing method, placing
the largest number while the most freedom remains to balance it, restricted to
groups of 2 to 4.

grasp. Randomised greedy (each step draws among the market's few cheapest
groups, weighted towards the cheapest), repeated, best kept, then exact-repair
local search (:func:`.lns.lns_dp`) on the best for the rest of the budget. The
deterministic greedy is the first construction, so the result is never worse.

lp_round. A MIP over the given columns (for example the root LP's), warm-started
from a design, with a time limit: the restricted master with a good start. Never
worse than the start."""
import time
import numpy as np
from numba import njit
from . import sg, dive, lns, dsr


@njit(cache=True)
def _top_groups(z, alive, i, sizes, top):
    """The `top` cheapest groups holding market i among alive markets, of the allowed sizes."""
    n = z.shape[0]
    others = np.empty(n, np.int64); m = 0
    for j in range(n):
        if alive[j] and j != i:
            others[m] = j; m += 1
    best = np.full(top, np.inf); grp = np.full((top, 4), -1, np.int64)
    idx = np.empty(4, np.int64)
    for a in range(m):
        for b in range(a, m + 1):
            for c in range(b, m + 1):
                k = 0
                if b == a and c == a:                     # size 2: (a)
                    k = 2
                elif b > a and c == b and b < m:           # size 3: (a, b)
                    k = 3
                elif b > a and c > b and c < m:            # size 4: (a, b, c)
                    k = 4
                else:
                    continue
                if not sizes[k]:
                    continue
                idx[0] = i; idx[1] = others[a]
                if k >= 3: idx[2] = others[b]
                if k == 4: idx[3] = others[c]
                s = sg._score(z, idx, k)
                if s < best[top - 1]:
                    p = top - 1
                    while p > 0 and best[p - 1] > s:
                        best[p] = best[p - 1]; grp[p] = grp[p - 1]; p -= 1
                    best[p] = s; grp[p, :] = -1; grp[p, :k] = idx[:k]
    return grp, best


def _sizes(r):
    s = np.zeros(5, np.bool_)
    for k in (2, 3, 4):
        if k <= r and r - k != 1:
            s[k] = True
    return s


def _construct(z, rng=None, top=1, temper=1.0):
    z = np.asarray(z, float); n = len(z)
    if n < 2:
        raise ValueError("a design needs at least two markets")
    alive = np.ones(n, np.bool_); design = []
    while alive.any():
        r = int(alive.sum())
        live = np.flatnonzero(alive)
        i = int(live[np.argmax(z[live])])
        grp, los = _top_groups(z, alive, i, _sizes(r), top)
        ok = np.isfinite(los)
        if rng is None or ok.sum() == 1:
            pick = 0
        else:
            w = np.exp(-(np.arange(ok.sum())) / temper); pick = int(rng.choice(ok.sum(), p=w / w.sum()))
        g = tuple(sorted(int(v) for v in grp[pick] if v >= 0))
        design.append(g); alive[list(g)] = False
    return design


def cost(z, design):
    z = np.asarray(z, float)
    return sum(dive._sc(z, tuple(c)) for c in design)


def greedy(z):
    return _construct(z)


def grasp(z, seconds=2.0, seed=0, top=5, repair_share=0.5):
    z = np.asarray(z, float); rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best = greedy(z); bc = cost(z, best)
    while time.perf_counter() - t0 < seconds * (1 - repair_share):
        d = _construct(z, rng, top=top)
        c = cost(z, d)
        if c < bc: best, bc = d, c
    left = seconds - (time.perf_counter() - t0)
    if left > 0 and len(z) >= 4:
        d, c, _ = lns.lns_dp(z, best, seconds=left, seed=seed, verbose=False)
        if c < bc: best, bc = [tuple(sorted(g)) for g in d], c
    return best


def lp_round(z, cols, start, seconds=1.0):
    z = np.asarray(z, float); n = len(z)
    start = [tuple(sorted(int(v) for v in g)) for g in start]
    c0 = cost(z, start)
    pool = list(dict.fromkeys([tuple(c) for c in cols] + start))
    costs = np.array([dive._sc(z, c) for c in pool])
    f = n / c0 if c0 > 0 else 1.0                          # objective near n, away from absolute tolerances
    pos = {c: j for j, c in enumerate(pool)}
    sel, val = dsr.solve_partition(pool, costs * f, n, warm=[pos[g] for g in start], time_limit=seconds)
    if sel is None:
        return start
    d = [pool[j] for j in sel]
    if sorted(g for c in d for g in c) != list(range(n)):
        return start
    return d if cost(z, d) < c0 else start
