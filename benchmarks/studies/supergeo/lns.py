"""Exact-repair large-neighbourhood search for the supergeo base problem.

Destroy: free m groups of the incumbent (the worst groups plus groups whose
geos sit near theirs in sorted order). Repair: enumerate every subset of size
lo..hi over the freed geos and solve that set partitioning MIP exactly.
Accept on strict improvement."""
import time
import numpy as np
from . import sg


def group_loss(z, g):
    return float(sg._score(z, np.array(tuple(g) + (0,) * (4 - len(g)), np.int64), len(g)))


def repair(z, free, lo, hi, cur):
    zf = z[free]
    rows, s_, rc, m = sg._scan(zf, np.zeros(len(free)), lo, hi, np.inf, 2_000_000)
    s = s_[:m].astype(float)
    k = max(cur, 1e-300) / len(free)                  # costs O(1) for HiGHS tolerances
    r = sg.master_mip([tuple(int(v) for v in rows[t] if v >= 0) for t in range(m)], s / k, len(free), 120)
    if r.x is None:
        return None, np.inf
    pick = np.flatnonzero(r.x > 0.5)
    groups = [tuple(int(free[v]) for v in rows[t] if v >= 0) for t in pick]
    return groups, float(s[pick].sum())


def lns(z, design, lo=2, hi=4, m=8, seconds=600, seed=0, verbose=True, log_every=30):
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    rank = np.empty(len(z), int); rank[np.argsort(z)] = np.arange(len(z))
    design = [tuple(g) for g in design]
    loss = [group_loss(z, g) for g in design]
    t0 = time.perf_counter(); last = t0; trace = [(0.0, sum(loss))]
    it = acc = 0
    while time.perf_counter() - t0 < seconds:
        it += 1
        # seed group: the worst with prob 1/2, else a random one; then its rank neighbours
        a = int(np.argmax(loss)) if rng.random() < 0.5 else int(rng.integers(len(design)))
        centre = np.mean([rank[g] for g in design[a]])
        width = rng.integers(5, 40)
        near = [i for i, g in enumerate(design) if abs(np.mean([rank[v] for v in g]) - centre) <= width]
        rng.shuffle(near)
        pick = [a] + [i for i in near if i != a][: m - 1]
        if len(pick) < m:
            pick += [int(i) for i in rng.choice([i for i in range(len(design)) if i not in pick], m - len(pick), replace=False)]
        free = np.array(sorted(v for i in pick for v in design[i]), dtype=np.int64)
        cur = sum(loss[i] for i in pick)
        groups, new = repair(z, free, lo, hi, cur)
        if groups is not None and new < cur * (1 - 1e-9):
            acc += 1
            keep = [i for i in range(len(design)) if i not in set(pick)]
            design = [design[i] for i in keep] + groups
            loss = [loss[i] for i in keep] + [group_loss(z, g) for g in groups]
        now = time.perf_counter()
        trace.append((now - t0, sum(loss)))
        if verbose and now - last > log_every:
            print(f"    [{now - t0:.0f}s] iter {it} accepted {acc} loss {sum(loss):.6g}", flush=True)
            last = now
    return design, sum(loss), trace


from numba import njit


@njit(cache=True)
def _dp(zf, lo, hi):
    """Exact set partition of the k = len(zf) geos into blocks of size lo..hi,
    minimising the sum of block scores, by DP over bitmasks. Returns (best, choice)."""
    k = zf.shape[0]
    full = (1 << k) - 1
    bs = np.full(1 << k, np.inf)
    idx = np.empty(4, np.int64)
    for a in range(k):
        for b in range(a + 1, k):
            idx[0] = a; idx[1] = b
            if lo <= 2: bs[(1 << a) | (1 << b)] = sg._score(zf, idx, 2)
            if hi < 3: continue
            for c in range(b + 1, k):
                idx[2] = c
                if lo <= 3: bs[(1 << a) | (1 << b) | (1 << c)] = sg._score(zf, idx, 3)
                if hi < 4: continue
                for d in range(c + 1, k):
                    idx[3] = d
                    bs[(1 << a) | (1 << b) | (1 << c) | (1 << d)] = sg._score(zf, idx, 4)
    f = np.full(1 << k, np.inf); ch = np.zeros(1 << k, np.int64)
    f[0] = 0.0
    bits = np.empty(k, np.int64)
    for mask in range(1, full + 1):
        low = mask & (-mask)
        nb = 0
        rest = mask ^ low
        while rest:
            bt = rest & (-rest); bits[nb] = bt; nb += 1; rest ^= bt
        best = np.inf; arg = 0
        for p in range(nb):
            B = low | bits[p]
            if bs[B] < np.inf and f[mask ^ B] + bs[B] < best:
                best = f[mask ^ B] + bs[B]; arg = B
            for q in range(p + 1, nb):
                B2 = B | bits[q]
                if bs[B2] < np.inf and f[mask ^ B2] + bs[B2] < best:
                    best = f[mask ^ B2] + bs[B2]; arg = B2
                for r in range(q + 1, nb):
                    B3 = B2 | bits[r]
                    if bs[B3] < np.inf and f[mask ^ B3] + bs[B3] < best:
                        best = f[mask ^ B3] + bs[B3]; arg = B3
        f[mask] = best; ch[mask] = arg
    return f[full], ch


def repair_dp(z, free, lo, hi):
    best, ch = _dp(z[free], lo, hi)
    groups, mask = [], (1 << len(free)) - 1
    while mask:
        B = int(ch[mask])
        groups.append(tuple(int(free[i]) for i in range(len(free)) if B >> i & 1))
        mask ^= B
    return groups, float(best)


def lns_dp(z, design, lo=2, hi=4, max_free=18, seconds=600, seed=0, verbose=True, log_every=30, steps=None):
    """Large-neighbourhood search from `design`: each step frees the groups near a chosen one (at most max_free
    markets) and re-partitions them exactly (repair_dp), keeping the result when it lowers their loss.

    steps: the number of steps. When given, the search runs exactly that many and never reads the clock, so the
    same inputs and seed give the same design on any machine at any load. When None, it stops after `seconds`,
    and the design then depends on how many steps the machine completed in that time. Returns (design, loss,
    trace of (seconds elapsed, loss) after each step)."""
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0):
        raise ValueError(f"steps must be a non-negative integer or None, not {steps!r}")
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    rank = np.empty(len(z), int); rank[np.argsort(z)] = np.arange(len(z))
    design = [tuple(g) for g in design]
    loss = [group_loss(z, g) for g in design]
    t0 = time.perf_counter(); last = t0; trace = [(0.0, sum(loss))]
    it = acc = 0
    while (it < steps) if steps is not None else (time.perf_counter() - t0 < seconds):
        it += 1
        a = int(np.argmax(loss)) if rng.random() < 0.3 else int(rng.integers(len(design)))
        centre = np.mean([rank[g] for g in design[a]])
        order = np.argsort([abs(np.mean([rank[v] for v in g]) - centre) + rng.exponential(rng.integers(2, 30)) for g in design])
        pick, size = [], 0
        for i in order:
            if size + len(design[i]) <= max_free:
                pick.append(int(i)); size += len(design[i])
        free = np.array(sorted(v for i in pick for v in design[i]), dtype=np.int64)
        cur = sum(loss[i] for i in pick)
        groups, new = repair_dp(z, free, lo, hi)
        if new < cur * (1 - 1e-9):
            acc += 1
            ps = set(pick)
            keep = [i for i in range(len(design)) if i not in ps]
            design = [design[i] for i in keep] + groups
            loss = [loss[i] for i in keep] + [group_loss(z, g) for g in groups]
        now = time.perf_counter()
        trace.append((now - t0, sum(loss)))
        if verbose and now - last > log_every:
            print(f"    [{now - t0:.0f}s] iter {it} accepted {acc} loss {sum(loss):.6g}", flush=True)
            last = now
    return design, sum(loss), trace
