"""Meet-in-the-middle pricing for the supergeo master.

A group G of 2..4 geos scores min over non-trivial signings s of (s . z_G)^2.
Every signed sum over at most four geos is v_L + v_R for two halves of one or
two geos each. With every signed half listed once and sorted by value, the
groups that can have reduced cost <= theta are found by, for each left half L,
a window search over right halves R with

    (v_L + v_R)^2 <= theta + pi_L + pi_R <= theta + pi_L + piR_max,

so |v_R + v_L| <= sqrt(theta + pi_L + piR_max). Each group found is then scored
exactly. Canonical form: L holds the group's smallest index with sign +, and R
holds indices above it. A group can still arise from several (L, R) and
signings; duplicates are removed before scoring. Cost O(N^2 log N + window
output) against the full scan's O(N^4)."""
import numpy as np
from numba import njit
from . import sg


@njit(cache=True)
def _halves(z, pi):
    n = z.shape[0]
    m = 2 * n + 4 * (n * (n - 1) // 2)
    hv = np.empty(m); ha = np.empty(m, np.int64); hb = np.empty(m, np.int64)
    hsa = np.empty(m, np.int64); hsb = np.empty(m, np.int64); hw = np.empty(m)
    t = 0
    for i in range(n):
        for s in (1, -1):
            hv[t] = s * z[i]; ha[t] = i; hb[t] = -1; hsa[t] = s; hsb[t] = 0; hw[t] = pi[i]; t += 1
    for i in range(n):
        for j in range(i + 1, n):
            for si in (1, -1):
                for sj in (1, -1):
                    hv[t] = si * z[i] + sj * z[j]; ha[t] = i; hb[t] = j
                    hsa[t] = si; hsb[t] = sj; hw[t] = pi[i] + pi[j]; t += 1
    return hv, ha, hb, hsa, hsb, hw


@njit(cache=True)
def _window(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, cap):
    """Candidate groups (as sorted 4-tuples, -1 padded) from every left half."""
    m = hv.shape[0]
    sv = hv[order]
    out = np.full((cap, 4), -1, np.int64); k = 0
    for L in range(m):
        if hsa[L] != 1:                 # left half's first (smallest) geo signed +
            continue
        a, b = ha[L], hb[L]
        nl = 1 if b < 0 else 2
        rad2 = theta + hw[L] + wmax
        if rad2 < 0:
            continue
        rad = np.sqrt(rad2)
        target = -hv[L]
        start = np.searchsorted(sv, target - rad)
        for q in range(start, m):
            R = order[q]
            if sv[q] > target + rad:
                break
            c, d = ha[R], hb[R]
            nr = 1 if d < 0 else 2
            size = nl + nr
            if size < lo or size > hi:
                continue
            if c <= a or (d >= 0 and d <= a):              # R above L's smallest geo
                continue
            if c == b or d == b and d >= 0:                # disjoint
                continue
            if (b >= 0 and hsb[L] == -1) or hsa[R] == -1 or (d >= 0 and hsb[R] == -1):
                pass
            else:
                continue                                   # all signs + : not a split
            v = hv[L] + hv[R]
            if v * v > theta + hw[L] + hw[R]:
                continue
            g = np.empty(4, np.int64); g[:] = -1
            g[0] = a; p = 1
            if b >= 0: g[p] = b; p += 1
            g[p] = c; p += 1
            if d >= 0: g[p] = d; p += 1
            g[:p] = np.sort(g[:p])
            if k < cap:
                out[k] = g
            k += 1
    return out[:min(k, cap)], k


def price(z, pi, lo, hi, theta, cap=20_000_000):
    """Every group of size lo..hi with reduced cost <= theta: (rows, scores, rc)."""
    z = np.asarray(z, float); pi = np.asarray(pi, float)
    hv, ha, hb, hsa, hsb, hw = _halves(z, pi)
    order = np.argsort(hv, kind="stable")
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))   # >= any right half's pi
    _, k = _window(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, 0)        # count
    if k > cap:
        raise MemoryError(f"{k} candidates exceed cap {cap}")
    rows, k = _window(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, k)
    if k > cap:
        raise MemoryError(f"{k} candidates exceed cap {cap}")
    rows = np.unique(rows, axis=0)
    sizes = (rows >= 0).sum(1)
    sc = np.array([sg._score(z, r, s) for r, s in zip(rows, sizes)])
    rc = sc - np.array([pi[r[:s]].sum() for r, s in zip(rows, sizes)])
    keep = rc <= theta
    return rows[keep], sc[keep], rc[keep], k
