"""Parallel (numba prange) versions of the meet-in-the-middle window and the full
scan: count per outer index, prefix sums, then fill at fixed offsets, so the
output is the serial output in the same order."""
import numpy as np
from numba import njit, prange
from . import sg, mitm


@njit(cache=True)
def _one_left(L, hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, out, base, fill):
    m = hv.shape[0]
    if hsa[L] != 1:
        return 0
    a, b = ha[L], hb[L]
    nl = 1 if b < 0 else 2
    rad2 = theta + hw[L] + wmax
    if rad2 < 0:
        return 0
    rad = np.sqrt(rad2); target = -hv[L]
    start = np.searchsorted(sv, target - rad)
    k = 0
    for q in range(start, m):
        if sv[q] > target + rad:
            break
        R = order[q]
        c, d = ha[R], hb[R]
        nr = 1 if d < 0 else 2
        size = nl + nr
        if size < lo or size > hi: continue
        if c <= a or (d >= 0 and d <= a): continue
        if c == b or (d >= 0 and d == b): continue
        if not ((b >= 0 and hsb[L] == -1) or hsa[R] == -1 or (d >= 0 and hsb[R] == -1)): continue
        v = hv[L] + hv[R]
        if v * v > theta + hw[L] + hw[R]: continue
        if fill:
            p = 0
            out[base + k, 0] = a; p = 1
            if b >= 0: out[base + k, p] = b; p += 1
            out[base + k, p] = c; p += 1
            if d >= 0: out[base + k, p] = d; p += 1
            out[base + k, :p] = np.sort(out[base + k, :p])
        k += 1
    return k


@njit(parallel=True, cache=True)
def _count(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax):
    m = hv.shape[0]; cnt = np.zeros(m, np.int64)
    dummy = np.zeros((1, 4), np.int64)
    for L in prange(m):
        cnt[L] = _one_left(L, hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, dummy, 0, False)
    return cnt


@njit(parallel=True, cache=True)
def _fill(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, offs, out):
    m = hv.shape[0]
    for L in prange(m):
        if offs[L + 1] > offs[L]:
            _one_left(L, hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, out, offs[L], True)


def price(z, pi, lo, hi, theta):
    z = np.asarray(z, float); pi = np.asarray(pi, float)
    hv, ha, hb, hsa, hsb, hw = mitm._halves(z, pi)
    order = np.argsort(hv, kind="stable"); sv = hv[order]
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))
    cnt = _count(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, float(theta), wmax)
    offs = np.zeros(len(cnt) + 1, np.int64); np.cumsum(cnt, out=offs[1:])
    out = np.full((int(offs[-1]), 4), -1, np.int64)
    _fill(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, float(theta), wmax, offs, out)
    rows = np.unique(out, axis=0)
    sizes = (rows >= 0).sum(1)
    sc = np.array([sg._score(z, r, s) for r, s in zip(rows, sizes)]) if len(rows) else np.zeros(0)
    rc = sc - np.array([pi[r[:s]].sum() for r, s in zip(rows, sizes)]) if len(rows) else np.zeros(0)
    keep = rc <= theta
    return rows[keep], sc[keep], rc[keep], int(offs[-1])


@njit(cache=True)
def _scan_i(z, pi, i, lo, hi, thresh, out, base, fill):
    n = z.shape[0]; idx = np.empty(4, np.int64); idx[0] = i; k = 0
    for j in range(i + 1, n):
        idx[1] = j
        if lo <= 2:
            if sg._score(z, idx, 2) - pi[i] - pi[j] <= thresh:
                if fill: out[base + k, 0] = i; out[base + k, 1] = j
                k += 1
        if hi < 3: continue
        for c in range(j + 1, n):
            idx[2] = c
            if lo <= 3:
                if sg._score(z, idx, 3) - pi[i] - pi[j] - pi[c] <= thresh:
                    if fill: out[base + k, 0] = i; out[base + k, 1] = j; out[base + k, 2] = c
                    k += 1
            if hi < 4: continue
            for d in range(c + 1, n):
                idx[3] = d
                if sg._score(z, idx, 4) - pi[i] - pi[j] - pi[c] - pi[d] <= thresh:
                    if fill: out[base + k, 0] = i; out[base + k, 1] = j; out[base + k, 2] = c; out[base + k, 3] = d
                    k += 1
    return k


@njit(parallel=True, cache=True)
def _scan_count(z, pi, lo, hi, thresh):
    n = z.shape[0]; cnt = np.zeros(n, np.int64); dummy = np.zeros((1, 4), np.int64)
    for i in prange(n):
        cnt[i] = _scan_i(z, pi, i, lo, hi, thresh, dummy, 0, False)
    return cnt


@njit(parallel=True, cache=True)
def _scan_fill(z, pi, lo, hi, thresh, offs, out):
    n = z.shape[0]
    for i in prange(n):
        _scan_i(z, pi, i, lo, hi, thresh, out, offs[i], True)


def scan(z, pi, lo, hi, thresh):
    z = np.asarray(z, float); pi = np.asarray(pi, float)
    cnt = _scan_count(z, pi, lo, hi, float(thresh))
    offs = np.zeros(len(cnt) + 1, np.int64); np.cumsum(cnt, out=offs[1:])
    out = np.full((int(offs[-1]), 4), -1, np.int64)
    _scan_fill(z, pi, lo, hi, float(thresh), offs, out)
    return out


# ---- balanced versions -------------------------------------------------------
BLOCKS = 256


@njit(parallel=True, cache=True)
def _scan_count_cyc(z, pi, lo, hi, thresh):
    n = z.shape[0]; cnt = np.zeros(n, np.int64); dummy = np.zeros((1, 4), np.int64)
    for b in prange(BLOCKS):                        # block b takes i = b, b + BLOCKS, ...: heavy and light mixed
        for i in range(b, n, BLOCKS):
            cnt[i] = _scan_i(z, pi, i, lo, hi, thresh, dummy, 0, False)
    return cnt


@njit(parallel=True, cache=True)
def _scan_fill_cyc(z, pi, lo, hi, thresh, offs, out):
    n = z.shape[0]
    for b in prange(BLOCKS):
        for i in range(b, n, BLOCKS):
            if offs[i + 1] > offs[i]:
                _scan_i(z, pi, i, lo, hi, thresh, out, offs[i], True)


def scan_cyc(z, pi, lo, hi, thresh):
    z = np.asarray(z, float); pi = np.asarray(pi, float)
    cnt = _scan_count_cyc(z, pi, lo, hi, float(thresh))
    offs = np.zeros(len(cnt) + 1, np.int64); np.cumsum(cnt, out=offs[1:])
    out = np.full((int(offs[-1]), 4), -1, np.int64)
    _scan_fill_cyc(z, pi, lo, hi, float(thresh), offs, out)
    return out


@njit(parallel=True, cache=True)
def _score_rows(z, pi, rows):
    m = rows.shape[0]; sc = np.empty(m); rc = np.empty(m)
    for t in prange(m):
        k = 0
        for q in range(4):
            if rows[t, q] >= 0: k += 1
        s = sg._score(z, rows[t], k); r = s
        for q in range(k):
            r -= pi[rows[t, q]]
        sc[t] = s; rc[t] = r
    return sc, rc


@njit(parallel=True, cache=True)
def _count_cyc(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax):
    m = hv.shape[0]; cnt = np.zeros(m, np.int64); dummy = np.zeros((1, 4), np.int64)
    for b in prange(BLOCKS):
        for L in range(b, m, BLOCKS):
            cnt[L] = _one_left(L, hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, dummy, 0, False)
    return cnt


@njit(parallel=True, cache=True)
def _fill_cyc(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, offs, out):
    m = hv.shape[0]
    for b in prange(BLOCKS):
        for L in range(b, m, BLOCKS):
            if offs[L + 1] > offs[L]:
                _one_left(L, hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, theta, wmax, out, offs[L], True)


def price2(z, pi, lo, hi, theta):
    """price() with balanced blocks, integer-key de-duplication and parallel scoring."""
    z = np.asarray(z, float); pi = np.asarray(pi, float); n = len(z)
    hv, ha, hb, hsa, hsb, hw = mitm._halves(z, pi)
    order = np.argsort(hv, kind="stable"); sv = hv[order]
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))
    cnt = _count_cyc(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, float(theta), wmax)
    offs = np.zeros(len(cnt) + 1, np.int64); np.cumsum(cnt, out=offs[1:])
    out = np.full((int(offs[-1]), 4), -1, np.int64)
    _fill_cyc(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, float(theta), wmax, offs, out)
    b1 = n + 1
    key = (((out[:, 0] + 1) * b1 + out[:, 1] + 1) * b1 + out[:, 2] + 1) * b1 + out[:, 3] + 1
    _, first = np.unique(key, return_index=True)
    rows = out[first]
    sc, rc = _score_rows(z, pi, rows)
    keep = rc <= theta
    return rows[keep], sc[keep], rc[keep], int(offs[-1])


def price_capped(z, pi, lo, hi, theta, cap=200_000):
    """price2 at theta, or, if more than `cap` window hits, at the lower threshold
    (found by bisection on the count-only pass) that keeps about `cap` of them:
    the most negative reduced costs, which is all column generation needs."""
    z = np.asarray(z, float); pi = np.asarray(pi, float)
    hv, ha, hb, hsa, hsb, hw = mitm._halves(z, pi)
    order = np.argsort(hv, kind="stable"); sv = hv[order]
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))
    def count(th):
        return int(_count_cyc(hv, ha, hb, hsa, hsb, hw, order, sv, lo, hi, float(th), wmax).sum())
    th = float(theta)
    if count(th) > cap:
        lo_t = -abs(th) - 1.0
        while count(lo_t) > cap:
            lo_t *= 4.0
        hi_t = th
        for _ in range(40):
            mid = 0.5 * (lo_t + hi_t)
            if count(mid) > cap: hi_t = mid
            else: lo_t = mid
        th = lo_t
    return price2(z, pi, lo, hi, th)
