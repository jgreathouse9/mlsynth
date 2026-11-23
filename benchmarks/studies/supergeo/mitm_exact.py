"""Exact (verified) meet-in-the-middle pricing.

Floating point finds a superset, exact rational arithmetic decides membership
(Rump 2010, Acta Numerica, Sect. 2-3: compute in floating point, bound the
rounding rigorously, settle what the bound cannot). Every double is an exact
rational, so `Fraction` gives the reduced cost of a group with no rounding:

    rc(G) = min over non-empty proper A of (2 Z_A - S_G)^2  -  sum_{i in G} pi_i.

Rounding model (Rump, eq. (2.x); Higham 2002, Sect. 3.1): recursive summation of
n terms has |fl(sum x) - sum x| <= gamma_{n-1} sum |x|, gamma_n = n u / (1 - n u),
u = 2^-53. The kernel `sg._score` forms S and Z_A by recursive summation of at
most 4 terms, d = 2a - S (2a exact), d*d, and `sg._scan` subtracts at most 4
duals. With Z = sum_G |z_i| and Pi = sum_G |pi_i| this gives

    |d~ - d| <= 4 gamma_4 Z,   |fl(d~^2) - d^2| <= 12 gamma_4 Z^2,
    |rc~ - rc| <= 14 gamma_4 Z^2 + gamma_4 Pi,

and over every group Z <= 4M, Pi <= 4P (M = max|z|, P = max|pi|):
DELTA_RC = 2 (224 gamma_4 M^2 + 4 gamma_4 P), the factor 2 covering the
floating-point evaluation of the bound itself (Rump's gamma~_{n+1} device, with
room to spare).

The window. A group with exact rc <= theta has |v_L + v_R| <= sqrt(theta + pi_L
+ pi_R) <= sqrt(theta + pi_L + Rmax) in exact arithmetic. The search compares
floating-point half values, each within gamma_1 * 2M of exact, and a radius
computed in floating point; both are widened so the floating-point window holds
every exact member. Membership is then decided by `exact_rc`."""
from fractions import Fraction
import itertools
import numpy as np
from numba import njit, prange
from . import sg, mitm

U = 2.0 ** -53


def gamma(n):
    return n * U / (1 - n * U)


def delta_rc(z, pi):
    M = float(np.max(np.abs(z))); P = float(np.max(np.abs(pi))) if len(pi) else 0.0
    return 2.0 * (224 * gamma(4) * M * M + 4 * gamma(4) * P)


def exact_rc(z, pi, g):
    """Reduced cost of group g in exact rational arithmetic."""
    zs = [Fraction(float(z[i])) for i in g]
    S = sum(zs); k = len(g)
    best = min((2 * sum(zs[t] for t in range(k) if m >> t & 1) - S) ** 2 for m in range(1, (1 << k) - 1))
    return best - sum(Fraction(float(pi[i])) for i in g)


def exact_set(z, pi, lo, hi, theta):
    """Every group with exact rc <= theta, by brute force (small n only)."""
    th = Fraction(float(theta))
    return {g for k in range(lo, hi + 1) for g in itertools.combinations(range(len(z)), k) if exact_rc(z, pi, g) <= th}


@njit(cache=True)
def _window_safe(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v, cap):
    """As mitm._window, with the radius widened and no floating-point acceptance test."""
    m = hv.shape[0]
    sv = hv[order]
    out = np.full((cap, 4), -1, np.int64); k = 0
    for L in range(m):
        if hsa[L] != 1:
            continue
        a, b = ha[L], hb[L]
        nl = 1 if b < 0 else 2
        r2 = theta + hw[L] + wmax + slack_r2
        if r2 < 0:
            continue
        rad = np.sqrt(r2) * (1.0 + 4e-16) + slack_v
        target = -hv[L]
        start = np.searchsorted(sv, target - rad)
        for q in range(start, m):
            if sv[q] > target + rad:
                break
            R = order[q]
            c, d = ha[R], hb[R]
            nr = 1 if d < 0 else 2
            size = nl + nr
            if size < lo or size > hi:
                continue
            if c <= a or (d >= 0 and d <= a):
                continue
            if c == b or (d >= 0 and d == b):
                continue
            if not ((b >= 0 and hsb[L] == -1) or hsa[R] == -1 or (d >= 0 and hsb[R] == -1)):
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


@njit(cache=True)
def _window_half(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v, sv, L, out, k, fill):
    """_window_safe's work for one left half L: the number of rows it emits, written from row k of `out` when
    `fill` is set. The same tests in the same order as _window_safe, so the rows are its rows."""
    if hsa[L] != 1:
        return 0
    a, b = ha[L], hb[L]
    nl = 1 if b < 0 else 2
    r2 = theta + hw[L] + wmax + slack_r2
    if r2 < 0:
        return 0
    rad = np.sqrt(r2) * (1.0 + 4e-16) + slack_v
    target = -hv[L]
    start = np.searchsorted(sv, target - rad)
    m = hv.shape[0]; c0 = 0
    g = np.empty(4, np.int64)
    for q in range(start, m):
        if sv[q] > target + rad:
            break
        R = order[q]
        c, d = ha[R], hb[R]
        nr = 1 if d < 0 else 2
        size = nl + nr
        if size < lo or size > hi:
            continue
        if c <= a or (d >= 0 and d <= a):
            continue
        if c == b or (d >= 0 and d == b):
            continue
        if not ((b >= 0 and hsb[L] == -1) or hsa[R] == -1 or (d >= 0 and hsb[R] == -1)):
            continue
        if fill:
            g[:] = -1
            g[0] = a; p = 1
            if b >= 0: g[p] = b; p += 1
            g[p] = c; p += 1
            if d >= 0: g[p] = d; p += 1
            g[:p] = np.sort(g[:p])
            for t in range(4):
                out[k + c0, t] = g[t]
        c0 += 1
    return c0


@njit(parallel=True, cache=True)
def _window_counts(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v):
    """The number of rows _window_safe emits for each left half, counted in parallel without allocating them."""
    sv = hv[order]; m = hv.shape[0]
    cnt = np.zeros(m, np.int64)
    dummy = np.empty((0, 4), np.int64)
    for L in prange(m):
        cnt[L] = _window_half(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v, sv, L,
                              dummy, 0, False)
    return cnt


@njit(parallel=True, cache=True)
def _window_fill(cnt, hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v):
    """The rows _window_counts counted, filled in parallel at offsets from the prefix sums of `cnt`."""
    sv = hv[order]; m = hv.shape[0]
    off = np.zeros(m + 1, np.int64)
    for L in range(m):
        off[L + 1] = off[L] + cnt[L]
    out = np.full((off[m], 4), -1, np.int64)
    for L in prange(m):
        if cnt[L] > 0:
            _window_half(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v, sv, L, out,
                         off[L], True)
    return out


@njit(parallel=True, cache=True)
def _window_safe_par(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v):
    """_window_safe on threads: each left half's rows counted in parallel (_window_counts), offsets from their prefix
    sums, and the rows filled in parallel at those offsets (_window_fill). Every row of _window_safe, the same number
    (with repeats: a group is emitted once per split that passes). At 120 markets (24,088,508 rows) 3.8 s against
    14.5 s for the serial count and fill."""
    cnt = _window_counts(hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v)
    return _window_fill(cnt, hv, ha, hb, hsa, hsb, hw, order, lo, hi, theta, wmax, slack_r2, slack_v)


def price_exact(z, pi, lo, hi, theta, cap=50_000_000, slack=True):
    """The exact set {G : rc(G) <= theta}: a widened floating-point window, then
    exact membership. `slack=False` drops the widening (for showing the tests
    can see it)."""
    z = np.asarray(z, float); pi = np.asarray(pi, float)
    hv, ha, hb, hsa, hsb, hw = mitm._halves(z, pi)
    order = np.argsort(hv, kind="stable")
    M = float(np.max(np.abs(z))); P = float(np.max(np.abs(pi)))
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))
    if slack:
        slack_r2 = 8 * U * (abs(theta) + 6 * P) + 2 * U * abs(wmax)
        slack_v = 16 * U * 4 * M
    else:
        slack_r2 = 0.0; slack_v = 0.0
    _, k = _window_safe(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, slack_r2, slack_v, 0)   # count
    if k > cap:
        raise MemoryError(f"{k} candidates exceed cap {cap}")
    rows, k = _window_safe(hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(theta), wmax, slack_r2, slack_v, k)
    if k > cap:
        raise MemoryError(f"{k} candidates exceed cap {cap}")
    rows = np.unique(rows, axis=0)
    th = Fraction(float(theta))
    out = set()
    for r in rows:
        g = tuple(int(v) for v in r if v >= 0)
        if exact_rc(z, pi, g) <= th:
            out.add(g)
    return out, k


def reference_exact(z, pi, lo, hi, theta, cap=5_000_000):
    """Exact set at any n: the full O(n^4) floating-point scan at theta + DELTA_RC
    (a superset, by the bound above), then exact membership."""
    d = delta_rc(z, pi)
    rows, s_, rc, m = sg._scan(np.asarray(z, float), np.asarray(pi, float), lo, hi, float(theta) + d, cap)
    if m >= cap:
        raise MemoryError("cap reached")
    th = Fraction(float(theta))
    return {g for g in (tuple(int(v) for v in r if v >= 0) for r in rows) if exact_rc(z, pi, g) <= th}
