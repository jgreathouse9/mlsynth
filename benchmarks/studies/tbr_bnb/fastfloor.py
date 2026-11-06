"""The floors of many treatment groups in one compiled loop, when numba is installed.

For each group the kernel computes what treatment.rss_floor computes, without a
Python call per group:

1. the series y = the sum of the treated columns, left to right in ascending column
   order, which is how numpy sums fewer than eight columns, so y is _split's bit for
   bit; with eight or more numpy sums pairwise and the group is left to rss_floor;
2. a bracket lo <= y'y <= hi from the plain sum s of the squares: each square of an
   entry that is zero or between 2^-500 and 2^500 is a normal double, and the sum is
   within gamma_n y'y of y'y (Bjorck 2015, eq. (1.4.10), with a = b = y), so
   y'y lies in [s / (1 + g), s / (1 - g)] with g = gamma_{n+2}, which the bounds
   s (1 - g) and s (1 + 2 g), each rounded one step outward, contain (derived here).
   numba has no math.fsum, so this bracket is wider than yy_bracket's, and the
   screen, which decides across its own margin and falls back to the exact y'y, reads
   it the same way;
3. the slope cap sqrt(hi / lambda) (1 + 1e-9), as treatment.slope_cap computes it in
   doubles, when the quotient is a normal double;
4. an NNLS point: Lawson and Hanson's (1974) active set, in Gram form, on the group's
   Gram and cross products, taken from the panel's, and
   started from the previous group's support carried over by geo. The certificate
   holds at any point of the box, so the point only decides how tight the floor is;
5. certificate.box_bound_float at that point, clipped to [0, cap], term for term:
   the same error bounds, each a sum computed left to right, which the a priori
   bounds cover for any order of summation, and the last three additions each
   rounded one step down.

A group the kernel cannot do in doubles comes back with a nan floor (and a nan
bracket when the entries leave the bracket's range), and treatment._floor_chunk
computes it with rss_floor.

Lawson, C. L. and Hanson, R. J. (1974). Solving Least Squares Problems.
    Prentice-Hall.
Bjorck, A. (2015). Numerical Methods in Matrix Computations. Springer.
"""
from __future__ import annotations

import math

import numpy as np

try:  # numba is the optional ``fast`` extra
    from numba import njit
    AVAILABLE = True
except ImportError:  # pragma: no cover - the suite runs with numba installed
    AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name. numba keys its cache
#: files on the source file but records the module's name in them, so a cache written
#: when the module was imported as ``studies.tbr_bnb.fastfloor`` (the suite's import
#: name) fails to load where it is ``benchmarks.studies.tbr_bnb.fastfloor``, with an
#: import error from inside numba. Under any other name the kernel compiles in memory.
CANONICAL_NAME = "benchmarks.studies.tbr_bnb.fastfloor"
_CACHE = __name__ == CANONICAL_NAME

#: numpy sums fewer than this many columns left to right.
MAX_TREATED = 7
#: The range of an entry whose square is a normal double.
_TINY, _HUGE = 2.0 ** -500, 2.0 ** 500
#: The range of y'y / lambda over which the float cap's rounding is relative.
_CAP_LO, _CAP_HI = 2.0 ** -1000, 2.0 ** 1000
_U = 2.0 ** -53
_ETA = 2.0 ** -1074


@njit(cache=_CACHE)
def _gamma(k):
    return k * _U / (1.0 - k * _U)


@njit(cache=_CACHE)
def _dot_err(k, moduli):
    return _gamma(k) * moduli + k * _ETA


@njit(cache=_CACHE)
def _add_down(a, b):
    s = a + b
    if not np.isfinite(s):
        return -np.inf
    return np.nextafter(s, -np.inf)


@njit(cache=_CACHE)
def _chol_solve(A, b, idx, k, L, z, x):
    """x = A[idx, idx]^{-1} b[idx] by Cholesky; False when A[idx, idx] is not positive
    definite in doubles."""
    for i in range(k):
        for j in range(i + 1):
            s = A[idx[i], idx[j]]
            for q in range(j):
                s -= L[i, q] * L[j, q]
            if i == j:
                if s <= 0.0:
                    return False
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    for i in range(k):
        s = b[idx[i]]
        for q in range(i):
            s -= L[i, q] * z[q]
        z[i] = s / L[i, i]
    for i in range(k - 1, -1, -1):
        s = z[i]
        for q in range(i + 1, k):
            s -= L[q, i] * x[q]
        x[i] = s / L[i, i]
    return True


@njit(cache=_CACHE)
def _nnls_gram(G, c, passive, w, idx, L, z, x, maxit):
    """min w'Gw - 2c'w over w >= 0, Lawson and Hanson's active set from ``passive``."""
    m = c.shape[0]
    cmax = 1.0
    for j in range(m):
        if abs(c[j]) > cmax:
            cmax = abs(c[j])
    tol = 1e-12 * cmax
    for j in range(m):
        w[j] = 0.0
    while True:                                  # a feasible start: drop until the solve is positive
        k = 0
        for j in range(m):
            if passive[j]:
                idx[k] = j
                k += 1
        if k == 0:
            break
        ok = _chol_solve(G, c, idx, k, L, z, x)
        bad = False
        for q in range(k):
            if (not ok) or not (x[q] > 0.0):
                passive[idx[q]] = False
                bad = True
        if not bad:
            for q in range(k):
                w[idx[q]] = x[q]
            break
    for _ in range(maxit):
        best = tol
        jbest = -1
        for j in range(m):                       # the coordinate that most lowers the objective
            if not passive[j]:
                g = c[j]
                for q in range(m):
                    g -= G[j, q] * w[q]
                if g > best:
                    best = g
                    jbest = j
        if jbest < 0:
            return
        passive[jbest] = True
        while True:
            k = 0
            for j in range(m):
                if passive[j]:
                    idx[k] = j
                    k += 1
            ok = _chol_solve(G, c, idx, k, L, z, x)
            if not ok:                           # the column is in the set's span: x is stale,
                passive[jbest] = False           # so stop at w, the last feasible point
                return
            allpos = True
            for q in range(k):
                if not (x[q] > 0.0):
                    allpos = False
            if allpos:
                for j in range(m):
                    w[j] = 0.0
                for q in range(k):
                    w[idx[q]] = x[q]
                break
            alpha = 1.0                          # step back to the first coordinate that hits zero
            for q in range(k):
                if not (x[q] > 0.0):
                    wq = w[idx[q]]
                    a = wq / (wq - x[q]) if wq - x[q] != 0.0 else 0.0
                    if a < alpha:
                        alpha = a
            for q in range(k):
                j = idx[q]
                w[j] = w[j] + alpha * (x[q] - w[j])
                if w[j] <= 1e-15 * max(1.0, abs(x[q])):
                    w[j] = 0.0
                    passive[j] = False


@njit(cache=_CACHE)
def _certificate(M, rest, y, w, cap, r, e, ar, g, d):
    """box_bound_float's pieces at w over the box {0 <= x <= cap}: (f, kn, gw, err),
    the bound being f + kn - gw - err with each addition rounded down. Term for term
    certificate.residual_bound, objective_bound, gradient_bound, the knapsack at
    multiplier 0 with perturbed_knapsack_bound's perturbation, linear_bound and
    inflate, each sum computed left to right."""
    n = y.shape[0]
    m = rest.shape[0]
    for t in range(n):
        acc = 0.0
        mod = 0.0
        for j in range(m):
            xv = M[t, rest[j]]
            acc += xv * w[j]
            mod += abs(xv) * abs(w[j])
        r[t] = acc - y[t]
        e[t] = _dot_err(m + 1, mod + abs(y[t]))
        ar[t] = abs(r[t])
    f = 0.0
    ff = 0.0
    fe = 0.0
    for t in range(n):
        f += r[t] * r[t]
        ff += ar[t] * ar[t]
        fe += ar[t] * e[t]
    f_err = _dot_err(n, ff) + 2.0 * fe
    dmax = 0.0
    for j in range(m):
        acc = 0.0
        mr = 0.0
        me = 0.0
        for t in range(n):
            xv = M[t, rest[j]]
            acc += xv * r[t]
            mr += abs(xv) * ar[t]
            me += abs(xv) * e[t]
        g[j] = 2.0 * acc
        d[j] = 2.0 * _dot_err(n, mr) + 2.0 * me
        if d[j] > dmax:
            dmax = d[j]
    kn = 0.0
    kabs = 0.0
    for j in range(m):
        tv = cap * min(0.0, g[j])
        kn += tv
        kabs += abs(tv)
    kn_err = _dot_err(m + 3, kabs) + dmax * (m * cap)
    gw = 0.0
    gwabs = 0.0
    dw = 0.0
    for j in range(m):
        gw += g[j] * w[j]
        gwabs += abs(g[j]) * abs(w[j])
        dw += d[j] * abs(w[j])
    gw_err = _dot_err(m, gwabs) + dw
    err = (f_err + kn_err + gw_err) * (1.0 + 1e-6)
    return f, kn, gw, err


@njit(cache=_CACHE)
def _one(M, Gf, T, lamf, rest, passive, y, r, e, ar, G, c, w, idx, L, z, x, g, d):
    """The floor, bracket and cap of one group (T sorted), the point left in w.
    Returns (floor, lo, hi, cap); nan where the group is left to rss_floor."""
    n, p = M.shape
    kk = T.shape[0]
    m = rest.shape[0]
    nan = np.nan
    if kk > MAX_TREATED:
        return nan, nan, nan, nan
    inrange = True
    s = 0.0
    for t in range(n):
        v = 0.0
        for q in range(kk):
            v += M[t, T[q]]
        y[t] = v
        av = abs(v)
        if not (np.isfinite(v) and (v == 0.0 or (av >= _TINY and av <= _HUGE))):
            inrange = False
        s += v * v
    if not inrange or n > 2 ** 20:
        return nan, nan, nan, nan
    if s == 0.0:                                 # every entry zero: in range, a nonzero square is >= 2^-1000
        lo = 0.0
        hi = 0.0
    else:
        gm = _gamma(n + 2)
        lo = np.nextafter(s * (1.0 - gm), -np.inf)
        hi = np.nextafter(s * (1.0 + 2.0 * gm), np.inf)
    if hi == 0.0:
        cap = 0.0
    else:
        if not lamf > 0.0:
            return nan, lo, hi, nan
        qv = hi / lamf
        if not (qv >= _CAP_LO and qv <= _CAP_HI):
            return nan, lo, hi, nan
        cap = math.sqrt(qv) * (1.0 + 1e-9)
    for i in range(m):                           # the group's Gram and cross products from the panel's
        acc = 0.0
        for q in range(kk):
            acc += Gf[rest[i], T[q]]
        c[i] = acc
        for j in range(m):
            G[i, j] = Gf[rest[i], rest[j]]
    _nnls_gram(G, c, passive, w, idx, L, z, x, 500)
    for j in range(m):
        if not (w[j] >= 0.0):
            w[j] = 0.0  # pragma: no cover - the active set keeps w >= 0 (positive solves, step-backs clipped to 0)
        if w[j] > cap:
            w[j] = cap
    f, kn, gw, err = _certificate(M, rest, y, w, cap, r, e, ar, g, d)
    return _add_down(_add_down(_add_down(f, kn), -gw), -err), lo, hi, cap


@njit(cache=_CACHE)
def _sorted(T):
    out = T.copy()
    for i in range(1, out.shape[0]):
        v = out[i]
        j = i - 1
        while j >= 0 and out[j] > v:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = v
    return out


@njit(cache=_CACHE)
def _controls(p, T):
    out = np.empty(p - T.shape[0], np.int64)
    q = 0
    for j in range(p):
        inT = False
        for t in range(T.shape[0]):
            if T[t] == j:
                inT = True
        if not inT:
            out[q] = j
            q += 1
    return out


@njit(cache=_CACHE)
def floors(M, Gf, groups, lamf):
    """(floor, lo, hi) for each row of ``groups``: the box bound (not yet clipped at
    zero) and the bracket on y'y, nan where the group is left to rss_floor. M is the
    panel, Gf = M'M, and lamf the Gram floor as a double."""
    n, p = M.shape
    N = groups.shape[0]
    kk = groups.shape[1]
    m = p - kk
    out_f = np.empty(N)
    out_lo = np.empty(N)
    out_hi = np.empty(N)
    y = np.empty(n); r = np.empty(n); e = np.empty(n); ar = np.empty(n)
    G = np.empty((m, m)); c = np.empty(m); w = np.empty(m)
    idx = np.empty(m, np.int64); L = np.empty((m, m)); z = np.empty(m); x = np.empty(m)
    g = np.empty(m); d = np.empty(m)
    passive = np.zeros(m, np.bool_)
    prev = np.full(m, -1, np.int64)
    carry = np.zeros(m, np.bool_)
    for gi in range(N):
        T = _sorted(groups[gi])
        rest = _controls(p, T)
        for i in range(m):                       # the previous support, carried over by geo
            carry[i] = False
            for i2 in range(m):
                if prev[i2] == rest[i] and passive[i2]:
                    carry[i] = True
        for i in range(m):
            passive[i] = carry[i]
            prev[i] = rest[i]
        fl, lo, hi, cap = _one(M, Gf, T, lamf, rest, passive, y, r, e, ar, G, c, w, idx, L, z, x, g, d)
        out_f[gi] = fl
        out_lo[gi] = lo
        out_hi[gi] = hi
    return out_f, out_lo, out_hi


@njit(cache=_CACHE)
def floor_parts(M, Gf, T, lamf):
    """One group from a cold start, with its point and series: (floor, lo, hi, w, y,
    cap), w over the controls in ascending order. For tests and inspection."""
    n, p = M.shape
    T = _sorted(T)
    rest = _controls(p, T)
    m = rest.shape[0]
    y = np.zeros(n); r = np.empty(n); e = np.empty(n); ar = np.empty(n)
    G = np.empty((m, m)); c = np.empty(m); w = np.zeros(m)
    idx = np.empty(m, np.int64); L = np.empty((m, m)); z = np.empty(m); x = np.empty(m)
    g = np.empty(m); d = np.empty(m)
    passive = np.zeros(m, np.bool_)
    fl, lo, hi, cap = _one(M, Gf, T, lamf, rest, passive, y, r, e, ar, G, c, w, idx, L, z, x, g, d)
    return fl, lo, hi, w, y, cap


@njit(cache=_CACHE)
def certificate_parts(M, T, w, cap):
    """The certificate's pieces (f, kn, gw, err) for group T at the point w (over the
    controls in ascending order) and cap. For tests."""
    n, p = M.shape
    T = _sorted(T)
    rest = _controls(p, T)
    m = rest.shape[0]
    y = np.empty(n)
    for t in range(n):
        v = 0.0
        for q in range(T.shape[0]):
            v += M[t, T[q]]
        y[t] = v
    r = np.empty(n); e = np.empty(n); ar = np.empty(n); g = np.empty(m); d = np.empty(m)
    return _certificate(M, rest, y, w.copy(), cap, r, e, ar, g, d)
