"""Exact reduced-cost membership in numba, by error-free transformations
(Rump 2010, Sect. 3; Shewchuk 1997).

  TwoSum (Knuth):           x = fl(a + b), x + y = a + b exactly.
  Split / TwoProduct (Dekker, Veltkamp): x = fl(a * b), x + y = a * b exactly,
                            provided no overflow or underflow.
  Grow-Expansion (Shewchuk): adds a double to a non-overlapping expansion
                            exactly; the expansion's sign is the sign of its
                            largest-magnitude non-zero component.

Membership: rc(G) <= theta  <=>  min over splits of d_A^2 - sum pi - theta <= 0,
which holds iff some split has d_A^2 - sum pi - theta <= 0. Each such sign is
decided exactly. Range guard: if any |z|, |pi|, |theta| is outside
[2^-400, 2^400] (or zero is fine), products could under- or overflow, and
the caller falls back to Fraction; `in_range` reports it."""
import numpy as np
from numba import njit, prange

SPLITTER = 134217729.0          # 2^27 + 1


@njit(inline="always")
def two_sum(a, b):
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y


@njit(inline="always")
def split(a):
    c = SPLITTER * a
    hi = c - (c - a)
    return hi, a - hi


@njit(inline="always")
def two_prod(a, b):
    x = a * b
    ah, al = split(a); bh, bl = split(b)
    y = al * bl - (((x - ah * bh) - al * bh) - ah * bl)
    return x, y


@njit
def grow(e, m, b):
    """Shewchuk's Grow-Expansion with zero elimination: e[:m] + b, in place; returns new length."""
    q = b; k = 0
    tmp = np.empty(m + 1)
    for i in range(m):
        q, h = two_sum(q, e[i])
        if h != 0.0:
            tmp[k] = h; k += 1
    if q != 0.0 or k == 0:
        tmp[k] = q; k += 1
    for i in range(k):
        e[i] = tmp[i]
    return k


@njit
def sign_of_sum(terms, nt):
    """Exact sign of sum(terms[:nt])."""
    e = np.zeros(nt + 1); m = 0
    for i in range(nt):
        if terms[i] != 0.0:
            m = grow(e, m, terms[i])
    # non-overlapping, increasing magnitude: the last non-zero component carries the sign
    for i in range(m - 1, -1, -1):
        if e[i] > 0.0: return 1
        if e[i] < 0.0: return -1
    return 0


@njit
def member(z, pi, g, k, theta):
    """Exact: is rc(g) <= theta?"""
    terms = np.empty(48)
    for mask in range(1, (1 << k) - 1):
        # d = sum_A z - sum_B z as an exact expansion (at most 4 doubles)
        d = np.zeros(4); nd = 0
        for t in range(k):
            v = z[g[t]] if (mask >> t) & 1 else -z[g[t]]
            nd = grow(d, nd, v)
        nt = 0
        for i in range(nd):                       # d^2 = sum_i sum_j d_i d_j, each product exact as two doubles
            for j in range(nd):
                x, y = two_prod(d[i], d[j])
                terms[nt] = x; terms[nt + 1] = y; nt += 2
        for t in range(k):
            terms[nt] = -pi[g[t]]; nt += 1
        terms[nt] = -theta; nt += 1
        if sign_of_sum(terms, nt) <= 0:
            return True
    return False


@njit(parallel=True)
def members(z, pi, rows, theta):
    out = np.zeros(rows.shape[0], np.bool_)
    for r in prange(rows.shape[0]):
        k = 0
        for q in range(4):
            if rows[r, q] >= 0: k += 1
        out[r] = member(z, pi, rows[r], k, theta)
    return out


def in_range(*arrays):
    for a in arrays:
        a = np.abs(np.atleast_1d(np.asarray(a, float))); a = a[a > 0]
        if a.size and (a.min() < 2.0 ** -400 or a.max() > 2.0 ** 400):
            return False
    return True


@njit
def member2(z, pi, g, k, theta, tau, strict):
    """Exact: rc(g) <= theta (strict: rc(g) < theta), with rc = score - sum pi - tau [k odd]."""
    terms = np.empty(48)
    for mask in range(1, (1 << k) - 1):
        d = np.zeros(4); nd = 0
        for t in range(k):
            v = z[g[t]] if (mask >> t) & 1 else -z[g[t]]
            nd = grow(d, nd, v)
        nt = 0
        for i in range(nd):
            for j in range(nd):
                x, y = two_prod(d[i], d[j])
                terms[nt] = x; terms[nt + 1] = y; nt += 2
        for t in range(k):
            terms[nt] = -pi[g[t]]; nt += 1
        if k % 2 == 1:
            terms[nt] = -tau; nt += 1
        terms[nt] = -theta; nt += 1
        s = sign_of_sum(terms, nt)
        if s < 0 or (s == 0 and not strict):
            return True
    return False


@njit
def _float_rc(z, pi, g, k, theta, tau):
    """rc(g) - theta in floating point, with rc = min over splits (Z_A - Z_B)^2 - sum pi - tau [k odd], and a bound
    on its error: (value, bound), the bound infinite where the error model does not hold.

    Error bound (u = 2^-53, gamma_n = n u / (1 - n u); Higham, "Accuracy and Stability of Numerical Algorithms",
    2nd ed., SIAM 2002, Lemma 3.1 and eq. 4.4 for recursive summation). With A = sum |z_i| over the k <= 4
    members, a split's signed sum d is computed within gamma_3 A of its value; its square within
    gamma_3 (2 + gamma_3) A^2 + u (1 + gamma_3)^2 A^2 <= 8 u A^2; the minimum over splits moves no more than its
    terms. The dual sum (P = sum |pi_i|) is within gamma_3 P, and the three subtractions of the dual sum, the
    parity dual and theta add at most gamma_3 (A^2 + P + |tau| + |theta|) more, so
    16 u (A^2 + P + |tau| + |theta|) bounds the error. The model assumes no underflow or overflow: when A is below
    2^-400 or above 2^400 (squares near the subnormal range or beyond 2^800) the bound is infinite. A row whose
    value is farther from 0 than its bound is decided by the sign of the value; the others need the exact test.
    This is the floating-point filter in front of an exact predicate of Shewchuk ("Adaptive precision
    floating-point arithmetic and fast robust geometric predicates", Discrete Comput. Geom. 18 (1997) 305-363)."""
    A = 0.0; P = 0.0
    for t in range(k):
        A += abs(z[g[t]]); P += abs(pi[g[t]])
    if not (2.0 ** -400 <= A <= 2.0 ** 400):
        return 0.0, np.inf
    best = np.inf
    for mask in range(1, (1 << k) - 1):
        d = 0.0
        for t in range(k):
            v = z[g[t]]
            d += v if (mask >> t) & 1 else -v
        sq = d * d
        if sq < best:
            best = sq
    ps = 0.0
    for t in range(k):
        ps += pi[g[t]]
    v = best - ps
    if k % 2 == 1:
        v -= tau
    v -= theta
    bnd = 16.0 * 2.0 ** -53 * (A * A + P + abs(tau) + abs(theta))
    if not np.isfinite(v) or not np.isfinite(bnd):
        return 0.0, np.inf
    return v, bnd


@njit(parallel=True)
def members2(z, pi, rows, theta, tau, strict):
    """member2 for every row: decided by the floating-point filter (_float_rc) where its value is farther from 0
    than its error bound, by the exact test otherwise. At 120 markets the 3,768,084 rows of the window took 6.0 to
    7.8 s with the exact test alone; the filter decides all of them, or all but 120 at theta = 0 (the reduced
    costs of the LP's own columns), in 0.04 s."""
    out = np.zeros(rows.shape[0], np.bool_)
    for r in prange(rows.shape[0]):
        k = 0
        for q in range(4):
            if rows[r, q] >= 0: k += 1
        v, b = _float_rc(z, pi, rows[r], k, theta, tau)
        if v > b:
            out[r] = False
        elif v < -b:
            out[r] = True
        else:
            out[r] = member2(z, pi, rows[r], k, theta, tau, strict)
    return out
