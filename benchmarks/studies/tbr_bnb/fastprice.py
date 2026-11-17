"""Pricing's bound pass in one compiled loop, when numba is installed (#965).

slopefree.price_subtree prices a subtree design by design: an upper bound in doubles
on each design's ratio (c'u)^2 / u'Gu (slopefree._upper_batch), the designs whose
bound exceeds the incumbent's ratio kept and evaluated exactly. In Python the bound
pass builds the designs with itertools in chunks and gathers a k x k block of G per
design, about a microsecond a design, nearly all of it the interpreter, and holds the
interpreter lock, so sizes on threads wait for each other's pricing.

``bound_pass`` walks the designs in place, in combinations' lexicographic order,
compiled with ``nogil``, and computes each one's bound with _upper_batch's formulas:

    s = c1 + sum_S c_j,  d = G11 + 2 sum_S g1_j + sum_{S x S} G_ij,
    bound = (|s| + e_s)^2 / (d - e_d) (1 + 1e-6),

with e_s and e_d twice certificate._dot_err over the moduli of the terms. Those error
terms bound recursive summation in any order (Bjorck 2015, Lemma 1.4.2), so the
bound is rigorous although the kernel adds in another order than numpy; at dyadic
data every sum is exact and the two passes agree to the bit (test_fastprice.py). A
design with s + e_s <= 0 has no positive slope and is never kept; one with
d - e_d <= 0 gets an infinite bound and is kept.
"""
from __future__ import annotations

import numpy as np

from mlsynth.exceptions import MlsynthConfigError

from .certificate import ETA, U

try:  # numba is the optional ``fast`` extra
    from numba import njit
    AVAILABLE = True
except ImportError:  # pragma: no cover - the suite runs with numba installed
    AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

#: Compiled code is cached on disk only under this module name (fastfloor.CANONICAL_NAME
#: gives the reason). Under any other name the kernel compiles in memory.
CANONICAL_NAME = "benchmarks.studies.tbr_bnb.fastprice"
_CACHE = __name__ == CANONICAL_NAME


@njit(nogil=True, cache=_CACHE)
def _bound_pass(c1, G11, g1, cf, Gf, k, floor):  # pragma: no cover - compiled; the suite checks it against _upper_batch
    nF = cf.shape[0]
    cap = 1024
    out_c = np.empty((cap, k), np.int64); out_u = np.empty(cap)
    n = 0; total = 0
    idx = np.arange(k)
    ks, kd = k + 2, k * k + k + 2
    gs = ks * U / (1.0 - ks * U); gd = kd * U / (1.0 - kd * U)
    while True:
        total += 1
        s = c1; sa = abs(c1); d = G11; da = abs(G11)
        for a in range(k):
            i = idx[a]
            s += cf[i]; sa += abs(cf[i])
            d += 2.0 * g1[i]; da += 2.0 * abs(g1[i])
            for b in range(k):
                v = Gf[i, idx[b]]
                d += v; da += abs(v)
        e_s = 2.0 * (gs * sa + ks * ETA); e_d = 2.0 * (gd * da + kd * ETA)
        if s + e_s > 0:
            if d - e_d > 0:
                ub = (abs(s) + e_s) ** 2 / (d - e_d) * (1.0 + 1e-6)
            else:
                ub = np.inf
            if ub > floor:
                if n == cap:
                    cap *= 2
                    nc = np.empty((cap, k), np.int64); nc[:n] = out_c[:n]; out_c = nc
                    nu = np.empty(cap); nu[:n] = out_u[:n]; out_u = nu
                out_c[n] = idx; out_u[n] = ub; n += 1
        a = k - 1                                  # the next combination, lexicographically
        while a >= 0 and idx[a] == nF - k + a:
            a -= 1
        if a < 0:
            break
        idx[a] += 1
        for b in range(a + 1, k):
            idx[b] = idx[b - 1] + 1
    return out_c[:n], out_u[:n], total


def bound_pass(fixed, cfF, GfF, k, floor):
    """Every design of k of the free geos (indices into cfF), its bound as
    _upper_batch computes it, and the designs whose bound exceeds ``floor``:
    (their index rows, their bounds, the number of designs bounded). ``fixed`` is
    slopefree._fixed's (c1,), (G11,), (g1,). A k outside 0..len(cfF) or blocks of
    the wrong shapes are refused."""
    (c1,), (G11,), (g1,) = fixed
    cf = np.ascontiguousarray(cfF, dtype=float); Gf = np.ascontiguousarray(GfF, dtype=float)
    g1 = np.ascontiguousarray(g1, dtype=float); nF = cf.shape[0]
    if Gf.shape != (nF, nF) or g1.shape != (nF,):
        raise MlsynthConfigError(f"cannot price: the free block is {Gf.shape} and the fixed row {g1.shape} for {nF} free geos.")
    if type(k) not in (int, np.int64) or not 0 <= k <= nF:
        raise MlsynthConfigError(f"cannot price designs of {k} of {nF} free geos.")
    return _bound_pass(float(c1), float(G11), g1, cf, Gf, int(k), float(floor))
