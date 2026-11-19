"""The pentagonal separation's local search in one compiled loop, when numba is
installed (#965).

slopefree.separate_pentagons looks for five-sets S and signs b with b'Y_SS b small,
Y the point's sign matrix: from each of its starts it replaces one member at a time
by the index and sign that lower b'Y_SS b most, positions 0 to 4 in order, the first
index on ties, and a move only when it lowers the value by more than 1e-12, until a
pass moves nothing (slopefree._pentagon_search). In Python that loop is a handful of
small numpy calls a step, about a second a call at any panel size, nearly all of it
the interpreter, and it holds the interpreter lock, so sizes run on threads wait on
one another. ``search`` is the same loop compiled with ``nogil``.

It takes the same steps. Each step's values are the same sums: b'Y_SS b over the
five members, the base over the four kept, and lin_v = sum_{a != t} Y_{v, S_a} b_a,
each candidate d_v + base - 2 |lin_v| at its best sign -sign(lin_v). Only the order of
the additions differs from numpy's, so at a point whose entries are dyadic with small
denominators every value is exact and the two searches agree step for step
(test_fastpent.py); elsewhere a value may differ in its last bit, which can change a
step only where two candidates or a candidate and the current value are within that
bit, and every start still ends at a local minimum of the same rule. A row is kept or
not on its own violation, computed from the value the search ends at, so whichever
row a start reports is a row the point violates by what is reported.
"""
from __future__ import annotations

import numpy as np

from mlsynth.exceptions import MlsynthConfigError

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
CANONICAL_NAME = "mlsynth.utils.tbr_helpers.design.exact.fastpent"
_CACHE = __name__ == CANONICAL_NAME


@njit(nogil=True, cache=_CACHE)
def _search(Y, d, S0, B0):  # pragma: no cover - compiled; the suite checks it against the Python search
    R = S0.shape[0]; n = Y.shape[0]
    S = S0.copy(); B = B0.copy(); val = np.empty(R)
    lin = np.empty(n); cand = np.empty(n)
    for i in range(R):
        v0 = 0.0
        for a in range(5):
            for c in range(5):
                v0 += B[i, a] * Y[S[i, a], S[i, c]] * B[i, c]
        vi = v0
        moved = True
        while moved:
            moved = False
            for t in range(5):
                base = 0.0
                for a in range(5):
                    if a != t:
                        for c in range(5):
                            if c != t:
                                base += B[i, a] * Y[S[i, a], S[i, c]] * B[i, c]
                for v in range(n):
                    s = 0.0
                    for a in range(5):
                        if a != t:
                            s += Y[v, S[i, a]] * B[i, a]
                    lin[v] = s
                    cand[v] = base + d[v] - 2.0 * abs(s)
                for a in range(5):
                    if a != t:
                        cand[S[i, a]] = np.inf
                best = 0
                for v in range(1, n):
                    if cand[v] < cand[best]:
                        best = v
                if cand[best] < vi - 1e-12:
                    S[i, t] = best
                    B[i, t] = -1 if lin[best] > 0 else 1
                    vi = cand[best]
                    moved = True
        val[i] = vi
    return S, B, val


def search(Y, d, S0, B0):
    """The local search from every start (S0[i], B0[i]): the five-sets and signs each
    ends at and their values b'Y_SS b, as slopefree._pentagon_search returns them.
    The starts are not changed. A start the search cannot run from (not five distinct
    indices of Y, signs not +-1) or a sign matrix that is not square is refused."""
    Y = np.ascontiguousarray(Y, dtype=float); d = np.ascontiguousarray(d, dtype=float)
    S0 = np.ascontiguousarray(S0, dtype=np.int64); B0 = np.ascontiguousarray(B0, dtype=np.int64)
    n = Y.shape[0]
    if Y.ndim != 2 or Y.shape[1] != n or d.shape != (n,):
        raise MlsynthConfigError(f"the sign matrix must be square with its diagonal beside it, not {Y.shape} and {d.shape}.")
    if S0.ndim != 2 or S0.shape[1] != 5 or B0.shape != S0.shape:
        raise MlsynthConfigError(f"each start is five indices and five signs, not {S0.shape} and {B0.shape}.")
    if len(S0) and (S0.min() < 0 or S0.max() >= n or any(len(set(r)) < 5 for r in S0.tolist())):
        raise MlsynthConfigError(f"a start's indices must be five distinct indices of the {n} rows of Y.")
    if len(B0) and not np.all(np.abs(B0) == 1):
        raise MlsynthConfigError("a start's signs must each be +1 or -1.")
    return _search(Y, d, S0, B0)
