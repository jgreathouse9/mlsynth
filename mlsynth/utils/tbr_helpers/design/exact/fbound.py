"""The slope-free certificate in floating point (#965).

slopefree.bound sums S = [[0, 0], [0, A]] - sum_k y_k E_k and k = sum_k y_k b_k in
rationals and bounds lambda_max(S) above exactly. Here they are summed in doubles,
with an entrywise bound E on the rounding,

    |fl(S) - S| <= E,   |fl(k) - k| <= e_k,

and lambda_max(S) <= lambda_max(fl(S)) + ||E||_inf by Weyl's inequality (Rump 2010,
Lemma 10.12, with ||S - fl(S)||_2 <= || |S - fl(S)| ||_2 <= ||E||_2 <= ||E||_inf),
lambda_max(fl(S)) bounded above by the Cholesky-residual bound of
certificate.lambda_min_lower_float. The rounding bounds are the a priori ones of
Rump (2010), (2.3) and Theorem 2.1: in rounding to nearest each product errs by a
relative u and, under underflow, by an absolute eta / 2, each sum by a relative u
and exactly under underflow, so a sum of N products errs by at most gamma_N times
the sum of their moduli plus N eta (certificate._dot_err). The face's T'ST (face.py)
is bounded the same way, T integer.

The same quantities give a lower bound: the dual function is at least
k - e_k + (1 + m) max(0, v'Sv / v'v) for any v (the Rayleigh quotient), at the
computed top eigenvector, with the rounding of v'Sv bounded as above. bound() is at
least the dual function, so the lower bound is below it.

certificate() decides with the floats and falls back on bound() only between them:
an upper bound at most zero closes the size; a lower bound above zero shows that
bound() would not close it either; otherwise bound() decides. Every value it returns
is an upper bound on the relaxation's maximum, and it closes every size bound()
closes.

Rump, S. M. (2010). Verification methods: Rigorous results using floating-point
    arithmetic. Acta Numerica 19, 287-449. Cambridge University Press.
    doi:10.1017/S096249291000005X.
"""
from __future__ import annotations

from fractions import Fraction as Q
from typing import NamedTuple, Optional

import numpy as np
import scipy.sparse as sp

from mlsynth.exceptions import MlsynthDataError

from .certificate import ETA, U, _dot_err, inflate, lambda_min_lower_float, round_down

#: The smallest normal double: a coefficient that rounds to at least this is within
#: a relative u of its double, since below it the absolute error eta / 2 is u times it.
NORMAL = float(np.finfo(float).tiny)


class FloatObjective(NamedTuple):
    """A in doubles, and a bound on its rounding: |A~ - A| <= EA entrywise."""
    Af: np.ndarray
    EA: np.ndarray


class FloatRows:
    """A Rows table as one sparse matrix over vec(Z) (column-major), transposed so a
    product with the multipliers sums each entry of Z's slack, with what the
    rounding bound needs: N, the most terms any entry sums, and on the face T, the
    most nonzeros in any column of T and its largest column sum. ``exact`` is False
    when a coefficient or right-hand side does not round to a normal double or an
    exact one; then the floats bound nothing."""

    def __init__(self, R):
        Me, Mi = R.matrices(1.0)
        self.n, self.n_eq, self.n_in = R.n, R.n_eq, R.n_in
        self.MT = sp.vstack([Me, Mi]).T.tocsr()
        self.aMT = abs(self.MT)
        self.N = int(np.diff(self.MT.indptr).max(initial=0))
        bs = list(R.beq) + list(R.bin)
        self.b = np.array([float(v) for v in bs], dtype=float)
        nz = np.abs(self.MT.data)
        self.exact = bool((nz.size == 0 or nz.min() >= NORMAL) and all(Q(float(v)) == v for v in set(bs)))
        self.face = R.face
        if R.face is not None:
            T = R.face.matrix().tocsc()
            self.T, self.aT = T, abs(T)
            self.c = int(np.diff(T.indptr).max(initial=0))
            self.tsum = float(np.asarray(self.aT.sum(axis=0)).max(initial=0.0))


def float_rows(R) -> FloatRows:
    """R's FloatRows, built once per Rows (a Rows is not changed after it is built)."""
    F = getattr(R, "_float_rows", None)
    if F is None:
        F = FloatRows(R)
        R._float_rows = F
    return F


def objective_float(A) -> FloatObjective:
    """A rounded to nearest, an entry past the largest double an infinity, and the
    rounding: u / (1 - u) |A~| for a normal result (|A - A~| <= u |A| <= u |A~| / (1 - u)),
    eta / 2 below it (Rump 2010, (2.3))."""
    def one(q):
        try:
            return float(q)
        except OverflowError:
            return np.inf if q > 0 else -np.inf
    Af = np.array([[one(q) for q in row] for row in A], dtype=float).reshape(len(A), len(A))
    return FloatObjective(Af, inflate(U / (1.0 - U) * np.abs(Af)) + ETA)


def slack_float(af: FloatObjective, F: FloatRows, y_eq, y_in):
    """fl(S), fl(k), and E, e_k with |fl(S) - S| <= E and |fl(k) - k| <= e_k, S and k
    those of slopefree._slack (the inequality multipliers clipped at zero).

    An entry of S is A's entry less a sum of at most N products y fl(v) (halved off
    the diagonal, exactly but for underflow), each coefficient v within a relative u
    of its double (FloatRows.exact), and one subtraction: gamma_{N+3} times the sum
    of the moduli, plus (N + 3) eta, plus A's own rounding. k is a sum of
    n_eq + n_in products of doubles."""
    y_eq, y_in = np.asarray(y_eq, dtype=float), np.asarray(y_in, dtype=float)
    if y_eq.shape != (F.n_eq,) or y_in.shape != (F.n_in,):
        raise MlsynthDataError(f"the multipliers have {y_eq.size} equality and {y_in.size} inequality entries; "
                               f"the rows have {F.n_eq} and {F.n_in}.")
    n = F.n
    y = np.concatenate([y_eq, np.maximum(y_in, 0.0)])
    ay = np.abs(y)
    W = (F.MT @ y).reshape(n, n, order="F")
    Wm = (F.aMT @ ay).reshape(n, n, order="F")
    A0 = np.zeros((n, n)); A0[1:, 1:] = af.Af
    EA0 = np.zeros((n, n)); EA0[1:, 1:] = af.EA
    S = A0 - (W + W.T) / 2
    E = inflate(EA0 + _dot_err(F.N + 3, np.abs(A0) + (Wm + Wm.T) / 2))
    k = float(y @ F.b)
    ek = inflate(_dot_err(y.size, float(ay @ np.abs(F.b))))
    return S, k, E, ek


def face_float(F: FloatRows, S, E):
    """fl(T'ST), symmetrised, and EM with |fl(T'ST) - T'S*T| <= EM for every S* within
    E of S. T'ST is summed in two stages of at most c products each, T integer, so
    its rounding is at most gamma_{2c+2} |T|'|S||T| plus c (3 + max column sum of
    |T|) eta (derived here from Rump 2010, (2.3) and Theorem 2.1); the symmetrising
    adds u of the result and eta / 2, and S* adds |T|'E|T|."""
    T, aT = F.T, F.aT
    Y = np.asarray((T.T @ S).T)                   # S T, as S is symmetric
    M = np.asarray(T.T @ Y)
    M = (M + M.T) / 2
    P = np.asarray(aT.T @ np.asarray((aT.T @ np.abs(S)).T))
    PE = np.asarray(aT.T @ np.asarray((aT.T @ E).T))
    EM = inflate(PE + _dot_err(2 * F.c + 2, P) + U * np.abs(M)) + F.c * (3.0 + F.tsum) * ETA
    return M, EM


def _lambda_up(S, E) -> Optional[Q]:
    """A rational at least lambda_max of every symmetric S* within E of S."""
    low = lambda_min_lower_float(-S)
    eS = inflate(float(E.sum(axis=1).max()))
    if low is None or not np.isfinite(eS):
        return None
    return Q(-low) + Q(eS)


def _lambda_low(S, E) -> Q:
    """A rational at most max(0, lambda_max) of every symmetric S* within E of S:
    the Rayleigh quotient v'S*v / v'v at the top eigenvector v of S, its numerator
    less the rounding of v'Sv (two stages of n products, |v_i| <= 1) and |v|'E|v|."""
    n = S.shape[0]
    _, V = np.linalg.eigh(S)
    v = V[:, -1]
    av = np.abs(v)
    num = float(v @ (S @ v))
    err = inflate(_dot_err(2 * n + 2, float(av @ (np.abs(S) @ av))) + float(av @ (E @ av))) + n * (n + 4) * ETA
    if not (np.isfinite(num) and np.isfinite(err)):
        return Q(0)
    low = Q(num) - Q(err)
    if low <= 0:
        return Q(0)
    return low / sum((Q(float(x)) ** 2 for x in v), Q(0))


def _round_up(x: Q) -> float:
    try:
        return -round_down(-x)
    except OverflowError:
        return np.inf


def _parts(af, R, y_eq, y_in):
    F = float_rows(R)
    S, k, E, ek = slack_float(af, F, y_eq, y_in)
    if not (F.exact and np.all(np.isfinite(S)) and np.all(np.isfinite(E)) and np.isfinite(k) and np.isfinite(ek)):
        return None
    return F, S, k, E, ek


def bound_float(af: FloatObjective, R, y_eq, y_in, m) -> float:
    """A double at least bound()'s dual function: k + e_k + (1 + m) max(0, lambda_up),
    and on the face (R.face) k + e_k + (1 + q) max(0, lambda_up(T'ST)), the face's
    alone when it is at most zero, else the smaller, as bound() combines them. +inf
    when the floats bound nothing (a multiplier or a sum that is not finite)."""
    parts = _parts(af, R, y_eq, y_in)
    if parts is None:
        return np.inf
    F, S, k, E, ek = parts
    kq = Q(k) + Q(ek)
    lu = _lambda_up(S, E)
    full = None if lu is None else kq + (1 + m) * max(Q(0), lu)
    if F.face is not None:
        M, EM = face_float(F, S, E)
        lf = _lambda_up(M, EM) if np.all(np.isfinite(M)) and np.all(np.isfinite(EM)) else None
        if lf is not None:
            face = kq + F.face.trace * max(Q(0), lf)
            full = face if face <= 0 or full is None else min(full, face)
    return np.inf if full is None else _round_up(full)


def bound_lower(af: FloatObjective, R, y_eq, y_in, m) -> float:
    """A double at most bound() at these multipliers: k - e_k + (1 + m) max(0, the
    Rayleigh bound), and on the face the smaller of that and the face's. -inf when
    the floats bound nothing."""
    parts = _parts(af, R, y_eq, y_in)
    if parts is None:
        return -np.inf
    F, S, k, E, ek = parts
    kq = Q(k) - Q(ek)
    low = kq + (1 + m) * _lambda_low(S, E)
    if F.face is not None:
        M, EM = face_float(F, S, E)
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(EM))):
            return -np.inf
        low = min(low, kq + F.face.trace * _lambda_low(M, EM))
    try:
        return round_down(low)
    except OverflowError:
        return -np.inf if low < 0 else float(np.finfo(float).max)


def bound(A, R, y_eq, y_in, m) -> Optional[Q]:
    """slopefree.bound, the exact certificate, looked up when called (slopefree
    imports this module)."""
    from . import slopefree
    return slopefree.bound(A, R, y_eq, y_in, m)


def certificate(A, R, y_eq, y_in, m, af: Optional[FloatObjective] = None) -> Optional[Q]:
    """An upper bound on u'Au over the relaxation of R at these multipliers, at most
    zero whenever bound() is: the float bound when it is at most zero, the float
    bound when the lower bound is above zero (bound() would be too), and bound()
    otherwise. None only when bound() finds no factor and the floats bound nothing."""
    af = objective_float(A) if af is None else af
    up = bound_float(af, R, y_eq, y_in, m)
    if up <= 0:
        return Q(up)
    if np.isfinite(up) and bound_lower(af, R, y_eq, y_in, m) > 0:
        return Q(up)
    exact = bound(A, R, y_eq, y_in, m)
    if exact is None and np.isfinite(up):  # pragma: no cover - bound() factors every finite S (its own pragma)
        return Q(up)
    return exact
