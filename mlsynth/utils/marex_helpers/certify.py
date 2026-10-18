"""A certified optimality gap for MAREX's exact backend.

The search in :mod:`mlsynth.utils.marex_helpers.exact` prices candidates and fits
them in floating point, so its design is optimal up to rounding that nothing
bounds. :func:`certify_cluster` bounds it. It returns two numbers for one
cluster, ``lower <= f* <= f(design) <= upper``, where ``f*`` is the optimal
objective and ``f(design)`` the objective of the returned design, both evaluated
exactly on the stored data. The design is optimal to within ``upper - lower``.

The problem the certificate refers to. ``B`` and ``A`` are taken as the doubles
they are, and ``g(S)`` is the exact minimum of ``||A - B_S w||^2 + a'w`` over the
simplex on ``S``. Because the weights sum to one this is ``||Bc w||^2 + a'w``
with ``Bc = B - A 1'`` computed exactly, not in floating point.

Exact arithmetic. Every finite double is an integer over a power of two, so a
matrix of doubles is exactly an integer matrix over one power of two, and sums
and products of such matrices stay in that form. :class:`Dyadic` holds one as
Python integers. Nothing is rounded until a final rational is rounded to a
double, and that rounding is directed (:func:`round_down`, :func:`round_up`).

One fit, exactly (:meth:`ExactData.interval`). Any point of the simplex gives an
upper bound: the weights a fit returned, clipped at zero and rescaled to sum to
one exactly. Any vector ``y`` gives a lower bound, the Wolfe dual of the fit:

    ||Bc w||^2 + a'w >= 2 y'Bc w - ||y||^2 + a'w >= min_j (2 y'b_j + a_j) - ||y||^2

on the simplex. At ``y = Bc w`` with ``w`` the rescaled weights the two differ
by the fit's Frank-Wolfe gap, which is zero at the optimum. Both are computed
exactly. Each costs ``T`` times the number of columns in integer operations.

Every candidate, in batches (:meth:`ExactData.treated_bounds`). Exact fits of
every candidate would cost far more than the search. The lower bound above at
``y = Bc c`` reads the data only through the Gram matrix,
``min_j (2 (G c)_j + a_j) - c'G c``, and holds for every ``c``, so ``c`` can be
the closed-form minimiser the search's bound computes in floating point. What
needs bounding is the arithmetic on ``G``. With ``u`` the unit roundoff,
``gamma_k = k u / (1 - k u)`` and ``eta`` the smallest subnormal (Higham 2002,
sec. 3.1), the floating-point Gram ``Gf`` of the floating-point difference
``Bf = fl(B - A 1')`` satisfies, entrywise,

    |Gf - G| <= gamma_{T+2} (|Bf|'|Bf| + 2 T eta) / (1 - gamma_T) + 2 T eta:

``gamma_T`` for the inner products, ``2u`` more for the rounding of the
difference, and the denominator for computing ``|Bf|'|Bf|`` itself. The product
``Gf c`` adds ``gamma_m |Gf||c|``, and ``c'(Gf c)`` adds ``gamma_m |c|'|Gf c|``.
Each error is carried to the bound, the total is doubled to cover the rounding
of the error terms themselves, and a floor of ``32 (m + 8) (1 + |c|_1)^2 eta``
covers their underflow. The result is rounded down. A block whose solve fails,
or any non-finite result, gets the trivial bound ``min_j a_j`` (``0`` without
a linear term), since ``||Bc w||^2 >= 0``.

One cluster (:func:`certify_cluster`). Naming ``S`` splits the objective into
``g(S) + h(C(S))``, and the control pool ``C(S)`` is part of the cluster, so
``h(C(S)) >= h(cluster)``: a fit over more markets is no worse. Every admissible
candidate is priced by its certified treated bound plus that floor. A candidate
priced above the design's certified upper bound cannot beat the design. The
rest are fitted and bracketed exactly; a candidate whose exact lower bound still
does not clear the upper bound is reported as unresolved, and the certificate's
lower bound is the smallest exact lower bound among the design and the
unresolved candidates. An exact tie is unresolved by construction.

The weakly-targeted design couples the treated and control fits, so naming ``S``
does not split it into the two fits above, and the certificate refuses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..syndes_helpers.exact import SubsetFilter
from .exact import CHUNK, ClusterDesign, _chunks, side_fit

_U = 2.0 ** -53
_ETA = 2.0 ** -1074
_MAX = np.finfo(float).max


# ----------------------------------------------------------------------
# exact arithmetic
# ----------------------------------------------------------------------


def _ints(values: Iterable) -> np.ndarray:
    return np.array([int(v) for v in values], dtype=object)


class Dyadic:
    """An exact dyadic array: the value is ``N / 2**e`` with ``N`` integers."""

    __slots__ = ("N", "e")

    def __init__(self, N: np.ndarray, e: int):
        self.N, self.e = N, int(e)

    @classmethod
    def of(cls, a) -> "Dyadic":
        """The doubles of ``a``, exactly."""
        a = np.asarray(a, dtype=float)
        if not np.all(np.isfinite(a)):
            raise MlsynthDataError(
                "A certificate needs finite data; the panel holds a NaN or an "
                "infinite value.")
        pairs = [float(x).as_integer_ratio() for x in a.ravel()]
        e = max((d.bit_length() - 1 for _, d in pairs), default=0)
        N = _ints(p << (e - (d.bit_length() - 1)) for p, d in pairs)
        return cls(N.reshape(a.shape), e)

    def _at(self, e: int) -> np.ndarray:
        return self.N * (1 << (e - self.e)) if e > self.e else self.N

    def __sub__(self, other: "Dyadic") -> "Dyadic":
        e = max(self.e, other.e)
        return Dyadic(self._at(e) - other._at(e), e)

    def __matmul__(self, other: "Dyadic") -> "Dyadic":
        return Dyadic(np.asarray(self.N.dot(other.N), dtype=object),
                      self.e + other.e)

    def __getitem__(self, idx) -> "Dyadic":
        return Dyadic(self.N[idx], self.e)

    def frac(self, *idx) -> Fraction:
        """One entry as a rational."""
        return Fraction(int(self.N[idx]), 1 << self.e)

    def total(self) -> Fraction:
        """The sum of the entries as a rational."""
        return Fraction(int(sum(self.N.ravel().tolist())), 1 << self.e)


def round_down(q: Fraction) -> float:
    """The largest double not above the rational ``q``."""
    try:
        x = float(q)
    except OverflowError:
        return _MAX if q > 0 else -math.inf
    if Fraction(x) > q:
        x = math.nextafter(x, -math.inf)
    return x


def round_up(q: Fraction) -> float:
    """The smallest double not below the rational ``q``."""
    return -round_down(-q)


def _gamma(k: int) -> float:
    return k * _U / (1.0 - k * _U)


# ----------------------------------------------------------------------
# the bounds
# ----------------------------------------------------------------------


class ExactData:
    """One cluster's data, exactly and in floating point.

    Parameters
    ----------
    B : np.ndarray, shape (T, n)
        The markets as columns.
    A : np.ndarray, shape (T,)
        The target.
    """

    def __init__(self, B: np.ndarray, A: np.ndarray):
        self.B = np.asarray(B, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.T, self.n = self.B.shape
        self.exact = Dyadic.of(self.B) - Dyadic.of(self.A[:, None])
        self.Bf = self.B - self.A[:, None]
        self.Gf = self.Bf.T @ self.Bf
        self.Gabs = np.abs(self.Bf).T @ np.abs(self.Bf)

    def interval(self, cols: Sequence[int], w: np.ndarray,
                 linear: Optional[np.ndarray] = None
                 ) -> Tuple[Fraction, Fraction]:
        """Exact bounds on the fit over ``cols``, from the weights ``w``.

        ``linear`` is indexed by the cluster's markets, as in the search.
        Weights with no positive entry are replaced by the uniform point.
        """
        cols = list(cols)
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        if not w.sum() > 0:
            w = np.ones(len(cols))
        W = Dyadic.of(w)
        Bs = self.exact[:, cols]
        Y = Bs @ W
        s = W.total()
        P = Y @ Bs
        yy = (Y @ Y).frac() / (s * s)
        a = ([Fraction(0)] * len(cols) if linear is None
             else [Fraction(float(linear[j])) for j in cols])
        lo = min(2 * P.frac(i) / s + a[i] for i in range(len(cols))) - yy
        hi = yy + sum(Fraction(float(w[i])) * a[i] for i in range(len(cols))) / s
        return max(lo, min(a)), hi

    def treated_bounds(self, sets: np.ndarray,
                       linear: Optional[np.ndarray] = None) -> np.ndarray:
        """A certified lower bound on the fit of every row of ``sets``."""
        sets = np.asarray(sets, dtype=np.int64)
        n_sub, m = sets.shape
        out = np.empty(n_sub)
        lin = None if linear is None else np.asarray(linear, dtype=float)
        T = self.T
        g_G = _gamma(T + 2) / (1.0 - _gamma(T))
        g_m = _gamma(m)
        under = 2.0 * T * _ETA
        for lo in range(0, n_sub, CHUNK):
            S = sets[lo:lo + CHUNK]
            b = len(S)
            G = self.Gf[S[:, :, None], S[:, None, :]]
            Ga = self.Gabs[S[:, :, None], S[:, None, :]]
            a = np.zeros((b, m)) if lin is None else lin[S]
            rhs = np.ones((b, m, 1 if lin is None else 2))
            if lin is not None:
                rhs[:, :, 1] = a
            with np.errstate(all="ignore"):
                try:
                    x = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    x = np.full_like(rhs, np.nan)
                    for i in range(b):
                        try:
                            x[i] = np.linalg.solve(G[i], rhs[i])
                        except np.linalg.LinAlgError:
                            pass
                s = x[:, :, 0].sum(axis=1)
                if lin is None:
                    c = x[:, :, 0] / s[:, None]
                else:
                    mu = (2.0 + x[:, :, 1].sum(axis=1)) / s
                    c = (mu[:, None] * x[:, :, 0] - x[:, :, 1]) / 2.0
                ok = (s > 0) & np.all(np.isfinite(c), axis=1)
                c = np.where(ok[:, None], c, 0.0)
                ac = np.abs(c)
                l1 = ac.sum(axis=1)
                v = np.einsum("bij,bj->bi", G, c)
                q = np.einsum("bi,bi->b", c, v)
                err_G = g_G * (Ga + under) + under
                err_v = (g_m * np.einsum("bij,bj->bi", np.abs(G), ac)
                         + np.einsum("bij,bj->bi", err_G, ac) + 2.0 * m * _ETA)
                r = 2.0 * v + a
                mn = r.min(axis=1)
                err_r = (2.0 * err_v + _U * np.abs(r)).max(axis=1)
                err_q = (g_m * np.einsum("bi,bi->b", ac, np.abs(v))
                         + np.einsum("bi,bi->b", ac, err_v) + 2.0 * m * _ETA)
                err = err_r + err_q + _U * (np.abs(mn) + np.abs(q))
                floor = 32.0 * (m + 8) * (1.0 + l1) ** 2 * _ETA
                lb = np.nextafter((mn - q) - (2.0 * err + floor), -np.inf)
            trivial = a.min(axis=1) if m else np.zeros(b)
            good = ok & np.isfinite(lb)
            out[lo:lo + b] = np.where(good, np.maximum(lb, trivial), trivial)
        return out


# ----------------------------------------------------------------------
# one cluster
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterCertificate:
    """Certified bounds for one cluster's design.

    ``lower <= f* <= f(design) <= upper``, with ``f*`` the cluster's optimal
    objective. ``unresolved`` lists the treated sets, other than the design's,
    whose exact lower bound does not clear ``upper``: each might be as good as
    the design. ``fitted`` counts the candidates the certificate fitted.
    """

    lower: float
    upper: float
    unresolved: Tuple[Tuple, ...]
    fitted: int

    @property
    def gap(self) -> float:
        """``upper - lower``, rounded up."""
        return round_up(Fraction(self.upper) - Fraction(self.lower))


def certify_cluster(
    B: np.ndarray,
    A: np.ndarray,
    sizes: Sequence[int],
    filt: SubsetFilter,
    design: ClusterDesign,
    *,
    linear_treated: Optional[np.ndarray] = None,
    linear_control: Optional[np.ndarray] = None,
    donor_exclusion: Sequence[Tuple[int, int]] = (),
    beta: Optional[float] = None,
) -> ClusterCertificate:
    """Certify ``design`` as the optimum of one cluster's search.

    Takes :func:`~mlsynth.utils.marex_helpers.exact.search_cluster`'s
    arguments and the design it returned.

    Raises
    ------
    MlsynthConfigError
        For the weakly-targeted design (``beta`` given).
    """
    if beta is not None:
        raise MlsynthConfigError(
            "The certificate splits each candidate into a treated fit and a "
            "control fit; the weakly-targeted design couples them, so it "
            "cannot be certified this way.")
    data = ExactData(B, A)
    n = data.n
    lin_t = None if linear_treated is None else np.asarray(linear_treated, float)
    lin_c = None if linear_control is None else np.asarray(linear_control, float)
    excluded = {}
    for i, j in donor_exclusion:
        excluded.setdefault(int(i), set()).add(int(j))

    lo_g, hi_g = data.interval(design.treated, design.w, lin_t)
    lo_h, hi_h = data.interval(design.control, design.v, lin_c)
    upper = hi_g + hi_h
    lower = lo_g + lo_h
    cut = round_up(upper)

    everyone = list(range(n))
    _, v_all = side_fit(data.B, data.A, everyone, lin_c)
    h_floor = data.interval(everyone, v_all, lin_c)[0]
    h_down = round_down(h_floor)

    unresolved: List[Tuple[int, ...]] = []
    fitted = 0
    for block in _chunks(n, sizes, filt):
        priced = np.nextafter(data.treated_bounds(block, lin_t) + h_down, -np.inf)
        for i in np.flatnonzero(priced <= cut):
            S = tuple(int(j) for j in block[i])
            if S == tuple(design.treated):
                continue
            cols = list(S)
            _, w = side_fit(data.B, data.A, cols,
                            None if lin_t is None else lin_t[cols])
            lo_S = data.interval(cols, w, lin_t)[0]
            if lo_S + h_floor > upper:
                continue
            drop = set(S)
            for j in S:
                drop |= excluded.get(j, set())
            control = [j for j in everyone if j not in drop]
            if not control:
                continue
            _, v = side_fit(data.B, data.A, control,
                            None if lin_c is None else lin_c[control])
            lo_S += data.interval(control, v, lin_c)[0]
            fitted += 1
            if lo_S <= upper:
                unresolved.append(S)
                lower = min(lower, lo_S)
    return ClusterCertificate(lower=round_down(lower), upper=cut,
                              unresolved=tuple(sorted(unresolved)),
                              fitted=fitted)
