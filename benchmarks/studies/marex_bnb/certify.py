"""Bounds that floating point cannot break: exact arithmetic on dyadic matrices.

The construction is that of Neumaier and Shcherbina (2004) for linear programs
and Jansson, Chaykin and Keil (2007) for semidefinite ones: any dual point gives
a valid bound, so a solver's approximate dual needs only to be evaluated with
its rounding under control. They control it with interval arithmetic; here the
arithmetic is exact.

Every finite double is a dyadic rational, an integer over a power of two. A
matrix of doubles is therefore exactly an integer matrix over one power of two,
and sums and products of such matrices stay in that form. ``Dy`` holds one, as
an object array of Python integers and an exponent, so every operation here is
exact: no rounding happens anywhere until a final rational is rounded to a
double, and that rounding is directed (``down`` for a lower bound, ``up`` for an
upper bound).

The problem the certificates refer to is the stored data. The centred columns
``Bc`` are taken as given doubles, and ``g(S)`` is the exact minimum of
``||Bc w||^2`` over the simplex on ``S``. Every bound below is a bound on that
exact number.

- ``side_lower``: for any vector ``y``, ``||Bc w||^2 >= 2 y'Bc w - ||y||^2``,
  and on the simplex ``y'Bc w >= min_j y'b_j``. So ``2 min_j y'b_j - ||y||^2``
  is a lower bound on ``g(S)`` for every ``y``, the dual of the fit in Wolfe's
  (1976) nearest-point problem. At
  ``y = Bc w`` with ``w`` optimal it equals ``g(S)``; a computed ``w`` gives a
  bound below ``g(S)`` by about the fit's own error.
- ``side_upper``: the exact objective at the computed weights after an exact
  renormalisation onto the simplex, ``w'Gw / (sum w)^2``. Any point of the
  simplex gives an upper bound.
- ``lambda_min_lower``: a lower bound on the smallest eigenvalue of an exact
  symmetric matrix ``S``. A floating-point Cholesky factor ``L`` of
  ``S - sigma I`` is exact as a dyadic matrix, so the residual
  ``R = S - sigma I - L L'`` is computed exactly. Then
  ``lambda_min(S) >= sigma + lambda_min(R) >= sigma - max_i sum_j |R_ij|`` by
  Weyl and Gershgorin, whatever rounding produced ``L``.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def _ints(values):
    return np.array([int(v) for v in values], dtype=object)


class Dy:
    """An exact dyadic matrix: the value is ``A / 2**e`` with ``A`` integers."""

    __slots__ = ("A", "e")

    def __init__(self, A, e):
        self.A, self.e = A, int(e)

    @classmethod
    def of(cls, a):
        a = np.asarray(a, dtype=float)
        if not np.all(np.isfinite(a)):
            raise ValueError("a certificate needs finite data")
        pairs = [float(x).as_integer_ratio() for x in a.ravel()]
        e = max((d.bit_length() - 1 for _, d in pairs), default=0)
        A = _ints(p << (e - (d.bit_length() - 1)) for p, d in pairs)
        return cls(A.reshape(a.shape), e)

    @classmethod
    def ints(cls, a):
        return cls(_ints(np.asarray(a).ravel()).reshape(np.shape(a)), 0)

    def _at(self, e):
        return self.A * (1 << (e - self.e)) if e > self.e else self.A

    def __add__(self, other):
        e = max(self.e, other.e)
        return Dy(self._at(e) + other._at(e), e)

    def __sub__(self, other):
        e = max(self.e, other.e)
        return Dy(self._at(e) - other._at(e), e)

    def __neg__(self):
        return Dy(-self.A, self.e)

    def __matmul__(self, other):
        return Dy(np.asarray(self.A.dot(other.A), dtype=object),
                  self.e + other.e)

    def __getitem__(self, idx):
        return Dy(self.A[idx], self.e)

    @property
    def T(self):
        return Dy(self.A.T, self.e)

    @property
    def shape(self):
        return self.A.shape

    def times(self, x):
        """Multiply by the double ``x``, exactly."""
        p, d = float(x).as_integer_ratio()
        return Dy(self.A * p, self.e + d.bit_length() - 1)

    def frac(self, *idx):
        return Fraction(int(self.A[idx]), 1 << self.e)

    def sum(self):
        return Fraction(int(sum(self.A.ravel().tolist())), 1 << self.e)

    def float(self):
        """The nearest doubles, for computing a factorisation; not exact."""
        d = 1 << self.e
        return np.array([int(v) / d for v in self.A.ravel()],
                        dtype=float).reshape(self.shape)


def down(q):
    """The largest double not above the rational ``q``."""
    x = float(q)
    if Fraction(x) > q:
        x = math.nextafter(x, -math.inf)
    return x


def up(q):
    """The smallest double not below the rational ``q``."""
    x = float(q)
    if Fraction(x) < q:
        x = math.nextafter(x, math.inf)
    return x


def eye(k):
    return Dy.ints(np.eye(k, dtype=np.int64))


def lambda_min_lower(S, tries=40):
    """A rational lower bound on ``lambda_min`` of the exact symmetric ``S``."""
    Sf = S.float()
    k = Sf.shape[0]
    lam = float(np.linalg.eigvalsh(Sf)[0])
    size = float(np.abs(Sf).sum(axis=1).max()) or 1.0
    step = max(abs(lam) * 1e-12, 64 * k * np.finfo(float).eps * size)
    sigma = lam - step
    for _ in range(tries):
        try:
            L = np.linalg.cholesky(Sf - sigma * np.eye(k))
        except np.linalg.LinAlgError:
            step *= 4
            sigma = lam - step
            continue
        Ld = Dy.of(L)
        R = S - eye(k).times(sigma) - Ld @ Ld.T
        row = max(sum(abs(int(v)) for v in R.A[i]) for i in range(k))
        return Fraction(sigma) - Fraction(row, 1 << R.e)
    raise RuntimeError("no Cholesky factor found for the shifted matrix")


class Panel:
    """The exact data of one panel: ``Bc`` as stored and its exact Gram matrix."""

    def __init__(self, Bc):
        self.Bc = np.asarray(Bc, dtype=float)
        self.BcD = Dy.of(self.Bc)
        self.G = self.BcD.T @ self.BcD
        # floating-point Gram and its absolute counterpart, for vectorised
        # pricing: |Gf - G| <= gamma_T |Bc|'|Bc| entrywise
        self.Gf = self.Bc.T @ self.Bc
        self.Gabs = np.abs(self.Bc).T @ np.abs(self.Bc)

    def side_lower(self, cols, w):
        """``2 min_j y'b_j - ||y||^2`` at ``y = Bc w``, a bound on ``g(cols)``."""
        cols = list(cols)
        y = Dy.of(self.Bc[:, cols] @ np.asarray(w, dtype=float))
        p = (y.T @ self.BcD[:, cols])
        smallest = min(p.A.tolist())
        lb = Fraction(2 * smallest, 1 << p.e) - (y.T @ y).sum()
        return max(lb, Fraction(0))

    def side_upper(self, cols, w):
        """``w'Gw / (sum w)^2`` with ``w`` clipped at zero, an upper bound."""
        cols = list(cols)
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        if not w.sum() > 0:
            w = np.ones(len(cols))
        wd = Dy.of(w)
        G = self.G[np.ix_(cols, cols)]
        num = (wd.T @ G @ wd).sum()
        den = wd.sum()
        return num / (den * den)
