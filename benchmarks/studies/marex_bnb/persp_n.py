"""The perspective-plus-nonnegativity bound ("persp+N"): one PSD block per side.

Reading the literature on quadratics with indicators (the readers' reports are
summarised in ``results/persp_n.md``) left one relaxation standing that is
positive where every decomposition is zero. Per side, with ``U`` the markets a
completion can give that side's weight to:

- ``M = [[1, x'], [x, X]] >= 0`` on ``(1, x)``, ``x`` the side's weights on
  ``U``: the global PSD coupling of the second moments;
- ``X >= 0`` elementwise and ``X 1 = x``, ``1'x = 1``: the products of
  ``x >= 0`` and of the simplex;
- ``X_jj z_j >= x_j^2`` for every free market (the perspective, with ``1 - z_j``
  on the control side): the only place the indicators meet the second moments.
  With ``sum z = m`` it implies ``sum X_jj >= 1/m``, the concentration of a
  design's weights on ``m`` markets;
- ``x_j <= z_j`` (treated) and ``x_j <= 1 - z_j`` (control), ``0 <= z <= 1``,
  ``sum_F z = r``.

The pieces come from the literature read for this study (``results/persp_n.md``
lists it in full). The PSD block with the perspective blocks
``[[ind_j, x_j], [x_j, X_jj]] >= 0`` is the semidefinite form of Dong, Chen and
Linderoth, eq. (24) of Wei, Gómez and Küçükyavuz (2022); with ``X 1 = x`` it is
the primal of the best-diagonal program of Zheng, Sun and Li (2014). The
perspective is that of Frangioni and Gentile (2006) and Günlük and Linderoth
(2010). ``X >= 0``, which those forms lack and which carries most of the bound
here, is the entrywise half of the doubly nonnegative relaxation of Burer's
(2009) completely positive form. The indicators never enter a
PSD block, so the blocks have ``|U| + 1`` rows where the treated lift has
about ``2 |U|``. Every constraint holds at
``M = (1; x)(1; x)'`` for a design, so the minimum of ``<G, X>`` (plus the
other side's) is a lower bound on the subtree's minimum.

``side="treated"`` lifts the treated side only and adds the control fit's
Wolfe-dual certificate over every market a completion could use as a control,
exactly as ``treated_lift_geolift`` does, so the two are compared on the same
footing. ``side="both"`` lifts both sides with a shared ``z``.

``bound`` reports the solver's value. ``certified`` turns the solver's
multipliers into a bound that no rounding can break: the rigorous bound of
Jansson, Chaykin and Keil (2007) for semidefinite programs, with exact
arithmetic in place of their interval arithmetic. For one side with block
``M = [[1, x'], [x, X]]``, multipliers ``y0`` on ``M_00 = 1``, ``H >= 0`` on
``X >= 0``, ``alpha`` on ``1'x = 1``, ``beta`` on ``X 1 = x``, ``mu >= 0`` on
``ind_j - x_j >= 0`` and ``Lambda_j >= 0`` (2x2 PSD) on the perspective block
``[[ind_j, x_j], [x_j, X_jj]]``, every design satisfies

    objective = L + y0 (M_00 - 1) + <H, X> + ... + sum <Lambda_j, Q_j> >= L,

because each subtracted term is zero or nonnegative there. ``L`` is linear:
``L = const + <S, M> + c'z`` with ``S`` collecting the coefficients of ``M``.
A design has ``tr M = 1 + |x|^2 <= 2`` and ``z`` in ``{0 <= z <= 1, sum z = r}``,
so

    objective >= const + 2 min(0, lambda_min(S)) + (sum of the r smallest c_j).

The multipliers are the solver's, clipped to their signs (``Lambda_j`` made PSD
by shrinking its off-diagonal), every number is a double and so a dyadic
rational, ``S`` is assembled exactly from the panel's exact Gram matrix, and
``lambda_min`` is bounded below exactly (``certify.lambda_min_lower``).
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

import cvxpy as cp
import numpy as np

from benchmarks.studies.marex_bnb.certify import Dy, down, lambda_min_lower


def _side(Gs, U, z_of, lin=None):
    """One side's block, its constraints and its objective."""
    k = len(U)
    M = cp.Variable((k + 1, k + 1), symmetric=True)
    x, X = M[0, 1:], M[1:, 1:]
    cons = [M >> 0, M[0, 0] == 1, X >= 0, cp.sum(x) == 1, X @ np.ones(k) == x]
    for i, j in enumerate(U):
        ind = z_of(j)
        if ind is not None:
            cons += [x[i] <= ind, cp.quad_over_lin(x[i], ind) <= X[i, i]]
    obj = cp.trace(Gs[np.ix_(U, U)] @ X)
    if lin is not None:
        obj = obj + lin[U] @ x
    return cons, obj, x


def bound(Bc, m, I, O, F, side="both", lin_t=None, lin_c=None, solver="CLARABEL"):
    """The persp+N value at a node, rescaled to ``Bc``'s units.

    Returns ``(value, status, seconds)``. ``side="treated"`` returns the treated
    block's value only; the caller adds the control certificate.
    """
    n = Bc.shape[1]
    G = Bc.T @ Bc
    scale = 2.0 ** round(math.log2(float(np.max(np.diag(G)))))
    Gs = G / scale
    r = m - len(I)
    z = cp.Variable(len(F)) if F else None
    pos = {j: i for i, j in enumerate(F)}
    cons = [] if z is None else [z >= 0, z <= 1, cp.sum(z) == r]
    zt = lambda j: z[pos[j]] if j in pos else None             # noqa: E731
    zc = lambda j: 1 - z[pos[j]] if j in pos else None         # noqa: E731
    Uw = sorted(I + F)
    c1, obj, _ = _side(Gs, Uw, zt, None if lin_t is None else lin_t / scale)
    cons += c1
    if side == "both":
        Uv = sorted(O + F)
        c2, o2, _ = _side(Gs, Uv, zc, None if lin_c is None else lin_c / scale)
        cons += c2
        obj = obj + o2
    t0 = time.perf_counter()
    p = cp.Problem(cp.Minimize(obj), cons)
    try:
        p.solve(solver=solver)
    except cp.error.SolverError:
        return float("nan"), "solver_error", time.perf_counter() - t0
    return float(p.value) * scale, p.status, time.perf_counter() - t0


# ----------------------------------------------------------------------
# the certified bound
# ----------------------------------------------------------------------


class _Side:
    """One side's block and the constraint handles whose duals certify it."""

    def __init__(self, U, F, z, pos, control):
        k = len(U)
        self.U, self.k, self.control = U, k, control
        self.M = cp.Variable((k + 1, k + 1), symmetric=True)
        x, X = self.M[0, 1:], self.M[1:, 1:]
        self.c00 = self.M[0, 0] == 1
        self.cnn = X >= 0
        self.csum = cp.sum(x) == 1
        self.crow = X @ np.ones(k) == x
        self.free = [(i, pos[j]) for i, j in enumerate(U) if j in pos]
        self.clin, self.cper = [], []
        for i, f in self.free:
            ind = (1 - z[f]) if control else z[f]
            self.clin.append(ind - x[i] >= 0)
            self.cper.append(cp.bmat([[cp.reshape(ind, (1, 1)), cp.reshape(x[i], (1, 1))],
                                      [cp.reshape(x[i], (1, 1)),
                                       cp.reshape(X[i, i], (1, 1))]]) >> 0)
        self.constraints = ([self.M >> 0, self.c00, self.cnn, self.csum, self.crow]
                            + self.clin + self.cper)


def _psd2(L):
    """A 2x2 symmetric matrix made PSD by clipping the diagonal and shrinking
    the off-diagonal; any PSD multiplier keeps the certificate valid.

    ``b^2 <= a c`` is checked in exact arithmetic: ``b`` is clipped to the
    floating-point ``sqrt(a c)`` and moved toward zero only the few units in
    the last place that rounding can put it over.
    """
    a, b, c = max(float(L[0, 0]), 0.0), float(L[0, 1] + L[1, 0]) / 2, max(float(L[1, 1]), 0.0)
    if Fraction(b) ** 2 <= Fraction(a) * Fraction(c):
        return a, b, c
    b = math.copysign(math.sqrt(a * c), b)
    while b != 0.0 and Fraction(b) ** 2 > Fraction(a) * Fraction(c):
        b = math.nextafter(b, 0.0)
    return a, b, c


def _dy(entries):
    """An exact dyadic matrix from a nested list of dyadic Fractions."""
    flat = [q for row in entries for q in row]
    e = max((q.denominator.bit_length() - 1 for q in flat), default=0)
    N = np.array([int(q.numerator) << (e - (q.denominator.bit_length() - 1)) for q in flat],
                 dtype=object).reshape(len(entries), len(entries[0]))
    return Dy(N, e)


def certified(pan, m, I, O, F, side="both", lin_t=None, lin_c=None, solver="CLARABEL",
              with_z=False):
    """``(certified lower bound, solver value, status, solve s, certify s)``,
    plus the relaxation's indicator values ``{j: z_j}`` when ``with_z``.

    ``pan`` is a ``certify.Panel``; the bound refers to its exact Gram matrix.
    ``side="treated"`` certifies the treated block alone (the caller adds a
    control certificate).
    """
    G = pan.G
    gmax = max(G.frac(i, i) for i in range(G.shape[0]))
    sh = round(math.log2(float(gmax)))
    scale = Fraction(2) ** sh
    Gs = (pan.Gf / float(scale))
    r = m - len(I)
    pos = {j: i for i, j in enumerate(F)}
    z = cp.Variable(len(F)) if F else None
    cons = [] if z is None else [z >= 0, z <= 1, cp.sum(z) == r]
    sides = [_Side(sorted(I + F), F, z, pos, False)]
    lins = [None if lin_t is None else np.asarray(lin_t, float)]
    if side == "both":
        sides.append(_Side(sorted(O + F), F, z, pos, True))
        lins.append(None if lin_c is None else np.asarray(lin_c, float))
    obj = 0
    for s, lin in zip(sides, lins):
        X, x = s.M[1:, 1:], s.M[0, 1:]
        obj = obj + cp.trace(Gs[np.ix_(s.U, s.U)] @ X)
        if lin is not None:
            obj = obj + (lin[s.U] / float(scale)) @ x
        cons += s.constraints
    t0 = time.perf_counter()
    p = cp.Problem(cp.Minimize(obj), cons)
    p.solve(solver=solver)
    t_solve = time.perf_counter() - t0

    t0 = time.perf_counter()
    const = Fraction(0)
    cz = [Fraction(0)] * len(F)
    for s, lin in zip(sides, lins):
        k, U = s.k, s.U
        y0 = -float(s.c00.dual_value)
        alpha = -float(s.csum.dual_value)
        beta = -np.asarray(s.crow.dual_value, dtype=float).reshape(k)
        H = np.asarray(s.cnn.dual_value, dtype=float).reshape(k, k)
        H = np.clip((H + H.T) / 2, 0.0, None)
        H = np.triu(H) + np.triu(H, 1).T
        const += Fraction(y0) + Fraction(alpha)
        S = [[Fraction(0)] * (k + 1) for _ in range(k + 1)]
        S[0][0] = -Fraction(y0)
        for a in range(k):
            for b in range(k):
                S[a + 1][b + 1] = (G.frac(U[a], U[b]) / scale - Fraction(H[a, b])
                                   - (Fraction(beta[a]) + Fraction(beta[b])) / 2)
        xc = [-Fraction(alpha) + Fraction(beta[a]) for a in range(k)]
        if lin is not None:
            xc = [xc[a] + Fraction(float(lin[U[a]])) / scale for a in range(k)]
        for (i, f), cl, cq in zip(s.free, s.clin, s.cper):
            mu = Fraction(max(float(cl.dual_value), 0.0))
            a_, b_, c_ = _psd2(np.asarray(cq.dual_value, dtype=float))
            a_, b_, c_ = Fraction(a_), Fraction(b_), Fraction(c_)
            xc[i] += mu - 2 * b_
            S[i + 1][i + 1] -= c_
            if s.control:                       # ind = 1 - z
                const -= mu + a_
                cz[f] += mu + a_
            else:                               # ind = z
                cz[f] -= mu + a_
        for a in range(k):
            S[0][a + 1] = S[a + 1][0] = xc[a] / 2
        lam = lambda_min_lower(_dy(S))
        const += 2 * min(Fraction(0), lam)
    if F:
        const += sum(sorted(cz)[:r])
    lb = const * scale
    out = (down(lb), float(p.value) * float(scale), p.status, t_solve, time.perf_counter() - t0)
    if with_z:
        zv = np.zeros(len(F)) if z is None or z.value is None else np.asarray(z.value)
        out = out + ({j: float(zv[pos[j]]) for j in F},)
    return out
