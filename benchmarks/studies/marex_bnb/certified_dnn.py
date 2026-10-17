"""The doubly nonnegative node bound, on its face and with a certified value.

``sdp_bound`` lifts ``x = (w, v)`` to ``Y = [[1, x'], [x, X]]``. Its
constraints ``sum w = 1`` and ``X_ww 1 = w``, ``X_vw 1 = v`` make
``(-1; 1_w; 0)`` a null vector of ``Y``, and likewise ``(-1; 0; 1_v)``, so the
lifted matrix is singular at every feasible point and the program has no
interior. Interior-point solvers then stop at "optimal_inaccurate", and the dual
optimum need not be attained.

Here the lift is written on that face. A design in a node's subtree puts
treated weight on ``U_w = I | F`` and control weight on ``U_v = O | F``. Its
vector ``y = (1; w; v)`` is ``P t`` with ``t = (1; w without its last entry;
v without its last entry)``, the dropped entries being ``1`` minus the others.
``P`` has integer entries, so ``Y = P T P'`` with ``T`` positive semidefinite
of size ``|U_w| + |U_v| - 1`` loses nothing and the sum and row-sum constraints
hold identically.

What is left, each valid for every design in the subtree:

- ``Y_00 = 1``;
- ``Y >= 0`` elementwise;
- complementarity, ``Y[w_j, v_j] = 0`` for each free market ``j``;
- cardinality, by Cauchy-Schwarz on a vector supported on ``k`` entries:
  ``m sum Y[w_i, w_i] >= Y_00`` (the treated set has ``m`` markets);
  ``r sum_F Y[w_i, w_i] >= sum_F Y[w_i, w_k]`` (at most ``r`` free markets are
  treated); ``(n - m) sum Y[v_i, v_i] >= Y_00``; and
  ``(|F| - r) sum_F Y[v_i, v_i] >= sum_F Y[v_i, v_k]``.

The objective is ``<G, Y_ww> + <G, Y_vv>``, ``G`` the Gram matrix of the
centred columns, divided by a power of two near its largest diagonal so the
division is exact.

The bound is not the solver's value. The dual is solved for multipliers: ``y0``
on ``Y_00 = 1``, ``mu`` on complementarity, ``H >= 0`` on ``Y >= 0`` and
``nu >= 0`` on the cuts. For any such multipliers and any design,

    <C, Y> = <S, T> + y0 + <H, Y> + sum nu_k <K_k, Y> >= y0 + <S, T>,
    S = P'(C - y0 E_00 - sum mu_j E_j - H - sum nu_k K_k) P,

and ``<S, T> >= lambda_min(S) tr T``, with ``tr T = |t|^2 <= 3`` for a design.
So ``y0 + 3 min(0, lambda_min(S))`` is a lower bound on the subtree's minimum
for any multipliers with the right signs. The multipliers are taken from the
solver, clipped to their signs, and ``S`` and a lower bound on its smallest
eigenvalue are computed in exact arithmetic (``certify``). No solver tolerance
and no rounding enters the bound.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certify import Dy, down, lambda_min_lower


def layout(I, O, F):
    """``(U_w, U_v, P)``: the supports and the integer map ``t -> y``."""
    Uw, Uv = sorted(I + F), sorted(O + F)
    kw, kv = len(Uw), len(Uv)
    N, K = 1 + kw + kv, kw + kv - 1
    P = np.zeros((N, K), dtype=np.int64)
    P[0, 0] = 1
    for off, k, col in ((1, kw, 1), (1 + kw, kv, kw)):
        P[off + k - 1, 0] = 1
        for a in range(k - 1):
            P[off + a, col + a] = 1
            P[off + k - 1, col + a] = -1
    return Uw, Uv, P


def constraints(n, m, I, O, F):
    """The integer matrices of the equalities and the cuts, in ``Y`` space."""
    Uw, Uv, P = layout(I, O, F)
    kw, kv = len(Uw), len(Uv)
    N = 1 + kw + kv
    iw = {j: 1 + a for a, j in enumerate(Uw)}
    iv = {j: 1 + kw + a for a, j in enumerate(Uv)}
    r = m - len(I)
    E00 = np.zeros((N, N), dtype=np.int64)
    E00[0, 0] = 1
    comp = []
    for j in F:
        E = np.zeros((N, N), dtype=np.int64)
        E[iw[j], iv[j]] = E[iv[j], iw[j]] = 1
        comp.append(E)
    cuts = []

    def card(idx, k, total):
        Kc = np.zeros((N, N), dtype=np.int64)
        for i in idx:
            Kc[i, i] += k
        if total:
            Kc[0, 0] -= 1
        else:
            Kc[np.ix_(idx, idx)] -= 1
        return Kc

    if kw > m:
        cuts.append(card([iw[j] for j in Uw], m, True))
    if r >= 1 and len(F) > r:
        cuts.append(card([iw[j] for j in F], r, False))
    if kv > n - m:
        cuts.append(card([iv[j] for j in Uv], n - m, True))
    s = len(F) - r
    if s >= 1 and len(F) > s:
        cuts.append(card([iv[j] for j in F], s, False))
    return Uw, Uv, P, E00, comp, cuts


def objective(G, Uw, Uv, shift):
    """``C`` in ``Y`` space, exact, divided by ``2**shift``."""
    kw, kv = len(Uw), len(Uv)
    N = 1 + kw + kv
    A = np.zeros((N, N), dtype=object)
    A[:] = 0
    A[1:1 + kw, 1:1 + kw] = G.A[np.ix_(Uw, Uw)]
    A[1 + kw:, 1 + kw:] = G.A[np.ix_(Uv, Uv)]
    return Dy(A, G.e + shift)


def dnn_bound(panel, m, I, O, F, solver="CLARABEL"):
    """``(certified lower bound, solver's value, status)`` for one node."""
    import cvxpy as cp
    n = panel.Bc.shape[1]
    Uw, Uv, P, E00, comp, cuts = constraints(n, m, I, O, F)
    diag = max(float(panel.G.frac(j, j)) for j in range(n))
    shift = math.frexp(diag)[1]                       # 2**shift >= max diag
    C = objective(panel.G, Uw, Uv, shift)
    Cf = C.float()
    N = Cf.shape[0]
    y0 = cp.Variable()
    H = cp.Variable((N, N), symmetric=True)
    mu = cp.Variable(len(comp)) if comp else None
    nu = cp.Variable(len(cuts), nonneg=True) if cuts else None
    Z = Cf - y0 * E00 - H
    if comp:
        Z = Z - sum(mu[i] * comp[i] for i in range(len(comp)))
    if cuts:
        Z = Z - sum(nu[k] * cuts[k] for k in range(len(cuts)))
    S = P.T @ Z @ P
    prob = cp.Problem(cp.Maximize(y0), [(S + S.T) / 2 >> 0, H >= 0])
    prob.solve(solver=getattr(cp, solver))
    if y0.value is None:
        return 0.0, float("nan"), prob.status
    lb = certify(C, P, E00, comp, cuts, float(y0.value), H.value,
                 None if mu is None else mu.value,
                 None if nu is None else nu.value)
    return down(max(lb, Fraction(0)) * (1 << shift)), \
        float(prob.value) * 2.0 ** shift, prob.status


def certify(C, P, E00, comp, cuts, y0, H, mu, nu):
    """``y0 + 3 min(0, lambda_min(S))``, exactly, for multipliers clipped to sign."""
    H = np.clip(np.asarray(H, dtype=float), 0.0, None)
    H = np.triu(H) + np.triu(H, 1).T                  # exactly symmetric
    Z = C - Dy.ints(E00).times(y0) - Dy.of(H)
    for i, E in enumerate(comp):
        Z = Z - Dy.ints(E).times(float(mu[i]))
    for k, K in enumerate(cuts):
        Z = Z - Dy.ints(K).times(max(float(nu[k]), 0.0))
    Pd = Dy.ints(P)
    S = Pd.T @ Z @ Pd
    lam = lambda_min_lower(S)
    return Fraction(y0) + 3 * min(Fraction(0), lam)
