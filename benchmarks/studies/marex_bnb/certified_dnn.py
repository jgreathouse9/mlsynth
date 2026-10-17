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

``indicator_lift`` adds the treated indicators of Wiegele and Zhao (2024) and
Burer (2009): ``z_j = 1`` when free market ``j`` is treated, and ``u_j = 1 -
z_j``, with ``sum_F z = r``. The vector is ``y = (1; w; v; z; u)``, still the
image of ``t = (1; w; v; z)`` less one entry of each block under an integer
map, and the products that vanish on every design are fixed at zero:
``w_j u_j`` (a market off the treated set has no treated weight), ``v_j z_j``,
``z_j u_j`` (``z_j`` is binary) and ``w_j v_j``. Nonnegativity of ``Y`` now
covers the products with ``u``, which are the reformulation-linearization
inequalities of the indicators. A design's ``t`` has ``|t|^2 <= 3 + r``.

Every lift is a ``Lift``: the integer map ``P``, the objective, the entries
fixed at zero, the cuts and the bound ``tau`` on ``|t|^2``. ``lift_bound``
solves any of them and certifies the result the same way.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from benchmarks.studies.marex_bnb.certify import Dy, down, lambda_min_lower


class Lift:
    """A lifted relaxation of one node: ``Y = P T P'``, ``T`` PSD, ``T_00 = 1``.

    ``index`` names the rows of ``Y``; ``zero`` marks entries fixed at zero
    (their multipliers are free); every other entry is nonnegative; each cut
    ``K`` asserts ``<K, Y> >= 0``; ``tau`` bounds ``|t|^2`` over the designs.
    """

    def __init__(self, P, rows, zero, cuts, tau):
        self.P, self.rows, self.zero, self.cuts, self.tau = P, rows, zero, cuts, tau


def _block_map(P, row, col, k, total):
    """Rows ``row..row+k-1`` of ``y``: ``t`` entries ``col..``, the last ``total`` minus them."""
    for a in range(k - 1):
        P[row + a, col + a] = 1
        P[row + k - 1, col + a] = -1
    P[row + k - 1, 0] = total


def trace_lift(n, m, I, O, F):
    """``y = (1; w; v)`` with complementarity and the cardinality cuts."""
    Uw, Uv = sorted(I + F), sorted(O + F)
    kw, kv = len(Uw), len(Uv)
    rows = [("1", None)] + [("w", j) for j in Uw] + [("v", j) for j in Uv]
    P = np.zeros((len(rows), kw + kv - 1), dtype=np.int64)
    P[0, 0] = 1
    _block_map(P, 1, 1, kw, 1)
    _block_map(P, 1 + kw, kw, kv, 1)
    pos = {key: i for i, key in enumerate(rows)}
    zero = np.zeros((len(rows), len(rows)), dtype=bool)
    for j in F:
        _zero(zero, pos[("w", j)], pos[("v", j)])
    cuts = _cardinality(n, m, I, O, F, pos, len(rows))
    return Lift(P, rows, zero, cuts, 3)


def indicator_lift(n, m, I, O, F, cuts=True):
    """``y = (1; w; v; z; u)``: ``trace_lift`` plus the treated indicators."""
    Uw, Uv = sorted(I + F), sorted(O + F)
    kw, kv, kf = len(Uw), len(Uv), len(F)
    r = m - len(I)
    rows = ([("1", None)] + [("w", j) for j in Uw] + [("v", j) for j in Uv]
            + [("z", j) for j in F] + [("u", j) for j in F])
    K = kw + kv + kf - 2
    P = np.zeros((len(rows), K), dtype=np.int64)
    P[0, 0] = 1
    _block_map(P, 1, 1, kw, 1)
    _block_map(P, 1 + kw, kw, kv, 1)
    _block_map(P, 1 + kw + kv, kw + kv - 1, kf, r)
    for a in range(kf):                               # u_j = 1 - z_j
        P[1 + kw + kv + kf + a] = -P[1 + kw + kv + a]
        P[1 + kw + kv + kf + a, 0] += 1
    pos = {key: i for i, key in enumerate(rows)}
    zero = np.zeros((len(rows), len(rows)), dtype=bool)
    for j in F:
        _zero(zero, pos[("w", j)], pos[("v", j)])
        _zero(zero, pos[("w", j)], pos[("u", j)])
        _zero(zero, pos[("v", j)], pos[("z", j)])
        _zero(zero, pos[("z", j)], pos[("u", j)])
    cuts = _cardinality(n, m, I, O, F, pos, len(rows)) if cuts else []
    return Lift(P, rows, zero, cuts, 3 + r)


def treated_lift(n, m, I, O, F):
    """``y = (1; w; z; u)``: the indicator lift of the treated side alone.

    The control weights are left out, so the lift bounds only ``g`` over the
    subtree; a caller adds a bound on the control side, ``h(O | F)`` being one,
    since every control set of the subtree lies in ``O | F``. The control fit
    is a small share of ``f`` on these panels (under 1% at GeoLift's optimum),
    and dropping its block shrinks the PSD block by a third.
    """
    Uw = sorted(I + F)
    kw, kf = len(Uw), len(F)
    r = m - len(I)
    rows = ([("1", None)] + [("w", j) for j in Uw]
            + [("z", j) for j in F] + [("u", j) for j in F])
    P = np.zeros((len(rows), kw + kf - 1), dtype=np.int64)
    P[0, 0] = 1
    _block_map(P, 1, 1, kw, 1)
    _block_map(P, 1 + kw, kw, kf, r)
    for a in range(kf):                               # u_j = 1 - z_j
        P[1 + kw + kf + a] = -P[1 + kw + a]
        P[1 + kw + kf + a, 0] += 1
    pos = {key: i for i, key in enumerate(rows)}
    zero = np.zeros((len(rows), len(rows)), dtype=bool)
    for j in F:
        _zero(zero, pos[("w", j)], pos[("u", j)])
        _zero(zero, pos[("z", j)], pos[("u", j)])
    return Lift(P, rows, zero, [], 2 + r)


def _zero(mask, a, b):
    mask[a, b] = mask[b, a] = True


def _cardinality(n, m, I, O, F, pos, N):
    """The Cauchy-Schwarz cuts on the treated and control weights."""
    Uw, Uv = sorted(I + F), sorted(O + F)
    r = m - len(I)
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

    if len(Uw) > m:
        cuts.append(card([pos[("w", j)] for j in Uw], m, True))
    if r >= 1 and len(F) > r:
        cuts.append(card([pos[("w", j)] for j in F], r, False))
    if len(Uv) > n - m:
        cuts.append(card([pos[("v", j)] for j in Uv], n - m, True))
    s = len(F) - r
    if s >= 1 and len(F) > s:
        cuts.append(card([pos[("v", j)] for j in F], s, False))
    return cuts


def objective(G, lift, shift):
    """``C`` in ``Y`` space, exact, divided by ``2**shift``."""
    N = len(lift.rows)
    A = np.zeros((N, N), dtype=object)
    A[:] = 0
    for side in ("w", "v"):
        idx = [i for i, (b, _) in enumerate(lift.rows) if b == side]
        mk = [j for b, j in lift.rows if b == side]
        A[np.ix_(idx, idx)] = G.A[np.ix_(mk, mk)]
    return Dy(A, G.e + shift)


def lift_bound(panel, lift, solver="CLARABEL", marginals=False, **opts):
    """``(certified lower bound, solver's value, status)`` for one lift.

    With ``marginals=True`` a fourth entry maps each row of ``y`` to its value
    in the relaxation's solution, ``P T e_0``, for branching and rounding.
    """
    import cvxpy as cp
    n = panel.Bc.shape[1]
    diag = max(float(panel.G.frac(j, j)) for j in range(n))
    shift = math.frexp(diag)[1]                       # 2**shift >= max diag
    C = objective(panel.G, lift, shift)
    Cf = C.float()
    N = Cf.shape[0]
    E00 = np.zeros((N, N))
    E00[0, 0] = 1
    y0 = cp.Variable()
    H = cp.Variable((N, N), symmetric=True)
    Z = Cf - y0 * E00 - H
    nu = cp.Variable(len(lift.cuts), nonneg=True) if lift.cuts else None
    if lift.cuts:
        Z = Z - sum(nu[k] * lift.cuts[k] for k in range(len(lift.cuts)))
    S = lift.P.T @ Z @ lift.P
    sign = (~lift.zero).astype(float)
    psd = (S + S.T) / 2 >> 0
    prob = cp.Problem(cp.Maximize(y0), [psd, cp.multiply(sign, H) >= 0])
    prob.solve(solver=getattr(cp, solver), **opts)
    if y0.value is None:
        out = (0.0, float("nan"), prob.status)
        return out + ({},) if marginals else out
    lb = certify(C, lift, float(y0.value), H.value,
                 None if nu is None else nu.value)
    out = (down(max(lb, Fraction(0)) * (1 << shift)),
           float(prob.value) * 2.0 ** shift, prob.status)
    if not marginals:
        return out
    T = np.asarray(psd.dual_value, dtype=float)
    y = lift.P @ (T[:, 0] / T[0, 0]) if T[0, 0] > 0 else np.zeros(N)
    return out + (dict(zip(lift.rows, y.tolist())),)


def certify(C, lift, y0, H, nu):
    """``y0 + tau min(0, lambda_min(S))``, exactly, for multipliers clipped to sign."""
    H = np.asarray(H, dtype=float)
    H = np.where(lift.zero, H, np.clip(H, 0.0, None))
    H = np.triu(H) + np.triu(H, 1).T                  # exactly symmetric
    N = H.shape[0]
    E00 = np.zeros((N, N), dtype=np.int64)
    E00[0, 0] = 1
    Z = C - Dy.ints(E00).times(y0) - Dy.of(H)
    for k, K in enumerate(lift.cuts):
        Z = Z - Dy.ints(K).times(max(float(nu[k]), 0.0))
    Pd = Dy.ints(lift.P)
    lam = lambda_min_lower(Pd.T @ Z @ Pd)
    return Fraction(y0) + lift.tau * min(Fraction(0), lam)


def dnn_bound(panel, m, I, O, F, solver="CLARABEL", kind="trace",
              marginals=False):
    """The certified bound of one node under the ``trace`` or ``indicator`` lift."""
    n = panel.Bc.shape[1]
    build = trace_lift if kind == "trace" else indicator_lift
    return lift_bound(panel, build(n, m, I, O, F), solver, marginals)
