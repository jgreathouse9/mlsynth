"""The persp+N bound of a simplex fit under a cardinality, certified exactly (#900).

A fit picks ``m`` of ``n`` units and weights on them,

    f(S) = min { w'(G + ridge I) w + a'w : w >= 0, 1'w = 1, w_j = 0 off S },

with ``G`` the Gram matrix of the centred units and ``a`` a penalty. A node of a
branch and bound fixes some units into ``S`` (``I``), some out (``O``), and
leaves the rest free (``F``). This module bounds ``f`` below over every set of
the node, and certifies the bound in exact arithmetic. MAREX's search
(:mod:`~mlsynth.utils.marex_helpers.branch`) bounds a design's two fits with
two such sides sharing their indicators; LEXSCM's treated-set search bounds one.

The relaxation. The persp+N relaxation (``benchmarks/studies/marex_bnb``) of one
side, in its face form: a semidefinite ``X`` standing for ``w w'``, with
``x = X 1`` standing for ``w``, and

- ``X >= 0`` entrywise and ``1'X1 = 1``, the products of ``w >= 0`` and the
  simplex;
- for every free unit, ``x_j <= ind_j`` and the perspective block
  ``[[ind_j, x_j], [x_j, X_jj]] >= 0``, with ``ind_j = z_j`` (``1 - z_j`` on a
  control side), ``0 <= z <= 1`` and ``sum_F z = m - |I|``.

The face form is the relaxation MAREX first used, ``M = [[1, x'], [x, X]] >= 0``
with ``X 1 = x`` and ``1'x = 1``, written on the face those two constraints
confine ``M`` to. They make ``(1, -1)`` a null vector of ``M``, so ``M`` has no
strictly feasible point; projecting onto the face restores one (Wolkowicz and
Zhao, Discrete Appl. Math. 1999). The bound is the same, the solve 7% to 23%
faster at Walmart's root, and the solver no longer stops short with an
inaccurate status (``benchmarks/studies/lexscm_bnb``).

The options, each valid at every design and so kept in the relaxation:

- ``ridge``: ``G + ridge I`` in place of ``G``, LEXSCM's ``targeting_penalty``;
- ``floor``: every chosen weight at least ``c``. A unit is out (``x_j = 0``) or
  in with ``c <= x_j <= 1 - c (m - 1)``, whose hull is
  ``c z_j <= x_j <= (1 - c (m - 1)) z_j`` (our reading of Balas, Disjunctive
  Programming, 2018, Theorem 2.1); a fixed-in unit gets the bounds themselves;
- ``costs`` and ``budget``: ``sum_F cost_j z_j <= budget - cost(I)``;
- ``conflicts``: pairs never chosen together, so ``X_ij = 0`` (``w_i w_j = 0`` at
  every design) and ``z_i + z_j <= 1`` less the pair's fixed-in units.

The certificate. Take any multipliers: ``alpha`` on ``1'X1 = 1``, ``H`` on
``X >= 0``, ``mu`` on ``ind_j - x_j >= 0``, a 2x2 ``Lam_j`` per perspective
block, ``nu`` and ``rho`` on the floor's two sides, ``tau`` on ``X_ij = 0``, and
``pi`` on each row in ``z``. Clipped to their signs (each ``Lam_j`` made
semidefinite by :func:`_psd2`), each subtracted term is non-negative at a design,
so the Lagrangian is at most the design's objective; and it is linear in the
lifted variables, ``const + <S, X> + c'z`` per side, with

    S = (G + ridge I)/scale - alpha 1 1' - H - diag(Lam_22) - tau/2 + (q 1' + 1 q')/2,
    q = a/scale + mu - 2 Lam_12 - nu + rho.

A design has ``X = w w'`` on the simplex, so ``tr X = |w|^2 <= 1`` and
``<S, X> >= min(0, lambda_min(S))``, with ``lambda_min`` bounded below exactly by
:func:`lambda_min_lower`; its ``z`` is a 0-1 vector with ``m - |I|`` free units
on, so ``c'z`` is at least the sum of the ``m - |I|`` smallest ``c_j``. The sum
is a lower bound on every design of the node, assembled from the exact Gram:
the rigorous bound of Jansson, Chaykin and Keil (SIAM J. Numer. Anal. 2007) with
exact arithmetic in place of interval arithmetic. The trace bound is 1, where
the full form's ``M`` has ``tr M = 1 + |w|^2 <= 2``. Any multipliers give a
valid bound and accurate ones a tight one, so a solve stopped at ``max_iter``
still certifies one: weaker, never wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import MlsynthConfigError
from .marex_helpers.certify import Dyadic

#: The solver of the persp+N relaxation.
SOLVER = "CLARABEL"


# ----------------------------------------------------------------------
# exact linear algebra
# ----------------------------------------------------------------------


def _float(D: Dyadic) -> np.ndarray:
    """The dyadic matrix ``D`` rounded to doubles; past the largest double,
    an infinity."""
    scale = 1 << D.e

    def one(x: int) -> float:
        try:
            return float(Fraction(int(x), scale))
        except OverflowError:
            return math.inf if int(x) > 0 else -math.inf
    flat = [one(x) for x in D.N.ravel()]
    return np.array(flat, dtype=float).reshape(D.N.shape)


def _from_fractions(rows: Sequence[Sequence[Fraction]]) -> Dyadic:
    """The matrix of rationals ``rows``, every one a dyadic rational, exactly."""
    flat = [q for row in rows for q in row]
    e = max((q.denominator.bit_length() - 1 for q in flat), default=0)
    N = np.array([int(q.numerator) << (e - (q.denominator.bit_length() - 1))
                  for q in flat], dtype=object)
    return Dyadic(N.reshape(len(rows), len(rows[0])), e)


def lambda_min_lower(D: Dyadic, tries: int = 40) -> Optional[Fraction]:
    """An exact lower bound on the smallest eigenvalue of the symmetric ``D``.

    A floating-point Cholesky factor ``L`` of ``D - sigma I``, with ``sigma``
    just under the floating-point smallest eigenvalue, is exact as a dyadic
    matrix, so the residual ``R = D - sigma I - L L'`` is computed exactly. Then
    ``lambda_min(D) >= sigma + lambda_min(R) >= sigma - max_i sum_j |R_ij|`` by
    Weyl's and Gershgorin's theorems, whatever rounding produced ``L``. Returns
    ``None`` when no factor is found.
    """
    S = _float(D)
    k = S.shape[0]
    if not np.all(np.isfinite(S)):
        return None
    lam = float(np.linalg.eigvalsh(S)[0])
    size = float(np.abs(S).sum(axis=1).max()) or 1.0
    step = max(abs(lam) * 1e-12, 64 * k * np.finfo(float).eps * size)
    for _ in range(tries):
        sigma = lam - step
        try:
            L = np.linalg.cholesky(S - sigma * np.eye(k))
        except np.linalg.LinAlgError:
            step *= 4.0
            continue
        Ld = Dyadic.of(L)
        R = D - Dyadic.of(sigma * np.eye(k)) - Ld @ Ld.T
        row = max(sum(abs(int(v)) for v in R.N[i]) for i in range(k))
        return Fraction(sigma) - Fraction(row, 1 << R.e)
    return None


def _psd2(L: np.ndarray) -> Tuple[float, float, float]:
    """A 2x2 symmetric matrix made positive semidefinite: the diagonal clipped
    at zero and the off-diagonal shrunk. Any semidefinite multiplier keeps the
    certificate valid.

    ``b^2 <= a c`` is checked in exact arithmetic: ``b`` is clipped to
    ``sqrt(a) sqrt(c)`` and moved toward zero the few units in the last place
    that rounding can put it over. The two square roots are taken apart because
    the product ``a c`` can be subnormal, where it carries too few digits for
    its square root to land within a few units of the bound. Should the few
    units not suffice, ``b = 0`` is semidefinite.
    """
    a = max(float(L[0, 0]), 0.0)
    b = float(L[0, 1] + L[1, 0]) / 2.0
    c = max(float(L[1, 1]), 0.0)
    if Fraction(b) ** 2 <= Fraction(a) * Fraction(c):
        return a, b, c
    b = math.copysign(math.sqrt(a) * math.sqrt(c), b)
    for _ in range(8):
        if Fraction(b) ** 2 <= Fraction(a) * Fraction(c):
            return a, b, c
        b = math.nextafter(b, 0.0)
    return a, 0.0, c   # pragma: no cover - sqrt(a) sqrt(c) is within 3 units


# ----------------------------------------------------------------------
# the node, its sides and its rows
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NodeBound:
    """A node's certified persp+N bound.

    ``lower`` is an exact lower bound on the objective of every design in the
    node's subtree, ``relaxation`` the solver's value of the relaxation and
    ``z`` its indicators on the free units; all three are ``None`` when the
    solve or its certificate failed, or the node holds no design. ``status`` is
    the solver's (``optimal``, or ``user_limit`` for a solve stopped at its
    iteration cap, whose bound is certified all the same).
    """

    lower: Optional[Fraction]
    relaxation: Optional[float]
    z: Optional[Dict[int, float]]
    status: str = ""


#: The bound of a node whose solve or certificate failed.
UNBOUNDED = NodeBound(lower=None, relaxation=None, z=None, status="failed")


@dataclass(frozen=True, eq=False)
class Side:
    """One fit's block, and multipliers for its constraints.

    The structure: ``U`` the units the side can use; ``free`` the pairs
    ``(i, f)`` of a position in ``U`` and the unit's position in ``F``;
    ``control`` whether the side's indicator is ``1 - z``; ``lin`` the penalty
    over every unit, or ``None``; ``floored`` the pairs ``(i, f)`` whose weight
    the floor holds, ``f`` being ``None`` for a fixed-in unit; ``apart`` the
    pairs of positions in ``U`` never chosen together. The multipliers default
    to zero, which is a valid (and weak) choice; :func:`solve` fills them.
    """

    U: Tuple[int, ...]
    free: Tuple[Tuple[int, int], ...]
    control: bool = False
    lin: Optional[np.ndarray] = None
    floored: Tuple[Tuple[int, Optional[int]], ...] = ()
    apart: Tuple[Tuple[int, int], ...] = ()
    alpha: float = 0.0
    H: Optional[np.ndarray] = None
    mu: Tuple[float, ...] = ()
    Lam: Tuple[np.ndarray, ...] = ()
    nu: Tuple[float, ...] = ()
    rho: Tuple[float, ...] = ()
    tau: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Row:
    """``rhs - sum_f coef_f z_f >= 0`` over the free units' indicators."""

    coef: Tuple[Tuple[int, float], ...]
    rhs: float


@dataclass(frozen=True, eq=False)
class Plan:
    """A node's relaxation, before it is solved: the exact Gram and its scale,
    the sides and the rows in ``z``."""

    gram: Dyadic
    Gs: np.ndarray
    scale: Fraction
    r: int
    F: Tuple[int, ...]
    sides: Tuple[Side, ...]
    rows: Tuple[Row, ...] = ()
    ridge: float = 0.0
    floor: float = 0.0
    upper: Fraction = Fraction(1)


def plan(data, m: int, I: Sequence[int], O: Sequence[int], F: Sequence[int], *,
         linear: Optional[np.ndarray] = None, ridge: float = 0.0, floor: float = 0.0,
         costs: Optional[np.ndarray] = None, budget: Optional[float] = None,
         conflicts: Sequence[Tuple[int, int]] = (), control: bool = False,
         linear_control: Optional[np.ndarray] = None) -> Optional[Plan]:
    """The node ``(I, O, F)`` with ``m`` chosen, as a :class:`Plan`.

    ``data`` is the cluster's :class:`~mlsynth.utils.marex_helpers.certify.ExactData`.
    One side, the chosen units', unless ``control``, which adds MAREX's control
    side over ``O + F`` with indicator ``1 - z`` and penalty ``linear_control``.
    The floor, the budget and the conflicts act on the chosen units' side.
    ``None`` when the Gram's scale is zero or past the largest double, when a
    conflicting pair is fixed in together, or when nothing is free and the units
    fixed in are over the budget.
    """
    if ridge < 0 or not np.isfinite(ridge):
        raise MlsynthConfigError(f"the ridge has to be a non-negative number; got {ridge}.")
    if floor < 0 or floor * m >= 1.0:
        raise MlsynthConfigError(
            f"a floor of {floor} on {m} units needs {floor * m:.3g} of the weight, "
            f"and the simplex has 1.")
    if budget is not None and costs is None:
        raise MlsynthConfigError("a budget needs unit costs to hold it against.")
    I, O, F = list(I), list(O), list(F)
    G = data.gram
    gmax = max(G.frac(i, i) for i in range(data.n)) + Fraction(ridge)
    if not 0 < gmax < Fraction(np.finfo(float).max):
        return None
    scale = Fraction(2) ** round(math.log2(gmax))
    pos = {j: f for f, j in enumerate(F)}
    U = tuple(sorted(I + F))
    at = {j: i for i, j in enumerate(U)}
    free = tuple((at[j], pos[j]) for j in U if j in pos)
    floored = tuple((at[j], pos.get(j)) for j in U) if floor > 0 else ()
    rows: List[Row] = []
    apart = []
    for i, j in conflicts:
        i, j = int(i), int(j)
        if i in at and j in at:
            apart.append((min(at[i], at[j]), max(at[i], at[j])))
        held = sum(u in I for u in (i, j))
        if held == 2:
            return None
        coef = tuple((pos[u], 1.0) for u in (i, j) if u in pos)
        if coef:
            rows.append(Row(coef=coef, rhs=float(1 - held)))
    if budget is not None:
        costs = np.asarray(costs, dtype=float)
        left = float(budget) - float(sum(Fraction(float(costs[j])) for j in I))
        if F:
            rows.append(Row(coef=tuple((pos[j], float(costs[j])) for j in F), rhs=left))
        elif left < 0:
            return None                       # the one design left is over budget
    sides = [Side(U=U, free=free, control=False,
                  lin=None if linear is None else np.asarray(linear, dtype=float),
                  floored=floored, apart=tuple(sorted(set(apart))))]
    if control:
        Uc = tuple(sorted(O + F))
        atc = {j: i for i, j in enumerate(Uc)}
        sides.append(Side(U=Uc, free=tuple((atc[j], pos[j]) for j in Uc if j in pos),
                          control=True,
                          lin=None if linear_control is None
                          else np.asarray(linear_control, dtype=float)))
    return Plan(gram=G, Gs=data.Gf / float(scale), scale=scale, r=m - len(I), F=tuple(F),
                sides=tuple(sides), rows=tuple(rows), ridge=float(ridge), floor=float(floor),
                upper=Fraction(1) - Fraction(float(floor)) * (m - 1))


# ----------------------------------------------------------------------
# the relaxation and its solve
# ----------------------------------------------------------------------


class _Block:
    """One side's semidefinite block in cvxpy, and the constraints whose
    multipliers certify it."""

    def __init__(self, cp, p: Plan, side: Side, z):
        k = len(side.U)
        self.side, self.k = side, k
        self.X = cp.Variable((k, k), PSD=True)
        x = self.X @ np.ones(k)
        self.x = x
        self.csum = cp.sum(x) == 1
        self.cnn = self.X >= 0
        self.clin, self.cper = [], []
        for i, f in side.free:
            ind = (1 - z[f]) if side.control else z[f]
            self.clin.append(ind - x[i] >= 0)
            self.cper.append(cp.bmat([
                [cp.reshape(ind, (1, 1), order="C"), cp.reshape(x[i], (1, 1), order="C")],
                [cp.reshape(x[i], (1, 1), order="C"),
                 cp.reshape(self.X[i, i], (1, 1), order="C")]]) >> 0)
        self.clo, self.chi = [], []
        upper = float(p.upper)
        for i, f in side.floored:
            ind = 1.0 if f is None else z[f]
            self.clo.append(x[i] - p.floor * ind >= 0)
            self.chi.append(upper * ind - x[i] >= 0)
        self.capart = [self.X[a, b] == 0 for a, b in side.apart]
        self.constraints = ([self.csum, self.cnn] + self.clin + self.cper + self.clo
                            + self.chi + self.capart)

    def objective(self, cp, p: Plan):
        U = list(self.side.U)
        Q = p.Gs[np.ix_(U, U)] + (p.ridge / float(p.scale)) * np.eye(self.k)
        obj = cp.trace(Q @ self.X)
        if self.side.lin is not None:
            obj = obj + (self.side.lin[U] / float(p.scale)) @ self.x
        return obj

    def duals(self) -> Optional[Side]:
        cons = self.constraints
        if not all(c.dual_value is not None
                   and np.all(np.isfinite(np.asarray(c.dual_value, dtype=float))) for c in cons):
            return None
        return replace(
            self.side, alpha=-float(self.csum.dual_value),
            H=np.asarray(self.cnn.dual_value, dtype=float).reshape(self.k, self.k),
            mu=tuple(float(c.dual_value) for c in self.clin),
            Lam=tuple(np.asarray(c.dual_value, dtype=float) for c in self.cper),
            nu=tuple(float(c.dual_value) for c in self.clo),
            rho=tuple(float(c.dual_value) for c in self.chi),
            tau=tuple(-float(c.dual_value) for c in self.capart))


def solve(p: Plan, max_iter: Optional[int] = None
          ) -> Tuple[NodeBound, Optional[List[Side]], Optional[List[float]]]:
    """Solve the relaxation of ``p``; its value, its indicators, and the
    multipliers of its sides and its rows (``None`` when the solve gave none)."""
    import cvxpy as cp

    nF = len(p.F)
    z = cp.Variable(nF) if nF else None
    cons = [] if z is None else [z >= 0, z <= 1, cp.sum(z) == p.r]
    zrows = [r.rhs - sum(c * z[f] for f, c in r.coef) >= 0 for r in p.rows]
    blocks = [_Block(cp, p, s, z) for s in p.sides]
    obj = 0
    for b in blocks:
        obj = obj + b.objective(cp, p)
        cons += b.constraints
    problem = cp.Problem(cp.Minimize(obj), cons + zrows)
    opts = {} if max_iter is None else {"max_iter": int(max_iter)}
    try:
        problem.solve(solver=SOLVER, **opts)
    except (cp.error.SolverError, ArithmeticError, ValueError, np.linalg.LinAlgError):
        return UNBOUNDED, None, None
    if problem.value is None or not np.isfinite(problem.value):
        return UNBOUNDED, None, None
    sides = [b.duals() for b in blocks]
    rows = [c.dual_value for c in zrows]
    if any(s is None for s in sides) or not all(
            v is not None and np.isfinite(float(np.asarray(v).ravel()[0])) for v in rows):
        return UNBOUNDED, None, None
    zv = np.asarray(z.value, dtype=float) if z is not None else np.zeros(0)
    nb = NodeBound(lower=None, relaxation=float(problem.value) * float(p.scale),
                   z={j: float(zv[f]) for f, j in enumerate(p.F)}, status=str(problem.status))
    return nb, sides, [float(np.asarray(v).ravel()[0]) for v in rows]


# ----------------------------------------------------------------------
# the certificate
# ----------------------------------------------------------------------


def certificate(p: Plan, sides: Sequence[Side], rows: Sequence[float]) -> Optional[Fraction]:
    """A lower bound on every design of the node ``p``, from any multipliers.

    Each multiplier is first made to have its constraint's sign, so every
    subtracted term is zero or positive at a design; the Lagrangian is then
    ``const + <S, X> + c'z`` per side in units of the scale, bounded below over
    the designs exactly (the module docstring). ``None`` when a smallest
    eigenvalue could not be bounded.
    """
    G, scale = p.gram, p.scale
    floor, upper = Fraction(p.floor), p.upper
    ridge = Fraction(p.ridge) / scale
    const = Fraction(0)
    cz = [Fraction(0)] * len(p.F)
    for s in sides:
        U, k = s.U, len(s.U)
        H = np.zeros((k, k)) if s.H is None else np.clip((s.H + s.H.T) / 2.0, 0.0, None)
        H = np.triu(H) + np.triu(H, 1).T
        alpha = Fraction(s.alpha)
        const += alpha
        q = [Fraction(0) if s.lin is None else Fraction(float(s.lin[U[a]])) / scale
             for a in range(k)]
        S = [[G.frac(U[a], U[b]) / scale - alpha - Fraction(H[a, b]) for b in range(k)]
             for a in range(k)]
        for a in range(k):
            S[a][a] += ridge
        for (i, f), mu_raw, L in zip(s.free, s.mu, s.Lam):
            mu = Fraction(max(float(mu_raw), 0.0))
            a_, b_, c_ = (Fraction(v) for v in _psd2(np.asarray(L, float)))
            q[i] += mu - 2 * b_
            S[i][i] -= c_
            if s.control:                       # the indicator is 1 - z
                const -= mu + a_
                cz[f] += mu + a_
            else:                               # the indicator is z
                cz[f] -= mu + a_
        for (i, f), nu_raw, rho_raw in zip(s.floored, s.nu, s.rho):
            nu = Fraction(max(float(nu_raw), 0.0))
            rho = Fraction(max(float(rho_raw), 0.0))
            q[i] += rho - nu
            held = floor * nu - upper * rho     # x_j - c ind >= 0 and u ind - x_j >= 0
            if f is None:
                const += held
            else:
                cz[f] += held
        for (a, b), t in zip(s.apart, s.tau):
            t = Fraction(t) / 2
            S[a][b] -= t
            S[b][a] -= t
        for a in range(k):
            for b in range(k):
                S[a][b] += (q[a] + q[b]) / 2
        lam = lambda_min_lower(_from_fractions(S))
        if lam is None:
            return None
        const += min(Fraction(0), lam)
    for row, pi_raw in zip(p.rows, rows):
        pi = Fraction(max(float(pi_raw), 0.0))
        const -= pi * Fraction(row.rhs)
        for f, c in row.coef:
            cz[f] += pi * Fraction(c)
    if cz:
        const += sum(sorted(cz)[:p.r])
    return const * scale


def bound(data, m: int, I: Sequence[int], O: Sequence[int], F: Sequence[int], *,
          max_iter: Optional[int] = None, **options) -> NodeBound:
    """The certified persp+N bound of the node ``(I, O, F)`` with ``m`` chosen.

    ``options`` are :func:`plan`'s; ``max_iter`` caps the solver's iterations,
    and the multipliers it stops at are certified as any others are.
    """
    p = plan(data, m, I, O, F, **options)
    if p is None:
        return UNBOUNDED
    nb, sides, rows = solve(p, max_iter=max_iter)
    if sides is None:
        return nb
    lower = certificate(p, sides, rows)
    if lower is None:
        return UNBOUNDED
    return replace(nb, lower=lower)
