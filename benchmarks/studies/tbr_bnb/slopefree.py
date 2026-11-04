"""The shared-slope control group without the slope: a fractional 0-1 program (#965).

For a fixed treatment group, min over C and b > 0 of ||y - b X u_C||^2 is, at the
best slope b = c'u / u'Gu (c = X'y, G = X'X, u the indicator of C),

    RSS(C) = y'y - (c'u)^2 / u'Gu   when c'u > 0,   y'y otherwise,

so the best control group maximises the ratio (c'u)^2 / u'Gu over binary u with
c'u > 0 (derived here from the normal equation of b). An incumbent with ratio lam*
is optimal exactly when, for every size m,

    max { u'(cc' - lam* G) u : u binary, sum u = m, c'u >= 0 } <= 0,

the parametric form of a fractional program (Dinkelbach 1967; Borrero, Gillen and
Prokopyev 2017, Sect. 4.2, for 0-1 problems). Each size's maximum is bounded by a
semidefinite relaxation in u alone, Z = [[1, u'], [u, U]] >= 0 with U standing for
uu', and the constraints every binary design meets (Rows):

  * diag U = u (u_j^2 = u_j), and U_ij >= 0, U_ij <= u_i, U_ij >= u_i + u_j - 1, the
    products of the bounds 0 <= u <= 1 (Sherali and Adams 1990);
  * sum u = m and its products sum_j U_ij = m u_i;
  * c'u >= 0 and its products with u_i and 1 - u_i;
  * a no-good cut sum_C u - sum_notC u <= |C| - 1 excluding the incumbent at its size
    (Balas and Jeroslow 1972), so that size's bound can fall below zero;
  * triangle inequalities U_ij + U_ik - U_jk <= u_i and u_i + u_j + u_k - U_ij -
    U_ik - U_jk <= 1, valid at every binary point (Padberg 1989), added where the
    relaxation's solution violates them;
  * branching fixings u_j = v.

The certificate does not trust the solver. Any multipliers, the inequality ones
clipped at zero, give the Lagrangian L(Z) = <S, Z> + k with obj(Z) <= L(Z) on the
relaxation's feasible set, and that set has tr Z = 1 + sum(diag U) = 1 + m exactly,
so the maximum is at most k + (1 + m) max(0, lambda_max(S)) (weak duality with a
trace bound, as for the joint lift: Jansson, Chaykin and Keil 2007, Thm 3.2). S and
k are summed in rationals from the double data and the double multipliers, and
lambda_max(S) is bounded above exactly by mlsynth's Cholesky-residual bound on -S.

Balas, E. and Jeroslow, R. (1972). Canonical cuts on the unit hypercube. SIAM J.
    Appl. Math. 23, 61-69.
Borrero, J. S., Gillen, C. and Prokopyev, O. A. (2017). Fractional 0-1 programming:
    applications and algorithms. J. Global Optim. 69, 255-282.
Dinkelbach, W. (1967). On nonlinear fractional programming. Manag. Sci. 13, 492-498.
Jansson, C., Chaykin, D. and Keil, C. (2007). Rigorous error bounds for the optimal
    value in semidefinite programming. SIAM J. Numer. Anal. 46, 180-200.
Padberg, M. (1989). The boolean quadric polytope: some characteristics, facets and
    relatives. Math. Program. 45, 139-172.
Sherali, H. D. and Adams, W. P. (1990). A hierarchy of relaxations between the
    continuous and convex hull representations for zero-one programming problems.
    SIAM J. Discrete Math. 3, 411-430.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction as Q
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from mlsynth.utils.marex_helpers.branch import _from_fractions, lambda_min_lower

from .certificate import ExactData, round_down


def dyadic_below(q) -> Q:
    """The largest double at most the rational q, as a rational. The certificate's
    eigenvalue bound reads its matrix as dyadic rationals (mlsynth's
    _from_fractions), so lam enters as one; certifying at lam' <= lam* proves the
    stronger statement that no design's ratio exceeds lam'."""
    return Q(round_down(Q(q)))


def objective(c, G, lam):
    """cc' - lam G, exact for rational c, G and lam."""
    c = np.asarray(c, dtype=object)
    return np.outer(c, c) - lam * np.asarray(G, dtype=object)


def design_value(A, u):
    """u'Au for a 0-1 vector u, exact."""
    idx = [j for j, v in enumerate(u) if v]
    return sum((A[i, j] for i in idx for j in idx), Q(0))


class Rows:
    """The relaxation's constraints as rows <E_k, Z> = b_k or <= b_k over the upper
    triangle of Z (index 0 the constant, j + 1 the geo j). A row is a dict
    {(a, b): coefficient} with a <= b, read as sum coefficient * Z_ab. The same table
    builds the solver's model and the certificate, so the certificate holds for the
    constraints written here whatever the solver did."""

    def __init__(self, c: Sequence, m: int, cut: Optional[Set[int]] = None,
                 triangles: Sequence[Tuple[int, Tuple[int, int, int]]] = (),
                 fixes: Sequence[Tuple[int, int]] = ()):
        p = len(c)
        if cut is not None and len(cut) != m:
            raise ValueError(f"the no-good cut's design has size {len(cut)}, not the relaxation's size {m}")
        self.p, self.n, self.m = p, p + 1, m
        self.eq: List[Dict] = []; self.beq: List = []
        self.ineq: List[Dict] = []; self.bin: List = []
        self.c_rows: List[int] = []             # inequality rows carrying c, scaled for the solver
        U = lambda i, j: (min(i, j) + 1, max(i, j) + 1)  # noqa: E731
        u = lambda i: (0, i + 1)                         # noqa: E731

        def row(*pairs):
            r = {}
            for key, v in pairs:
                r[key] = r.get(key, 0) + v
            return {k: v for k, v in r.items() if v != 0}

        def eq(r, b):
            self.eq.append(r); self.beq.append(b)

        def le(r, b, carries_c=False):
            if carries_c:
                self.c_rows.append(len(self.ineq))
            self.ineq.append(r); self.bin.append(b)

        eq(row(((0, 0), 1)), 1)
        for i in range(p):
            eq(row((U(i, i), 1), (u(i), -1)), 0)
        eq(row(*((u(i), 1) for i in range(p))), m)
        for i in range(p):
            eq(row(*((U(i, j), 1) for j in range(p)), (u(i), -m)), 0)
        for j, v in fixes:
            eq(row((u(j), 1)), v)
        for i, j in itertools.combinations(range(p), 2):
            le(row((U(i, j), -1)), 0)
            le(row((U(i, j), 1), (u(i), -1)), 0)
            le(row((U(i, j), 1), (u(j), -1)), 0)
            le(row((u(i), 1), (u(j), 1), (U(i, j), -1)), 1)
        le(row(*((u(j), -c[j]) for j in range(p))), 0, True)                      # c'u >= 0
        for i in range(p):
            le(row(*((U(i, j), -c[j]) for j in range(p))), 0, True)              # u_i c'u >= 0
            le(row(*((U(i, j), c[j]) for j in range(p)), *((u(j), -c[j]) for j in range(p))), 0, True)
        if cut is not None:
            le(row(*((u(j), 1 if j in cut else -1) for j in range(p))), m - 1)
        for kind, (i, j, k) in triangles:
            if kind == 0:
                le(row((U(i, j), 1), (U(i, k), 1), (U(j, k), -1), (u(i), -1)), 0)
            else:
                le(row((u(i), 1), (u(j), 1), (u(k), 1), (U(i, j), -1), (U(i, k), -1), (U(j, k), -1)), 1)
        self.n_eq, self.n_in = len(self.eq), len(self.ineq)

    def holds(self, Z) -> bool:
        """Every row at the exact matrix Z."""
        val = lambda r: sum((v * Z[a, b] for (a, b), v in r.items()), Q(0))  # noqa: E731
        return all(val(r) == b for r, b in zip(self.eq, self.beq)) and \
            all(val(r) <= b for r, b in zip(self.ineq, self.bin))

    def matrices(self, c_scale: float):
        """The rows as sparse matrices over vec(Z) (column-major), in doubles, the
        rows carrying c divided by ``c_scale``."""
        n = self.n

        def build(rows, scale_rows=()):
            data, ri, ci = [], [], []
            scale = set(scale_rows)
            for k, r in enumerate(rows):
                f = 1.0 / c_scale if k in scale else 1.0
                for (a, b), v in r.items():
                    data.append(float(v) * f); ri.append(k); ci.append(a + n * b)
            return sp.csr_matrix((data, (ri, ci)), shape=(len(rows), n * n))
        return build(self.eq), build(self.ineq, self.c_rows)


@dataclass
class Solve:
    value: float                 # the relaxation's value in the solver's units
    scale: float                 # the objective's scale: value * scale is in the data's units
    y_eq: np.ndarray             # multipliers in the data's units, for bound()
    y_in: np.ndarray
    Z: np.ndarray
    status: str


def _pow2(x):
    return 2.0 ** round(math.log2(x)) if x > 0 else 1.0


def solve(c, G, lam, R: Rows) -> Solve:
    """The relaxation by Clarabel in scaled units: the objective divided by a power
    of two near its largest entry and the rows carrying c by one near max |c|, both
    exact, so the multipliers return to the data's units by those two factors."""
    A = np.asarray(objective(c, G, lam), dtype=float)
    s = _pow2(float(np.abs(A).max())); cs = _pow2(float(np.max(np.abs(np.asarray(c, dtype=float)))))
    Me, Mi = R.matrices(cs)
    n = R.n
    A0 = np.zeros((n, n)); A0[1:, 1:] = A / s
    Z = cp.Variable((n, n), symmetric=True)
    z = cp.vec(Z, order="F")
    ce = Me @ z == np.asarray(R.beq, dtype=float)
    ci = Mi @ z <= np.asarray(R.bin, dtype=float)
    prob = cp.Problem(cp.Maximize(cp.trace(A0 @ Z)), [Z >> 0, ce, ci])
    prob.solve(solver="CLARABEL")
    if ce.dual_value is None:            # pragma: no cover - Clarabel returns duals whenever it returns
        return Solve(-np.inf, s, np.zeros(R.n_eq), np.zeros(R.n_in), np.zeros((n, n)), prob.status)
    y_in = np.array(ci.dual_value, dtype=float) * s
    y_in[R.c_rows] /= cs
    return Solve(float(prob.value), s, np.array(ce.dual_value, dtype=float) * s, y_in, Z.value, prob.status)


def bound(A, R: Rows, y_eq, y_in, m) -> Optional[Q]:
    """k + (1 + m) max(0, lambda_max(S)) with S = [[0, 0], [0, A]] - sum_k y_k E_k and
    k = sum_k y_k b_k, in rationals; the inequality multipliers clipped at zero. At
    least the relaxation's maximum, and so every design's u'Au (module docstring).
    None when the eigenvalue bound finds no factor."""
    n = R.n
    S = np.full((n, n), Q(0), dtype=object)
    S[1:, 1:] = np.asarray(A, dtype=object)
    k = Q(0)
    for rows, bs, ys, clip in ((R.eq, R.beq, y_eq, False), (R.ineq, R.bin, y_in, True)):
        for r, b, yv in zip(rows, bs, ys):
            yq = Q(float(yv))
            if clip and yq < 0:
                yq = Q(0)
            if yq == 0:
                continue
            k += yq * b
            for (a, bb), v in r.items():
                if a == bb:
                    S[a, a] -= yq * v
                else:
                    h = yq * Q(v) / 2
                    S[a, bb] -= h; S[bb, a] -= h
    if any(q.denominator & (q.denominator - 1) for q in S.flat):
        raise ValueError("the certificate's matrix must be dyadic: pass lam through dyadic_below "
                         "(mlsynth's _from_fractions reads every entry as a dyadic rational)")
    low = lambda_min_lower(_from_fractions((-S).tolist()))   # lambda_max(S) <= -low
    if low is None:  # pragma: no cover - 40 quadrupled shifts factor any finite symmetric matrix, and S is finite
        return None
    return k + (1 + m) * max(Q(0), -low)


def separate_triangles(u, U, k=400, tol=1e-6):
    """The ``k`` most violated triangle inequalities at (u, U), as
    (violation, kind, (i, j, k)): kind 0 is U_ij + U_ik - U_jk <= u_i, kind 1 is
    u_i + u_j + u_k - U_ij - U_ik - U_jk <= 1."""
    u = np.asarray(u, float); U = np.asarray(U, float); p = u.size
    if p < 3:
        return []
    t = np.array(list(itertools.combinations(range(p), 3)))
    a, b, d = t.T
    out = []
    for i, j, kk in ((a, b, d), (b, a, d), (d, a, b)):
        v = U[i, j] + U[i, kk] - U[j, kk] - u[i]
        out += [(float(v[q]), 0, (int(i[q]), int(j[q]), int(kk[q]))) for q in np.flatnonzero(v > tol)]
    v = u[a] + u[b] + u[d] - U[a, b] - U[a, d] - U[b, d] - 1
    out += [(float(v[q]), 1, (int(a[q]), int(b[q]), int(d[q]))) for q in np.flatnonzero(v > tol)]
    out.sort(key=lambda r: -r[0])
    return out[:k]


@dataclass
class SizeResult:
    certified: bool
    bound: Optional[Q]
    rounds: int
    triangles: int
    seconds: float
    u: Optional[np.ndarray] = None       # the relaxation's last u, for branching


def size_certificate(c, G, lam, m, cut=None, fixes=(), rounds=6, per_round=400) -> SizeResult:
    """Whether no design of size m (within the fixings) has u'(cc' - lam G)u > 0:
    the relaxation's certificate, with up to ``rounds`` rounds of the ``per_round``
    most violated triangle inequalities where it is above zero."""
    t0 = time.perf_counter()
    lam = dyadic_below(lam)
    A = objective(c, G, lam)
    tri: List = []
    for r in range(rounds + 1):
        R = Rows(c, m, cut=cut, triangles=tri, fixes=fixes)
        sol = solve(c, G, lam, R)
        b = bound(A, R, sol.y_eq, sol.y_in, m)
        u = None if sol.Z is None else np.asarray(sol.Z)[0, 1:]
        if b is not None and b <= 0:
            return SizeResult(True, b, r, len(tri), time.perf_counter() - t0, u)
        new = [] if sol.Z is None else separate_triangles(sol.Z[0, 1:], sol.Z[1:, 1:], k=per_round)
        new = [(kind, idx) for _, kind, idx in new if (kind, idx) not in set(tri)]
        if not new or r == rounds:
            return SizeResult(False, b, r, len(tri), time.perf_counter() - t0, u)
        tri += new
    return SizeResult(False, None, rounds, len(tri), time.perf_counter() - t0)  # pragma: no cover - the loop returns


def _ratio(c, G, C):
    u = np.zeros(len(c), dtype=object); u[:] = Q(0); u[sorted(C)] = Q(1)
    cu = c @ u
    return cu, (cu * cu / (u @ G @ u)) if cu > 0 else Q(0)


@dataclass
class SlopeFreeCertificate:
    """The outcome of :func:`certify_slope_free` for one treatment group."""

    incumbent: Q                          # the incumbent's exact RSS at its best slope
    controls: List[int]
    ratio: Q                              # (c'u)^2 / u'Gu at the incumbent, y'y - incumbent
    sizes_after_hull: List[int]
    sizes: Dict[int, dict] = field(default_factory=dict)
    nodes: int = 0
    open: List[tuple] = field(default_factory=list)
    certified: bool = False
    seconds: float = 0.0
    hull_seconds: float = 0.0


def _branch(c, G, out, m, cut, fixes, rounds, per_round, depth, max_depth, inc):
    """Prune the node by its certificate, or branch on its most fractional geo; a
    fully fixed node is a design, evaluated exactly, and one better than the
    incumbent replaces it. The nodes after it are searched at the new ratio; those
    before it were pruned at a smaller ratio, the stronger claim, and no other node
    holds the new design, since the fixings partition the designs."""
    p = len(c)
    ones = sum(v for _, v in fixes)
    if ones > m or len(fixes) - ones > p - m:
        return
    out.nodes += 1
    if len(fixes) == p:
        C = {j for j, v in fixes if v}
        cu, r = _ratio(c, G, C)
        if cu > 0 and r > inc["lam"]:
            inc.update(lam=r, C=C)
        return
    res = size_certificate(c, G, inc["lam"], m, cut=cut if cut is not None and len(cut) == m else None,
                           fixes=fixes, rounds=rounds, per_round=per_round)
    if res.certified:
        return
    if depth >= max_depth:
        out.open.append((m, tuple(fixes)))
        return
    free = [j for j in range(p) if j not in dict(fixes)]
    j = min(free, key=lambda j: abs(res.u[j] - 0.5)) if res.u is not None else free[0]
    for v in (1, 0):
        _branch(c, G, out, m, cut, fixes + [(j, v)], rounds, per_round, depth + 1, max_depth, inc)


def certify_slope_free(y: np.ndarray, X: np.ndarray, *, restarts: int = 50, seed: int = 0,
                       rounds: int = 6, per_round: int = 400, max_depth: int = 12,
                       hull: bool = True) -> SlopeFreeCertificate:
    """Certify that no control group with a positive slope has a lower RSS than the incumbent.

    The incumbent comes from the study's RSS descent (search.incumbent). The hull
    stage (search.hull_survivors) rules out every size none of whose slope pieces
    survives it; each remaining size is certified by the slope-free relaxation at
    the incumbent's ratio, with triangle cuts, and branched on where it is not.
    """
    from .search import hull_survivors, incumbent, pieces_to

    t_all = time.perf_counter()
    p = X.shape[1]
    _, C = incumbent(y, X, restarts=restarts, seed=seed)
    D = ExactData(y, X)
    c, G = D.c, D.G
    cu, lam = _ratio(c, G, C)
    if not cu > 0:
        raise ValueError("no control group has a positive slope: every design fails the "
                         "correlation gate, so there is no incumbent to certify")
    sizes = list(range(1, p + 1)); t_h = 0.0
    cap = D.slope_cap()
    if hull and cap is not None:
        t0 = time.perf_counter()
        b_inc = float(lam / cu)           # c'u / u'Gu, since lam = (c'u)^2 / u'Gu
        s1, _ = hull_survivors(y, X, D, pieces_to(b_inc, cap), D.yy - lam)
        sizes = sorted({pc[2] for pc in s1})
        t_h = time.perf_counter() - t0
    inc = {"lam": lam, "C": set(C)}
    out = SlopeFreeCertificate(incumbent=D.yy - lam, controls=sorted(C), ratio=lam, sizes_after_hull=sizes,
                               hull_seconds=t_h)
    for m in sizes:                      # a size certified at a ratio stays certified at a larger one
        t0 = time.perf_counter(); n0 = out.nodes
        _branch(c, G, out, m, inc["C"], [], rounds, per_round, 0, max_depth, inc)
        out.sizes[m] = dict(nodes=out.nodes - n0, seconds=time.perf_counter() - t0)
    out.incumbent, out.ratio, out.controls = D.yy - inc["lam"], inc["lam"], sorted(inc["C"])
    out.certified = not out.open
    out.seconds = time.perf_counter() - t_all
    return out
