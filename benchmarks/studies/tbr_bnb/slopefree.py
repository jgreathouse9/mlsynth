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
  * no-good cuts sum_C u - sum_notC u <= |C| - 1, one per excluded design and one for
    the incumbent, so the incumbent's size can fall below zero; each excludes its C
    and no other binary vector at any size (Balas and Jeroslow 1972), since a design
    of size m sharing t geos with C has 2t - m <= |C| - 1 unless it is C (derived here);
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

Since the bound holds at any multipliers, and for any subset of the rows (a
relaxation with fewer rows contains more points, and every design among them), a
size is tried by a cascade of certificates, cheapest first (size_certificate):
the previous size's multipliers at this size's rows, screened in doubles and
then evaluated exactly; multipliers from a first-order method on the certificate
itself (dual_multipliers: L-BFGS-B on its smoothing, warm-started from the previous
size), evaluated exactly; the Shor relaxation, which keeps the equalities, c'u >= 0
and the cuts and drops the bound products and the products of c'u; then the full
relaxation with its rounds of triangles. On the 242 root sizes of the four-geo
GeoLift search the first closed 43 of the 239 that had a previous size, the
first-order method 240 of 242 at 0.04 s a size, the Shor relaxation 212 of 242 at
0.37 s a solve, against 1.07 s for the full relaxation.

The smoothing of T max(0, lambda_max(S)) is max {<Z, S> - a/2 ||Z||^2 : Z >= 0,
tr Z <= T}, within a T^2 / 2 of it, with the maximiser given by projecting the
eigenvalues of S / a onto {z >= 0, sum z <= T} (derived here; dual.smoothed_lbfgs
smooths the joint lift's dual by Young's inequality instead), minimised by
L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995).

Balas, E. and Jeroslow, R. (1972). Canonical cuts on the unit hypercube. SIAM J.
    Appl. Math. 23, 61-69.
Borrero, J. S., Gillen, C. and Prokopyev, O. A. (2017). Fractional 0-1 programming:
    applications and algorithms. J. Global Optim. 69, 255-282.
Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm for bound
    constrained optimization. SIAM J. Sci. Comput. 16, 1190-1208.
Dinkelbach, W. (1967). On nonlinear fractional programming. Manag. Sci. 13, 492-498.
Duchi, J., Shalev-Shwartz, S., Singer, Y. and Chandra, T. (2008). Efficient projections
    onto the l1-ball for learning in high dimensions. Proc. 25th ICML, 272-279.
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

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.marex_helpers.branch import _from_fractions, lambda_min_lower

from .certificate import ExactData, _dot_err, round_down


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
    constraints written here whatever the solver did.

    With ``m`` None the rows hold the designs of every size: sum u >= 1 and its
    products with 1 - u_i, (sum u - 1)(1 - u_i) >= 0, replace sum u = m and its
    products (the products with u_i follow from U_ii = u_i and U_ij >= 0), and the
    trace bound is 1 + p (group_bound)."""

    def __init__(self, c: Sequence, m: Optional[int], cuts: Sequence[Set[int]] = (),
                 triangles: Sequence[Tuple[int, Tuple[int, int, int]]] = (),
                 fixes: Sequence[Tuple[int, int]] = (), level: str = "full"):
        if level not in ("full", "shor"):
            raise ValueError(f"the rows' level is 'full' or 'shor', not {level!r}")
        p = len(c)
        self.p, self.n, self.m, self.level = p, p + 1, m, level
        full = level == "full"
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
        if m is not None:
            eq(row(*((u(i), 1) for i in range(p))), m)
            for i in range(p):
                eq(row(*((U(i, j), 1) for j in range(p)), (u(i), -m)), 0)
        else:
            le(row(*((u(j), -1) for j in range(p))), -1)                      # sum u >= 1
            for i in range(p):                                                 # (sum u - 1)(1 - u_i) >= 0
                le(row(*((u(j), -1) for j in range(p)), (u(i), -1), *((U(i, j), 1) for j in range(p))), -1)
        for j, v in fixes:
            eq(row((u(j), 1)), v)
        for i, j in itertools.combinations(range(p) if full else (), 2):
            le(row((U(i, j), -1)), 0)
            le(row((U(i, j), 1), (u(i), -1)), 0)
            le(row((U(i, j), 1), (u(j), -1)), 0)
            le(row((u(i), 1), (u(j), 1), (U(i, j), -1)), 1)
        le(row(*((u(j), -c[j]) for j in range(p))), 0, True)                      # c'u >= 0
        for i in range(p if full else 0):
            le(row(*((U(i, j), -c[j]) for j in range(p))), 0, True)              # u_i c'u >= 0
            le(row(*((U(i, j), c[j]) for j in range(p)), *((u(j), -c[j]) for j in range(p))), 0, True)
        for cut in cuts:
            le(row(*((u(j), 1 if j in cut else -1) for j in range(p))), len(cut) - 1)
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
    exact, so the multipliers return to the data's units by those two factors.

    Clarabel runs on one thread. Its automatic threading starts a native thread
    pool on a relaxation of the GeoLift groups' size, and a process forked after
    that (a worker of certify_treatment's pool) inherits the pool's state but not
    its threads, so its next solve never returns."""
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
    prob.solve(solver="CLARABEL", max_threads=1)   # a thread pool would hang a worker forked after it
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
class Multipliers:
    """A certificate's multipliers and the level of the rows they belong to."""
    level: str
    y_eq: np.ndarray
    y_in: np.ndarray


@dataclass
class SizeResult:
    certified: bool
    bound: Optional[Q]
    rounds: int
    triangles: int
    seconds: float
    u: Optional[np.ndarray] = None       # the relaxation's last u, for branching
    tier: str = "full"                   # the cascade's tier that decided: reuse, dual, shor or full
    multipliers: Optional[Multipliers] = None   # those of the certificate that closed it


def _screen(A, R: Rows, y_eq, y_in, m) -> float:
    """bound() in doubles, unrounded: it decides only whether the exact bound is
    tried, never whether a size is certified."""
    n = R.n
    Me, Mi = R.matrices(1.0)
    y_in = np.maximum(np.asarray(y_in, dtype=float), 0.0)
    E = (Me.T @ np.asarray(y_eq, dtype=float) + Mi.T @ y_in).reshape(n, n, order="F")
    S = np.zeros((n, n)); S[1:, 1:] = np.asarray(A, dtype=float); S -= (E + E.T) / 2
    k = float(np.dot(y_eq, np.asarray(R.beq, dtype=float)) + np.dot(y_in, np.asarray(R.bin, dtype=float)))
    return k + (1 + m) * max(0.0, float(np.linalg.eigvalsh(S)[-1]))


def _capped(v, T):
    """The projection of v onto {z >= 0, sum z <= T}: max(v - theta, 0), with
    theta >= 0 the smallest shift that brings the sum to at most T. When the sum of
    max(v, 0) is above T the projection is onto the scaled simplex, by the sort and
    scan of Duchi, Shalev-Shwartz, Singer and Chandra (2008); the capped form is
    derived here from its KKT conditions."""
    z = np.maximum(v, 0.0)
    if z.sum() <= T:
        return z
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    ok = np.nonzero(u - (css - T) / np.arange(1, len(u) + 1) > 0)[0]
    rho = ok[-1] if ok.size else 0       # in doubles none passes when u_1 dwarfs T; the largest is kept
    z = np.maximum(v - (css[rho] - T) / (rho + 1), 0.0)
    if not z.any():
        z = np.where(v == u[0], 1.0, 0.0)
    return z * min(1.0, T / z.sum())


class _Dual:
    """The certificate of bound() as a function of the multipliers, in the solver's
    units (the objective divided by s, the rows carrying c by cs, as solve does):
    F(y) = b'y + T max(0, lambda_max(S(y))), T = 1 + m, and its smoothing
    b'y + max {<Z, S(y)> - a/2 ||Z||^2 : Z >= 0, tr Z <= T}, whose maximiser is
    Q diag(_capped(lambda / a, T)) Q' and whose gradient in y is b - rows(Z)."""

    def __init__(self, A, R: Rows, m, c):
        A = np.asarray(A, dtype=float)
        self.n, self.T, self.ne = R.n, 1.0 + m, R.n_eq
        self.s = _pow2(float(np.abs(A).max()))
        self.cs = _pow2(float(np.max(np.abs(np.asarray(c, dtype=float)))))
        Me, Mi = R.matrices(self.cs)
        self.M = sp.vstack([Me, Mi]).tocsr()
        self.MT = self.M.T.tocsr()
        self.b = np.concatenate([np.asarray(R.beq, dtype=float), np.asarray(R.bin, dtype=float)])
        self.size = len(self.b)
        self.c_idx = self.ne + np.asarray(R.c_rows, dtype=int)
        self.A0 = np.zeros((self.n, self.n)); self.A0[1:, 1:] = A / self.s
        self.scale = max(1e-300, float(np.abs(np.linalg.eigvalsh(self.A0)).max()))

    def S(self, y):
        E = (self.MT @ y).reshape(self.n, self.n, order="F")
        return self.A0 - (E + E.T) / 2

    def value(self, y):
        return float(self.b @ y + self.T * max(0.0, np.linalg.eigvalsh(self.S(y))[-1]))

    def smoothed(self, y, a):
        return self.smoothed_and_value(y, a)[:2]

    def smoothed_and_value(self, y, a):
        """The smoothing and its gradient, and value(y) from the same eigendecomposition:
        lambda_max is the last of eigh's eigenvalues, so S(y) is built and decomposed
        once an evaluation, not twice."""
        lam, Q = np.linalg.eigh(self.S(y))
        z = _capped(lam / a, self.T)
        Z = (Q * z) @ Q.T
        by = float(self.b @ y)
        return (by + float(z @ lam) - a / 2 * float(z @ z), self.b - self.M @ Z.reshape(-1, order="F"),
                by + self.T * max(0.0, float(lam[-1])))

    def start(self, w: Optional["Multipliers"]):
        y = np.zeros(self.size)
        if w is not None and w.level == "full" and len(w.y_eq) + len(w.y_in) == self.size:
            y[:self.ne] = np.asarray(w.y_eq, dtype=float) / self.s
            y[self.ne:] = np.maximum(np.asarray(w.y_in, dtype=float), 0.0) / self.s
            y[self.c_idx] *= self.cs
        return y

    def back(self, y):
        """The multipliers in the data's units, as bound() reads them."""
        out = np.asarray(y, dtype=float) * self.s
        out[self.c_idx] /= self.cs
        return out[:self.ne], out[self.ne:]


#: The dual's evaluations before it gives up: on the 242 root sizes of the four-geo
#: GeoLift search the 240 it closed took a median of 9, at most 309.
DUAL_BUDGET = 400


def dual_multipliers(A, R: Rows, m, c, warm: Optional["Multipliers"] = None,
                     maxfun=150, stages=6, budget: Optional[int] = None):
    """Multipliers at which the float certificate of R is below zero, or None.

    L-BFGS-B (bounds y_in >= 0) on the smoothed certificate of _Dual, from ``warm``
    when it is the full rows' and of their shape, else from zero; the smoothing
    starts with a T^2 / 2 at about |F| at the start and shrinks tenfold per stage,
    and the search stops at the first iterate whose unsmoothed F is below zero, or
    gives up after ``budget`` evaluations (DUAL_BUDGET when not given). The
    float value only decides whether the exact bound is tried: size_certificate
    certifies with bound() at the multipliers returned."""
    from scipy.optimize import Bounds, minimize

    D = _Dual(A, R, m, c)
    y = D.start(warm)
    info = {"evals": 1}
    f0 = D.value(y)
    if f0 < 0:
        return Multipliers("full", *D.back(y)), info
    best = [f0, y]

    limit = DUAL_BUDGET if budget is None else budget

    class _Closed(Exception):
        pass

    class _Spent(Exception):
        pass

    def fun(yv, a):
        info["evals"] += 1
        if info["evals"] > limit:
            raise _Spent
        f, g, true = D.smoothed_and_value(yv, a)
        if true < best[0]:
            best[0], best[1] = true, yv.copy()
        if true < 0:
            raise _Closed
        return f, g

    # one Bounds for every stage: a list of (lo, hi) pairs is converted on each call
    bounds = Bounds(np.r_[np.full(D.ne, -np.inf), np.zeros(D.size - D.ne)], np.full(D.size, np.inf))
    a = max(abs(f0), 1e-6 * D.scale) / D.T ** 2   # a start at F = 0 would leave a near zero and lambda / a past 1e17
    try:
        for _ in range(stages):
            y = minimize(fun, y, args=(a,), jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxfun=maxfun, maxcor=20)).x
            a *= 0.1
    except _Closed:
        return Multipliers("full", *D.back(best[1])), info
    except _Spent:
        pass
    return None, info


def size_certificate(c, G, lam, m, cuts=(), fixes=(), rounds=6, per_round=400,
                     warm: Optional[Multipliers] = None) -> SizeResult:
    """Whether no design of size m (within the fixings) has u'(cc' - lam G)u > 0,
    by the cheapest certificate of a cascade (module docstring): ``warm``, another
    size's multipliers, at this size's rows; then the Shor relaxation's; then the
    full relaxation's, with up to ``rounds`` rounds of the ``per_round`` most violated
    triangle inequalities where it is above zero."""
    t0 = time.perf_counter()
    lam = dyadic_below(lam)
    A = objective(c, G, lam)
    if warm is not None:
        R = Rows(c, m, cuts=cuts, fixes=fixes, level=warm.level)
        if len(warm.y_eq) == R.n_eq and len(warm.y_in) == R.n_in and \
                _screen(A, R, warm.y_eq, warm.y_in, m) <= 0:
            b = bound(A, R, warm.y_eq, warm.y_in, m)
            if b is not None and b <= 0:
                return SizeResult(True, b, 0, 0, time.perf_counter() - t0, tier="reuse", multipliers=warm)
    R = Rows(c, m, cuts=cuts, fixes=fixes)
    w, _ = dual_multipliers(A, R, m, c, warm=warm)
    if w is not None:
        b = bound(A, R, w.y_eq, w.y_in, m)
        if b is not None and b <= 0:
            return SizeResult(True, b, 0, 0, time.perf_counter() - t0, tier="dual", multipliers=w)
    R = Rows(c, m, cuts=cuts, fixes=fixes, level="shor")
    try:
        sol = solve(c, G, lam, R)
    except cp.error.SolverError:
        sol = None                       # the full relaxation decides
    if sol is not None:
        b = bound(A, R, sol.y_eq, sol.y_in, m)
        if b is not None and b <= 0:
            return SizeResult(True, b, 0, 0, time.perf_counter() - t0, tier="shor",
                              multipliers=Multipliers("shor", sol.y_eq, sol.y_in))
    tri: List = []
    for r in range(rounds + 1):
        R = Rows(c, m, cuts=cuts, triangles=tri, fixes=fixes)
        sol = solve(c, G, lam, R)
        b = bound(A, R, sol.y_eq, sol.y_in, m)
        u = None if sol.Z is None else np.asarray(sol.Z)[0, 1:]
        if b is not None and b <= 0:
            w = Multipliers("full", sol.y_eq, sol.y_in) if not tri else None
            return SizeResult(True, b, r, len(tri), time.perf_counter() - t0, u, multipliers=w)
        new = [] if sol.Z is None else separate_triangles(sol.Z[0, 1:], sol.Z[1:, 1:], k=per_round)
        new = [(kind, idx) for _, kind, idx in new if (kind, idx) not in set(tri)]
        if not new or r == rounds:
            return SizeResult(False, b, r, len(tri), time.perf_counter() - t0, u)
        tri += new
    return SizeResult(False, None, rounds, len(tri), time.perf_counter() - t0)  # pragma: no cover - the loop returns


@dataclass
class GroupResult:
    certified: bool
    bound: Optional[Q]
    tier: str                            # dual, full, or open when neither closed it
    seconds: float


def group_bound(A, R: Rows, y_eq, y_in) -> Optional[Q]:
    """bound() on the size-free rows (Rows with m None), whose trace is 1 + sum u,
    at most 1 + p."""
    return bound(A, R, y_eq, y_in, R.p)


def group_certificate(c, G, lam) -> GroupResult:
    """Whether no design of any size has u'(cc' - lam G)u > 0 (with c'u >= 0), by the
    size-free rows' certificate: the first-order multipliers first, then the
    relaxation's own (Clarabel). Certified at lam' = dyadic_below(lam) <= lam, so no
    design's ratio exceeds lam' either."""
    t0 = time.perf_counter()
    if len(c) == 0:
        raise MlsynthDataError("the group screen needs a candidate control: there is no candidate control.")
    lam = dyadic_below(lam)
    A = objective(c, G, lam)
    R = Rows(c, None)
    w, _ = dual_multipliers(A, R, R.p, c)
    b = None
    if w is not None:
        b = group_bound(A, R, w.y_eq, w.y_in)
        if b is not None and b <= 0:
            return GroupResult(True, b, "dual", time.perf_counter() - t0)
    try:
        sol = solve(c, G, lam, R)
    except cp.error.SolverError:        # pragma: no cover - Clarabel solves the size-free rows of every test panel
        return GroupResult(False, b, "open", time.perf_counter() - t0)
    b = group_bound(A, R, sol.y_eq, sol.y_in)
    if b is not None and b <= 0:
        return GroupResult(True, b, "full", time.perf_counter() - t0)
    return GroupResult(False, b, "open", time.perf_counter() - t0)


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
    ceiling_held: bool = False            # certified: no design has an RSS below the ceiling
    priced: int = 0                       # designs priced in batched subtrees
    priced_nodes: int = 0                 # subtrees priced instead of bounded
    exact: int = 0                        # priced designs the float bound could not discard
    tiers: Dict[str, int] = field(default_factory=dict)   # sizes and nodes decided by each tier
    group_tier: Optional[str] = None      # the group screen's tier, None when it was not tried
    group_seconds: float = 0.0


#: The cost model of the choice between pricing a subtree and bounding it, as MAREX's
#: branch and bound makes it (search_skeleton.cheaper_to_price): seconds for one
#: relaxation of a size, 1.5-1.9 s at the root on the GeoLift groups (4.3 s with a
#: round of triangles), and for one design in the batched pass, 0.6-2.0 us
#: (README, "Pricing"). The limit, 750,000 designs, prices sizes up to 5 of 38.
BOUND_SECONDS = 1.5
PRICE_SECONDS = 2e-6
PRICE_LIMIT = int(BOUND_SECONDS / PRICE_SECONDS)
#: Designs per batch, so that the gathered s-by-s blocks stay a few million doubles.
_CELLS = 2_000_000


def _floats(c, G, inc):
    """The doubles nearest c and G, made once per search and kept on ``inc``."""
    if "_floats" not in inc:
        inc["_floats"] = (np.array([float(v) for v in c]), np.array([[float(v) for v in r] for r in G]))
    return inc["_floats"]


def _upper_batch(fixed, cfF, GfF, combos):
    """An upper bound in doubles on (c'u)^2 / u'Gu for each row of ``combos`` (indices
    into the free geos), the fixed-in part given exactly in ``fixed``; -inf where c'u
    cannot be positive, inf where u'Gu cannot be bounded away from zero.

    c'u = c1 + sum_S c_j and u'Gu = G11 + 2 sum_S g1_j + sum_{S x S} G_ij, with c1,
    G11 and g1 the fixed-in sums, exact in rationals and then rounded. Each double sum
    of k terms errs by at most gamma_k times the sum of its terms' moduli, and each
    term's own rounding from the exact value by u times its modulus, so twice
    _dot_err(k + 1, moduli) bounds both (Bjorck 2015, Lemma 1.4.2, with the underflow
    allowance of certificate._dot_err; the factor 2 covers the product of the two
    relative errors, derived here). The bound is (|s| + e_s)^2 / (d - e_d), raised
    by inflate's 1e-6 for its own three roundings."""
    c1, G11, g1 = fixed
    k = combos.shape[1]
    s = c1[0] + cfF[combos].sum(axis=1)
    sa = abs(c1[0]) + np.abs(cfF)[combos].sum(axis=1)
    blk = GfF[combos[:, :, None], combos[:, None, :]]
    d = G11[0] + 2 * g1[0][combos].sum(axis=1) + blk.sum(axis=(1, 2))
    da = abs(G11[0]) + 2 * np.abs(g1[0])[combos].sum(axis=1) + np.abs(blk).sum(axis=(1, 2))
    e_s = 2 * _dot_err(k + 2, sa)
    e_d = 2 * _dot_err(k * k + k + 2, da)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ub = (np.abs(s) + e_s) ** 2 / (d - e_d) * (1.0 + 1e-6)
    ub = np.where(d - e_d > 0, ub, np.inf)
    return np.where(s + e_s > 0, ub, -np.inf)


def _fixed(c, G, cf, ins, F):
    c1 = sum((c[i] for i in ins), Q(0))
    G11 = sum((G[i, j] for i in ins for j in ins), Q(0))
    g1 = np.array([float(sum((G[i, j] for i in ins), Q(0))) for j in F])
    return (float(c1),), (float(G11),), (g1,)


def ratio_upper(c, G, cf, Gf, ins, S) -> float:
    """The batched bound for the one design ``ins`` + ``S`` (S a list of free geos)."""
    ins = sorted(ins); F = sorted(set(range(len(c))) - set(ins))
    fixed = _fixed(c, G, cf, ins, F)
    pos = [F.index(j) for j in S]
    c1, G11, g1 = fixed
    return float(_upper_batch(((c1[0],), (G11[0],), (g1[0],)), cf[F], Gf[np.ix_(F, F)],
                              np.array([pos]))[0])


def price_subtree(c, G, out, m, fixes, inc, excluded):
    """Price every design of size m within the fixings: discard those whose double
    bound (_upper_batch) is at most the incumbent's ratio, and evaluate the rest
    exactly, best bound first, adopting one that beats the incumbent and is not
    excluded. Every discard is on a rigorous bound, so the subtree is closed."""
    p = len(c)
    fx = dict(fixes)
    ins = sorted(j for j, v in fixes if v)
    F = [j for j in range(p) if j not in fx]
    k = m - len(ins)
    cf, Gf = _floats(c, G, inc)
    fixed = _fixed(c, G, cf, ins, F)
    cfF, GfF = cf[F], Gf[np.ix_(F, F)]
    lam_lo = round_down(inc["lam"])
    per = max(1, _CELLS // max(1, k * k))
    gen = itertools.combinations(range(len(F)), k)
    cands = []
    out.priced_nodes += 1
    while True:
        chunk = list(itertools.islice(gen, per))
        if not chunk:
            break
        combos = np.array(chunk, dtype=np.int64)
        out.priced += len(chunk)
        ub = _upper_batch(fixed, cfF, GfF, combos)
        for i in np.nonzero(ub > lam_lo)[0]:
            cands.append((float(ub[i]), chunk[i]))
    cands.sort(key=lambda t: -t[0])
    for ub, combo in cands:
        if Q(ub) <= inc["lam"]:                # every later bound is at most this one
            break
        C = set(ins) | {F[i] for i in combo}
        out.exact += 1
        cu, r = _ratio(c, G, C)
        if cu > 0 and r > inc["lam"] and C not in excluded:
            inc.update(lam=r, C=C)


def _branch(c, G, out, m, cuts, fixes, rounds, per_round, depth, max_depth, inc, excluded):
    """Prune the node by its certificate, or branch on its most fractional geo; a
    node whose fixings leave one design of size m (m geos fixed in, or p - m fixed
    out) is that design, evaluated exactly, and one better than the incumbent
    replaces it. Bounding such a node instead branches on geos with one value
    left, down to the depth limit: on GeoLift a size-9 node with nine geos fixed
    in was left open that way. The nodes after it are searched at the new ratio; those
    before it were pruned at a smaller ratio, the stronger claim, and no other node
    holds the new design, since the fixings partition the designs."""
    p = len(c)
    ones = sum(v for _, v in fixes)
    if ones > m or len(fixes) - ones > p - m:
        return
    out.nodes += 1
    if ones == m or len(fixes) - ones == p - m:   # the fixings leave one design: evaluate it
        C = {j for j, v in fixes if v}
        if ones < m:                              # every free geo is in
            C |= {j for j in range(p) if j not in dict(fixes)}
        cu, r = _ratio(c, G, C)
        if cu > 0 and r > inc["lam"] and C not in excluded:
            inc.update(lam=r, C=C)
        return
    if math.comb(p - len(fixes), m - ones) <= PRICE_LIMIT:   # cheaper_to_price at the cost model
        price_subtree(c, G, out, m, fixes, inc, excluded)
        return
    res = size_certificate(c, G, inc["lam"], m, cuts=cuts, fixes=fixes, rounds=rounds, per_round=per_round,
                           warm=None if fixes else inc.get("warm"))
    out.tiers[res.tier] = out.tiers.get(res.tier, 0) + 1
    if res.certified:
        if not fixes and res.multipliers is not None:
            inc["warm"] = res.multipliers       # the next size's first try
        return
    if depth >= max_depth:
        out.open.append((m, tuple(fixes)))
        return
    free = [j for j in range(p) if j not in dict(fixes)]
    j = min(free, key=lambda j: abs(res.u[j] - 0.5)) if res.u is not None else free[0]
    for v in (1, 0):
        _branch(c, G, out, m, cuts, fixes + [(j, v)], rounds, per_round, depth + 1, max_depth, inc, excluded)


class NoDesignLeft(ValueError):
    """No positive-slope design remains to certify: none has a positive slope, or
    every one near the incumbent is excluded."""


def _best_neighbour(c, G, seeds, excluded):
    """The best positive-slope design one add, drop or swap from any seed that is
    not excluded, or None."""
    p = len(c); cand = set()
    for S in seeds:
        S = set(S)
        for j in range(p):
            cand.add(frozenset(S ^ {j}))
            if j not in S:
                cand.update(frozenset((S - {k}) | {j}) for k in S)
    best = None
    for T in cand:
        if not T or set(T) in excluded:
            continue
        cu, r = _ratio(c, G, T)
        if cu > 0 and (best is None or r > best[0]):
            best = (r, set(T))
    return None if best is None else best[1]


def certify_slope_free(y: np.ndarray, X: np.ndarray, *, restarts: int = 50, seed: int = 0,
                       rounds: int = 6, per_round: int = 400, max_depth: int = 12,
                       hull: bool = True, exclude: Sequence[Set[int]] = (),
                       ceiling: Optional[Q] = None,
                       start: Optional[Set[int]] = None,
                       group_screen: bool = True) -> SlopeFreeCertificate:
    """Certify that no control group with a positive slope, other than the designs in
    ``exclude``, has a lower RSS than the incumbent.

    The incumbent comes from the study's RSS descent (search.incumbent), or, when that
    design is excluded, from its best add, drop or swap neighbour, or that of an
    excluded design, that is not. The hull stage (search.hull_survivors) rules out
    every size none of whose slope pieces survives it; each remaining size is
    certified by the slope-free relaxation at the incumbent's ratio, with a no-good
    cut for each excluded design and triangle cuts, and branched on where it is not.

    A caller that already holds a design passes it as ``start``, and the search starts
    from it unless the RSS descent finds one with a lower RSS. A start comes from a
    cheaper search (the treatment pass descends from 10 starts, this from
    ``restarts``), and every size is tested at the start's ratio: on GeoLift at five
    geos a pass design with RSS 5,089,437.7, where the descent reached 4,860,705.3,
    left sizes 13 to 17 open at the root and the certificate open at the depth limit.

    With a ``ceiling`` above the incumbent's RSS the question is narrower: whether any
    design has an RSS below the ceiling. Each size is then tested at the ratio
    y'y - ceiling, larger than the incumbent's, and a leaf replaces the incumbent only
    when it beats that ratio. When none does the ceiling is held: the incumbent is
    reported at its own RSS, and ``certified`` means no design is below the ceiling,
    not that the incumbent is optimal.

    Before any size is searched against a ceiling, ``group_screen`` tries one
    certificate for every size at that ratio (group_certificate); when it closes,
    the ceiling is held without the hull stage or a size.
    """
    from .search import hull_survivors, incumbent, pieces_to

    t_all = time.perf_counter()
    p = X.shape[1]
    excluded = [set(e) for e in exclude]
    C = set(incumbent(y, X, restarts=restarts, seed=seed)[1])
    D = ExactData(y, X)
    c, G = D.c, D.G
    if start is not None:                # kept unless the descent found a design with a lower RSS
        cs, ls = _ratio(c, G, set(start))
        cd, ld = _ratio(c, G, C)
        if cs > 0 and (not cd > 0 or ls >= ld):
            C = set(start)
    cu, lam = _ratio(c, G, C)
    if set(C) in excluded or not cu > 0:
        C = _best_neighbour(c, G, [C] + excluded, excluded)
        if C is None:
            if excluded:
                raise NoDesignLeft("no design is left: every positive-slope design near the "
                                 "incumbent is excluded")
            raise NoDesignLeft("no control group has a positive slope: every design fails the "
                               "correlation gate, so there is no incumbent to certify")
        cu, lam = _ratio(c, G, C)
    test = lam
    if ceiling is not None and D.yy - ceiling > lam:
        test = D.yy - ceiling              # only a design below the ceiling matters
    screen = None
    if group_screen and test > lam:
        screen = group_certificate(c, G, test)
        if screen.certified:               # no design of any size is below the ceiling
            return SlopeFreeCertificate(incumbent=D.yy - lam, controls=sorted(C), ratio=lam, sizes_after_hull=[],
                                        certified=True, ceiling_held=True, group_tier=screen.tier,
                                        group_seconds=screen.seconds,
                                        seconds=time.perf_counter() - t_all)
    sizes = list(range(1, p + 1)); t_h = 0.0
    cap = D.slope_cap()
    if hull and cap is not None:
        t0 = time.perf_counter()
        b_inc = float(lam / cu)           # c'u / u'Gu, since lam = (c'u)^2 / u'Gu
        s1, _ = hull_survivors(y, X, D, pieces_to(b_inc, cap), D.yy - test)
        sizes = sorted({pc[2] for pc in s1})
        t_h = time.perf_counter() - t0
    inc = {"lam": test, "C": set(C)}
    out = SlopeFreeCertificate(incumbent=D.yy - lam, controls=sorted(C), ratio=lam, sizes_after_hull=sizes,
                               hull_seconds=t_h, group_tier=None if screen is None else screen.tier,
                               group_seconds=0.0 if screen is None else screen.seconds)
    for m in sizes:                      # a size certified at a ratio stays certified at a larger one
        t0 = time.perf_counter(); n0 = out.nodes
        _branch(c, G, out, m, excluded + [set(inc["C"])], [], rounds, per_round, 0, max_depth, inc, excluded)
        out.sizes[m] = dict(nodes=out.nodes - n0, seconds=time.perf_counter() - t0)
    out.ceiling_held = inc["lam"] == test and test > lam
    if out.ceiling_held:                  # no design beat the ceiling: report the incumbent itself
        inc.update(lam=lam)
    out.incumbent, out.ratio, out.controls = D.yy - inc["lam"], inc["lam"], sorted(inc["C"])
    out.certified = not out.open
    out.ceiling_held &= out.certified
    out.seconds = time.perf_counter() - t_all
    return out
