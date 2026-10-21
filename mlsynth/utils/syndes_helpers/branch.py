"""SYNDES two-way above the candidate limit: a certified branch and bound (#803).

Below ``CANDIDATE_LIMIT`` admissible treated sets the exact backend
(:mod:`~mlsynth.utils.syndes_helpers.exact`) prices every one under the closed
form ``lb(S) = 4 alpha / sigma'R sigma`` of :mod:`~.gram`. Above it, the sets are
searched here, and the search returns the same optimum.

Nodes. A node fixes some units into the treated set (``I``), some out of it
(``O``), and leaves ``F`` free; its subtree is every admissible treated set of
size ``K`` that contains ``I`` and avoids ``O``. A node is either priced set by
set, exactly as the enumeration prices, or bounded and split on its first free
unit, in the order of the root relaxation's maximiser, into a child that
includes the unit and one that excludes it. Best first: the open node whose
parent's bound is smallest is bounded next, so a search stopped at its limit
has raised the smallest open bound as far as its work allowed; between equal
bounds the node pushed last goes first, so a split dives into the child that
includes. Until it has a design to return, from a seed, a cutoff or a set
priced or waiting to be, the search dives depth first, since best first can spend its limit near
the root and price none. Each open node carries the certified bound of its nearest lifted
ancestor, valid for every set below it; a stopped search reports, for each node
it left open, that bound or the node's own trust-region bound, whichever is
higher.

The node bound. Every set below the node has ``f(S) >= 4 alpha / U`` for any
``U >= max sigma'R sigma`` over the subtree. With ``sigma``'s decided part fixed
and ``x = sigma_F`` in ``{-1, +1}^n`` with ``1'x = s = 2r - n``, ``r = K - |I|``,

    sigma'R sigma = c + 2 b'x + x'A x                      (:func:`node_parts`).

:func:`trust_region` relaxes ``x`` to the sphere ``|x|^2 = n`` cut by the plane
``1'x = s``, a trust-region subproblem solved by Newton's method on the secular
equation (Moré and Sorensen 1983), whose iterates stay left of the root, so each
is a bound. :func:`lift` bounds the doubly nonnegative relaxation of the lifted
``X = [1; x][1; x]'`` (unit diagonal, ``X r = 0`` for ``r = [-s; 1]``, the
triangle inequalities) by the bounding procedure of BiqCrunch (Krislock, Malick
and Roupin): L-BFGS-B on a regularised dual, ``alpha`` halved from 0.1 to 5e-5,
triangle inequalities added as cuts when violated by more than 0.05 and dropped
when slack with a zero multiplier, read from BiqCrunch's source for the procedure
and its defaults, nothing copied. Every evaluation reports the eigenvalue bound,
so the lift stops as soon as the node prunes. No interior-point solve is made:
on 32 to 40 free units the lift is one to seven times faster than Clarabel on
the same relaxation and within 0.4% of its bound
(``benchmarks/studies/syndes_global_bnb``).

The certificate. Both bounds are read as multipliers (:class:`Multipliers`):
``d`` on the unit diagonal, ``u`` on ``X r = 0`` and ``w >= 0`` on the triangle
inequalities. For any such multipliers and every design below the node,

    sigma'R sigma = <Q, X> <= sum(d) + sum(w) + (n + 1) lambda_max(C),

with ``Q = [[c, b'], [b, A]]`` and ``C = Q - Diag(d) - (u r' + r u') / 2 +
sum_k w_k S_k``, since ``X`` is semidefinite with trace ``n + 1``. A node is
discarded only when this bound, assembled in exact arithmetic from the closed
form's own ``alpha H - p p'`` and with ``lambda_max`` bounded above by
:func:`~mlsynth.utils.marex_helpers.branch.lambda_min_lower`, clears the
incumbent: the rigorous bound of Jansson, Chaykin and Keil (2007), as MAREX's
branch and bound certifies its own. A bound that cannot be certified, or a lift
that fails, leaves the node to be split.

Where the time goes. A node is bounded when pricing its subtree would cost more
than the trust region, weighted by the chance it prunes; a bounded node that
survives is lifted when pricing it costs more than a lift, weighted by the
lift's own chance. Each chance is estimated per subtree size as the search runs,
pooled across sizes (:class:`PruneRates`), and the lift's cost is learned from
the lifts paid for, in evaluations (:class:`LiftCost`). The costs are constants
and counts, never a clock, so a search makes the same decisions on every
machine; they decide where it spends time, never what it proves.

Restrictions. Forced and forbidden units start the root's ``I`` and ``O``;
forced units and stratum quotas decide how many sets a node holds
(:class:`~.enumeration.SearchSpace`), and a node with none is dropped; a unit
included at a node sends its conflict partners out; and every set priced passes
the filter. A bound over more sets than the admissible ones is still a bound.

Limits. ``work_limit`` caps the search's work in the cost model's seconds: the
sets priced, the bounds, the lifts' evaluations and the certificates, each at its
modelled cost, so it caps a search whose bounds do not prune as well as one whose
lifts do. It reads counts, never a clock, so a search stops at the same node on
every machine. A search that reaches it stops and returns its best design, with
the smallest certified bound over the nodes still open.

A cutoff. Searching every size (``K=None``), a design of one size bounds the
others: a node whose bound clears the best design of any size holds nothing
better. Given that design's value as ``cutoff``, the search discards on it, and
returns no design when nothing at its size is better.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .enumeration import SearchSpace
from .exact import (CHUNK, CERTIFY_RTOL, WORK_LIMIT, SubsetFilter,
                    _no_feasible_design, _resolve_open, _settle, _stream_from_space,
                    with_complements)
from .gram import BoundEngine, TwoWayProblem, maxcut_candidates, signs_from_subsets
from ..marex_helpers.branch import lambda_min_lower
from ..marex_helpers.certify import Dyadic, round_down
from ...exceptions import MlsynthEstimationError

#: A node is discarded when its bound is at least the incumbent times
#: ``1 - PRUNE_RTOL``: ties go, and nothing better than the incumbent does.
PRUNE_RTOL = 1e-12

#: Restarts of the swap search whose local optima seed the incumbent.
RESTARTS = 32

#: Seconds to price one set per unit of ``N``: 1.0 µs a set at 40 units on the
#: enumeration's path after #828, on one thread.
PRICE_SECONDS_PER_UNIT = 2.5e-8

# the lift's schedule, BiqCrunch's defaults on parts divided by their largest entry
ALPHA0, ALPHA_SCALE, ALPHA_MIN = 1e-1, 0.5, 5e-5
TOL0, TOL_SCALE, TOL_MIN = 1e-1, 0.95, 1e-2
CUT_GAP = 5e-2              # a triangle enters when violated by more than this
CUT_LIMIT = 500             # triangles added per round
MIN_CUTS = 50               # fewer added than this: halve alpha
ROUNDS_AT_ALPHA = 50        # or this many rounds at one alpha
MIN_ROUNDS, MAX_ROUNDS = 12, 100
EVALUATIONS = 2000          # per round
MAXCOR = 10
KAPPA = 1.0 / (1.0 + math.sqrt(3.0))   # scale of the triangles' multipliers

#: Evaluations a lift is expected to take before any is observed.
LIFT_EVALUATIONS = 300.0

# signs of (X_ij, X_ik, X_jk) in each type; type 1 has no negative edge, the
# others one positive edge: (i, j), (i, k), (j, k)
_SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int64)


def trust_region_seconds(n: int) -> float:
    """Seconds for one trust-region bound on ``n`` free units: 0.2 ms at 10 and
    0.55 ms at 40."""
    return 2e-4 + 5e-9 * n ** 3


def certificate_seconds(n: int) -> float:
    """Seconds to certify a bound on ``n`` free units in exact arithmetic: 1.4 ms
    at 10 and 11 ms at 40. Paid only by a bound that prunes."""
    return 6e-4 + 6.6e-6 * n ** 2


def evaluation_seconds(n: int) -> float:
    """Seconds for one evaluation of the lift's dual on ``n`` free units, an
    eigendecomposition of an ``(n + 1)``-row matrix: 0.11 ms at 10 and 0.24 ms
    at 40."""
    return 1e-4 + 2e-9 * (n + 1) ** 3


# ----------------------------------------------------------------------
# the node, its multipliers and its certificate
# ----------------------------------------------------------------------


def node_parts(R: np.ndarray, I: Sequence[int], O: Sequence[int],
               F: Sequence[int]) -> Tuple[float, np.ndarray, np.ndarray]:
    """``c``, ``b`` and ``A`` with ``sigma'R sigma = c + 2 b'x + x'A x``."""
    D = list(I) + list(O)
    sD = np.concatenate([np.ones(len(I)), -np.ones(len(O))])
    F = list(F)
    c = float(sD @ R[np.ix_(D, D)] @ sD) if D else 0.0
    b = R[np.ix_(F, D)] @ sD if D else np.zeros(len(F))
    return c, b, R[np.ix_(F, F)]


@dataclass(frozen=True)
class Cuts:
    """Triangle inequalities ``1 + s1 X_ij + s2 X_ik + s3 X_jk >= 0`` on
    ``i < j < k`` of a node's lifted indices (0 the homogenising one), with signs
    by type ``t`` in 1..4. Each holds at every design."""

    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    t: np.ndarray

    @staticmethod
    def empty() -> "Cuts":
        z = np.zeros(0, dtype=np.int64)
        return Cuts(z, z, z, z)

    def __len__(self) -> int:
        return len(self.i)

    def take(self, idx) -> "Cuts":
        return Cuts(self.i[idx], self.j[idx], self.k[idx], self.t[idx])

    def keys(self, size: int) -> np.ndarray:
        return ((self.i * size + self.j) * size + self.k) * 4 + (self.t - 1)

    def values(self, X: np.ndarray) -> np.ndarray:
        s = _SIGNS[self.t - 1]
        return (1.0 + s[:, 0] * X[self.i, self.j] + s[:, 1] * X[self.i, self.k]
                + s[:, 2] * X[self.j, self.k])

    @staticmethod
    def concat(a: "Cuts", b: "Cuts") -> "Cuts":
        return Cuts(*(np.concatenate([x, y]) for x, y in
                      ((a.i, b.i), (a.j, b.j), (a.k, b.k), (a.t, b.t))))


@lru_cache(maxsize=None)
def _triples(size: int) -> np.ndarray:
    idx = np.array([(i, j, k) for i in range(size) for j in range(i + 1, size)
                    for k in range(j + 1, size)], dtype=np.int64).reshape(-1, 3)
    idx.flags.writeable = False
    return idx


def separate(X: np.ndarray, *, gap: float = CUT_GAP, limit: int = CUT_LIMIT) -> Cuts:
    """The triangle inequalities ``X`` violates by more than ``gap``, most
    violated first, at most ``limit``."""
    T = _triples(X.shape[0])
    if not len(T):
        return Cuts.empty()
    xij, xik, xjk = X[T[:, 0], T[:, 1]], X[T[:, 0], T[:, 2]], X[T[:, 1], T[:, 2]]
    vals = 1.0 + (np.stack([xij, xik, xjk], axis=1) @ _SIGNS.T)      # (triples, 4)
    p, t = np.nonzero(vals < -gap)
    order = np.argsort(vals[p, t], kind="stable")[:limit]
    p, t = p[order], t[order]
    return Cuts(T[p, 0].copy(), T[p, 1].copy(), T[p, 2].copy(), (t + 1).astype(np.int64))


@dataclass(frozen=True)
class Multipliers:
    """A node bound's multipliers: ``diag`` on ``X_ii = 1`` and ``rows`` on
    ``X r = 0`` (each ``n + 1``, index 0 the homogenising one), ``weights`` on
    the triangle inequalities ``cuts``. Any values give a bound; a negative
    weight is read as zero."""

    diag: np.ndarray
    rows: np.ndarray
    cuts: Cuts
    weights: np.ndarray


def _cut_matrix(size: int, cuts: Cuts, w: np.ndarray) -> np.ndarray:
    """``sum_k w_k S_k``, with ``<S_k, X>`` the signed sum of cut ``k``'s edges."""
    M = np.zeros((size, size))
    if len(cuts):
        sg = _SIGNS[cuts.t - 1] * (0.5 * w)[:, None]
        for (p, q), col in (((cuts.i, cuts.j), 0), ((cuts.i, cuts.k), 1),
                            ((cuts.j, cuts.k), 2)):
            np.add.at(M, (p, q), sg[:, col])
            np.add.at(M, (q, p), sg[:, col])
    return M


class ExactR:
    """The closed form's matrix ``R = alpha H - p p'``, exactly, from the
    engine's ``alpha``, ``H`` and ``p``, and the certificate of a node bound on
    it."""

    def __init__(self, engine: BoundEngine):
        H, p = Dyadic.of(engine.H), Dyadic.of(engine.p)
        a = Dyadic.of(np.array(engine.alpha))
        aH = Dyadic(H.N * int(a.N), H.e + a.e)
        pp = Dyadic(np.multiply.outer(p.N, p.N), 2 * p.e)
        self.exact = aH - pp
        self.N = engine.N

    def certify(self, I: Sequence[int], O: Sequence[int], F: Sequence[int],
                K: int, m: Multipliers) -> Optional[Fraction]:
        """An upper bound on ``sigma'R sigma`` over the node's subtree, in exact
        arithmetic, from ``m``; ``None`` when ``lambda_max`` cannot be bounded."""
        n = len(F)
        size = n + 1
        s = 2 * (K - len(I)) - n
        D = list(I) + list(O)
        sD = np.array([1] * len(I) + [-1] * len(O), dtype=object)
        Rn, e = self.exact.N, self.exact.e
        Q = np.zeros((size, size), dtype=object)
        Q[...] = 0
        if D:
            Q[0, 0] = int(sD.dot(Rn[np.ix_(D, D)]).dot(sD))
            if n:
                b = Rn[np.ix_(list(F), D)].dot(sD)
                Q[0, 1:] = b
                Q[1:, 0] = b
        if n:
            Q[1:, 1:] = Rn[np.ix_(list(F), list(F))]
        d = Dyadic.of(m.diag)
        u = Dyadic.of(m.rows)
        w = Dyadic.of(np.maximum(np.asarray(m.weights, dtype=float), 0.0))
        E = max(e, d.e, u.e + 1, w.e + 1)
        r = np.array([-s] + [1] * n, dtype=object)
        C = Q * (1 << (E - e))
        C[np.diag_indices(size)] -= d.N * (1 << (E - d.e))
        C -= (np.multiply.outer(u.N, r) + np.multiply.outer(r, u.N)) * (1 << (E - u.e - 1))
        cuts = m.cuts
        if len(cuts):
            sg = _SIGNS[cuts.t - 1].astype(object) * (w.N * (1 << (E - w.e - 1)))[:, None]
            for (pp, qq), col in (((cuts.i, cuts.j), 0), ((cuts.i, cuts.k), 1),
                                  ((cuts.j, cuts.k), 2)):
                for a, b2, v in zip(pp.tolist(), qq.tolist(), sg[:, col].tolist()):
                    C[a, b2] += v
                    C[b2, a] += v
        low = lambda_min_lower(Dyadic(-C, E))
        if low is None:
            return None
        const = (Fraction(int(sum(d.N.tolist())), 1 << d.e)
                 + Fraction(int(sum(w.N.tolist())), 1 << w.e))
        return const - size * low


# ----------------------------------------------------------------------
# the bounds
# ----------------------------------------------------------------------


def _rows_cancelling(H: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Multipliers ``u`` on ``X r = 0`` that leave ``H - (u r' + r u') / 2``
    equal to ``H`` compressed to ``r``'s orthogonal complement: whatever ``H``
    does along ``r`` no design sees."""
    rr = float(r @ r)
    Hr = H @ r
    return (2.0 / rr) * Hr - (float(r @ Hr) / rr ** 2) * r


def trust_region(c: float, b: np.ndarray, A: np.ndarray, s: int
                 ) -> Tuple[float, np.ndarray, Multipliers]:
    """The trust-region bound on ``c + 2 b'x + x'A x`` over ``|x|^2 = n``,
    ``1'x = s``, its maximiser, and its multipliers."""
    n = len(b)
    if n == 0:
        return c, np.zeros(0), Multipliers(np.array([c]), np.zeros(1),
                                           Cuts.empty(), np.zeros(0))
    x0 = np.full(n, s / n)
    rho2 = n - s * s / n
    const = c + 2.0 * float(b @ x0) + float(x0 @ A @ x0)
    Q = np.empty((n + 1, n + 1))
    Q[0, 0], Q[0, 1:], Q[1:, 0], Q[1:, 1:] = c, b, b, A
    r = np.concatenate([[-float(s)], np.ones(n)])
    if rho2 <= 1e-12 * n:                         # r = 0 or r = n: x is decided
        v = np.concatenate([[1.0], x0])
        d = (Q @ v) * v
        return const, x0, Multipliers(d, _rows_cancelling(Q - np.diag(d), r),
                                      Cuts.empty(), np.zeros(0))
    # An orthonormal basis P of the plane's direction space, 1-perp, so the
    # constant direction never enters the spectrum: with it, its zero
    # eigenvalue would force mu > 0 when A is negative on the plane.
    P = np.linalg.svd(np.ones((1, n)))[2][1:].T
    B = P.T @ A @ P
    B = 0.5 * (B + B.T)
    g = P.T @ (b + A @ x0)
    lam, V = np.linalg.eigh(B)
    gt = V.T @ g
    top = float(lam[-1]) if len(lam) else 0.0

    live = gt != 0.0          # a direction the linear part does not reach adds nothing

    def zeta(mu: float) -> np.ndarray:
        return np.where(live, gt / np.where(live, mu - lam, 1.0), 0.0)

    def dual(mu: float) -> float:
        return const + mu * rho2 + float(gt @ zeta(mu))

    # The dual is convex in mu with derivative rho^2 - |z(mu)|^2, z = (mu I -
    # B)^-1 g. Its minimiser solves h(mu) = 1 / |z(mu)| = 1 / rho, and h is
    # concave and increasing on (top, inf) (Moré and Sorensen 1983), so Newton's
    # method on h started left of the root stays left of it. Every iterate is
    # above top, so every dual value is a bound.
    scale = max(abs(top), float(np.abs(lam).max(initial=0.0)), 1e-300)
    # without a linear part the dual is const + mu rho^2, smallest at mu = top
    mu = top + 1e-12 * scale if gt.any() else top
    target = 1.0 / math.sqrt(rho2)
    for _ in range(60):
        z = zeta(mu)
        z2 = float(z @ z)
        if z2 <= rho2:                            # at or right of the root: the hard case, or g = 0
            break
        h = 1.0 / math.sqrt(z2)
        dh = float(z @ (z / (mu - lam))) * h ** 3
        step = (target - h) / dh
        mu += step
        if step <= 1e-15 * max(abs(mu), scale):
            break
    t = dual(mu)
    x = x0 + P @ (V @ zeta(mu))
    d = np.concatenate([[t - mu * n], np.full(n, mu)])
    return t, x, Multipliers(d, _rows_cancelling(Q - np.diag(d), r),
                             Cuts.empty(), np.zeros(0))


def _theta(c: float, b: np.ndarray, A: np.ndarray, s: int, y: np.ndarray,
           cuts: Cuts, v: np.ndarray, alpha: float):
    """The eigenvalue bound, the regularised bound, the regularised bound's
    gradients in ``y`` and in ``v``, and ``X_alpha``.

    ``y``: the diagonal's ``n + 1`` multipliers (index 0 first), then
    ``1'x = s``'s, then the ``n`` rows ``sum_j X_ij - s X_0i = 0``. Both bounds
    hold for any ``y`` and any ``v >= 0``: a design's ``X`` has trace and
    Frobenius norm ``n + 1``, and by Young's inequality ``(n + 1) |C_+|_F <=
    |C_+|_F^2 / (2 alpha) + alpha (n + 1)^2 / 2``."""
    n = len(b)
    size = n + 1
    C = np.empty((size, size))
    C[0, 0] = 0.0
    C[0, 1:] = C[1:, 0] = b
    C[1:, 1:] = A
    yd, ys, yr = y[:size], y[size], y[size + 1:]
    C[np.diag_indices(size)] -= yd
    C[0, 1:] -= 0.5 * ys
    C[1:, 0] -= 0.5 * ys
    u = np.concatenate([[-float(s)], np.ones(n)])
    rows = np.concatenate([[0.0], yr])
    C -= 0.5 * (np.outer(rows, u) + np.outer(u, rows))
    w = KAPPA * v
    C += _cut_matrix(size, cuts, w)
    const = c + yd.sum() + ys * s + w.sum()
    lam, U = np.linalg.eigh(C)
    pos = np.maximum(lam, 0.0)
    X = (U * (pos / alpha)) @ U.T
    exact = const + size * lam[-1]
    smooth = const + float(pos @ pos) / (2.0 * alpha) + 0.5 * alpha * size * size
    gy = np.empty(len(y))
    gy[:size] = 1.0 - np.diag(X)
    gy[size] = s - X[0, 1:].sum()
    gy[size + 1:] = -(X[1:, :] @ u)
    gv = KAPPA * cuts.values(X) if len(cuts) else np.zeros(0)
    return exact, smooth, gy, gv, X


def _as_multipliers(c: float, s: int, y: np.ndarray, cuts: Cuts, v: np.ndarray,
                    unit: float) -> Multipliers:
    """:func:`_theta`'s multipliers on the parts divided by ``unit``, in the
    general form on the parts themselves: the constant ``c`` and ``1'x = s``'s
    multiplier move onto the diagonal's index 0, ``1'x = s`` being row 0 of
    ``X r = 0``."""
    n = (len(y) - 2) // 2
    diag = unit * y[:n + 1]
    diag[0] += c + unit * s * y[n + 1]
    rows = unit * np.concatenate([[y[n + 1]], y[n + 2:]])
    return Multipliers(diag, rows, cuts, unit * KAPPA * np.maximum(v, 0.0))


class _Pruned(Exception):
    pass


class _Converged(Exception):
    pass


def lift(c: float, b: np.ndarray, A: np.ndarray, s: int, *,
         target: Optional[float] = None) -> Tuple[float, Multipliers, int]:
    """The lift's bound by its regularised dual, never above the trust-region
    bound, with its multipliers and the dual evaluations it took. ``target``:
    stop once the bound is at or under it."""
    n = len(b)
    fallback, _, trm = trust_region(c, b, A, s)
    if n <= 1 or (target is not None and fallback <= target):
        return fallback, trm, 0
    unit = max(abs(c), float(np.abs(b).max()), float(np.abs(A).max()))
    if unit == 0.0:
        return fallback, trm, 0
    cs, bs, As = c / unit, b / unit, A / unit
    goal = -np.inf if target is None else target / unit
    floor = fallback / unit
    size = n + 1
    ny = 2 * n + 2
    y, cut, v = np.zeros(ny), Cuts.empty(), np.zeros(0)
    alpha, tol = ALPHA0, TOL0
    best = {"value": np.inf, "y": None, "cuts": None, "v": None}
    evaluations = [0]
    rounds = at_alpha = 0
    previous = None

    while True:
        rounds += 1
        at_alpha += 1
        last: Dict[str, object] = {}

        def fun(xv, cut=cut, alpha=alpha):
            exact, smooth, gy, gv, _ = _theta(cs, bs, As, s, xv[:ny], cut, xv[ny:], alpha)
            evaluations[0] += 1
            if exact < best["value"]:
                best.update(value=exact, y=xv[:ny].copy(), cuts=cut, v=xv[ny:].copy())
            last["x"], last["g"] = xv.copy(), (gy, gv)
            if min(best["value"], floor) <= goal:
                raise _Pruned
            return smooth, np.concatenate([gy, gv])

        def stop(xk, tol=tol):
            if "x" not in last or not np.array_equal(last["x"], xk):
                return  # pragma: no cover - L-BFGS-B calls back at the point it evaluated last; this keeps a stale gradient from deciding convergence
            gy, gv = last["g"]
            if (np.abs(gy).max() < tol
                    and (not len(gv) or -min(gv.min(), 0.0) / KAPPA < tol)):
                raise _Converged

        x0 = np.concatenate([y, v])
        bounds = [(None, None)] * ny + [(0.0, None)] * len(v)
        try:
            res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           callback=stop,
                           options={"maxcor": MAXCOR, "maxfun": EVALUATIONS,
                                    "maxiter": EVALUATIONS, "ftol": 0.0, "gtol": 0.0})
            xv = res.x
        except _Pruned:
            break
        except _Converged:
            xv = last["x"]
        y, v = xv[:ny], xv[ny:]
        _, smooth, gy, gv, X = _theta(cs, bs, As, s, y, cut, v, alpha)
        stalled = (previous is not None and abs(previous - smooth) <= 1e-6 * abs(smooth)
                   and alpha < 1e-4)
        previous = smooth
        giveup = rounds >= MIN_ROUNDS and (rounds >= MAX_ROUNDS or stalled)
        # drop the slack cuts whose multiplier is zero, add the violated ones
        violation = -min(gv.min(), 0.0) / KAPPA if len(gv) else 0.0
        if len(cut):
            keep = ~((v < 1e-8) & (cut.values(X) > violation / 10.0))
            cut, v = cut.take(keep), v[keep]
        found = separate(X)
        fresh = ~np.isin(found.keys(size), cut.keys(size))
        added = int(fresh.sum())
        if added:
            cut = Cuts.concat(cut, found.take(fresh))
            v = np.concatenate([v, np.zeros(added)])
        if giveup or (alpha == ALPHA_MIN and added == 0):
            break
        if added < MIN_CUTS or at_alpha > ROUNDS_AT_ALPHA:
            at_alpha = 0
            alpha = max(alpha * ALPHA_SCALE, ALPHA_MIN)
            tol = max(tol * TOL_SCALE, TOL_MIN)
    if best["value"] < floor:
        m = _as_multipliers(c, s, best["y"], best["cuts"], best["v"], unit)
        return unit * best["value"], m, evaluations[0]
    return fallback, trm, evaluations[0]


# ----------------------------------------------------------------------
# the cost model
# ----------------------------------------------------------------------


class PruneRates:
    """The chance a bound prunes a node, by subtree size.

    Each size's estimate is shrunk toward the rate over every size, which
    starts at one half: ``(hits + w p) / (tries + w)`` with ``p = (all hits +
    1) / (all tries + 2)``. Where no bound has pruned at any size, a size not
    yet seen starts low, and the search does not pay a bound or two at every
    size to learn what the others already showed."""

    def __init__(self, weight: float = 2.0):
        self.weight = weight
        self.hits: Dict[int, int] = {}
        self.tries: Dict[int, int] = {}
        self.all_hits = 0
        self.all_tries = 0

    def rate(self, size: int) -> float:
        pooled = (self.all_hits + 1.0) / (self.all_tries + 2.0)
        return ((self.hits.get(size, 0) + self.weight * pooled)
                / (self.tries.get(size, 0) + self.weight))

    def record(self, size: int, pruned: bool) -> None:
        self.tries[size] = self.tries.get(size, 0) + 1
        self.all_tries += 1
        if pruned:
            self.hits[size] = self.hits.get(size, 0) + 1
            self.all_hits += 1


class LiftCost:
    """Seconds per lift on ``n`` free units: :func:`evaluation_seconds` times
    the evaluations a lift takes, a prior of ``evaluations`` counted as ``weight``
    lifts updated by each lift paid for. The count is averaged in logarithms:
    a lift that prunes stops early, a hundred times under the prior, and an
    arithmetic mean lets two such pseudo-lifts outweigh two hundred real ones."""

    def __init__(self, evaluations: float = LIFT_EVALUATIONS, weight: float = 2.0):
        self.prior = math.log(evaluations)
        self.weight = weight
        self.total = 0.0
        self.count = 0

    def observe(self, evaluations: int) -> None:
        self.total += math.log(max(evaluations, 1))
        self.count += 1

    def __call__(self, n: int) -> float:
        mean = (self.weight * self.prior + self.total) / (self.weight + self.count)
        return math.exp(mean) * evaluation_seconds(n)


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BranchOutcome:
    """What the branch and bound returns for one size.

    ``lower`` is a certified bound on the optimum: the smallest bound over
    the nodes discarded or still open and the design's own, and never below the
    Rayleigh bound. ``closed`` says no node was left open; ``certified`` says
    the bound reaches the design (or, with a cutoff and no design, the cutoff),
    which a search stopped at its limit can show too. ``nodes`` counts the
    nodes bounded, ``lifts`` the lifts computed, ``pruned`` the nodes
    discarded on a certified bound, ``priced`` the sets priced in closed form
    and ``solved`` those solved because their closed form broke a sign
    condition. ``work`` is the work done in the cost model's seconds, the
    quantity ``work_limit`` caps. With a cutoff that no design at this size
    beats, ``treated`` and the weights are ``None`` and ``value`` is infinite.
    """

    treated: Optional[np.ndarray]
    value: float
    treated_weights: Optional[np.ndarray]
    control_weights: Optional[np.ndarray]
    duality_gap: float
    lower: float
    closed: bool
    certified: bool
    nodes: int
    lifts: int
    pruned: int
    priced: int
    solved: int
    work: float


def branch_and_bound(
    problem: TwoWayProblem,
    engine: BoundEngine,
    K: int,
    filt: SubsetFilter,
    *,
    incumbent_sets: Optional[np.ndarray] = None,
    restarts: int = RESTARTS,
    seed: int = 0,
    work_limit: float = WORK_LIMIT,
    cutoff: Optional[float] = None,
    price_seconds: Optional[float] = None,
    bound_seconds: Optional[Callable[[int], float]] = None,
    lift_seconds: Optional[Callable[[int], float]] = None,
) -> BranchOutcome:
    """The two-way design's optimum at size ``K`` by a certified branch and bound.

    ``cutoff``: the value of a design found elsewhere (another size); only a
    better one is sought. ``price_seconds`` (per set), ``bound_seconds`` and
    ``lift_seconds`` (per trust-region bound and per lift, as functions of the
    free units) replace the cost model; they change where the search spends
    time, never its answer.

    Raises
    ------
    MlsynthEstimationError
        When no admissible design exists, or the work limit is reached before
        one is found.
    """
    N, R, alpha = problem.N, engine.R, engine.alpha
    xr = ExactR(engine)
    four_alpha = Fraction(4.0 * alpha)
    keep = Fraction(1) - Fraction(PRUNE_RTOL)
    modelled = PRICE_SECONDS_PER_UNIT * N
    price = modelled if price_seconds is None else float(price_seconds)
    learned = LiftCost()
    # a bound's expected cost: the bound, and its certificate when it prunes
    if bound_seconds is None:
        def bound_cost(n, p):
            return trust_region_seconds(n) + p * certificate_seconds(n)
    else:
        def bound_cost(n, p):
            return bound_seconds(n)
    if lift_seconds is None:
        def lift_cost(n, p):
            return learned(n) + p * certificate_seconds(n)
    else:
        def lift_cost(n, p):
            return lift_seconds(n)
    bound_rates, lift_rates = PruneRates(), PruneRates()

    I0 = tuple(sorted(filt.forced_in))
    O0 = tuple(sorted(filt.forbidden))
    if set(I0) & set(O0):
        raise _no_feasible_design()
    free0 = [u for u in range(N) if u not in filt.forced_in and u not in filt.forbidden]
    partners: Dict[int, set] = {}
    for i, j in filt.conflict_pairs:
        partners.setdefault(int(i), set()).add(int(j))
        partners.setdefault(int(j), set()).add(int(i))

    best: List = [np.inf if cutoff is None else float(cutoff), None]
    floor = [np.inf]
    pending: List[Tuple[np.ndarray, np.ndarray]] = []
    buffer: List[np.ndarray] = []
    counts = {"priced": 0, "buffered": 0, "nodes": 0, "lifts": 0, "pruned": 0,
              "work": 0.0}

    def price_block(sets: np.ndarray) -> None:
        out = engine.bound(signs_from_subsets(N, sets))
        counts["priced"] += len(sets)
        if out.exact.any():
            j = int(np.argmin(np.where(out.exact, out.lower, np.inf)))
            if out.lower[j] < best[0]:
                best[0], best[1] = float(out.lower[j]), sets[j].copy()
        open_ = (~out.exact) & (out.lower < best[0])
        if open_.any():
            pending.append((sets[open_], out.lower[open_]))

    def flush() -> None:
        """Price the buffered subtrees in one call: a subtree a bound did not
        pay for is small, and one call per subtree is mostly call overhead."""
        if buffer:
            price_block(np.concatenate(buffer))
            buffer.clear()
            counts["buffered"] = 0

    def enumerate_subtree(space: SearchSpace) -> None:
        if space.size > CHUNK:
            flush()
            for block in _stream_from_space(space, filt, CHUNK):
                price_block(block)
            return
        for block in _stream_from_space(space, filt, CHUNK):
            buffer.append(block)
            counts["buffered"] += len(block)
        if counts["buffered"] >= CHUNK:
            flush()

    def discards(I, O, F, U: float, m: Optional[Multipliers]) -> bool:
        """Whether the bound ``U`` clears the incumbent, certified exactly."""
        if not (np.isfinite(best[0]) and m is not None and np.isfinite(U) and U > 0.0
                and 4.0 * alpha / U >= best[0] * (1.0 - PRUNE_RTOL)):
            return False
        counts["work"] += certificate_seconds(len(F))
        exact = xr.certify(I, O, F, K, m)
        if exact is None or exact <= 0 or not four_alpha >= Fraction(best[0]) * keep * exact:
            return False
        floor[0] = min(floor[0], round_down(four_alpha / exact))
        return True

    def space_of(I, O) -> SearchSpace:
        return SearchSpace.build(N, K, forced=I, forbidden=O, strata=filt.strata)

    # the incumbent: the caller's sets and the swap search's local optima
    seeds = []
    if incumbent_sets is not None and len(np.atleast_2d(incumbent_sets)):
        seeds.append(np.atleast_2d(np.asarray(incumbent_sets, dtype=np.int64)))
    if restarts:
        seeds.append(maxcut_candidates(engine, K, n_restarts=restarts, seed=seed))
    for block in seeds:
        # at K = N / 2 a seed's complement is a design of the same value
        for S in with_complements(block, N):
            if len(S) == K and filt.accepts(tuple(int(i) for i in S)):
                value = _settle(problem, engine, S)[0]
                if value < best[0]:
                    best[0], best[1] = value, S.copy()

    r0 = K - len(I0)
    if free0 and 0 < r0 < len(free0):
        _, x, _ = trust_region(*node_parts(R, I0, O0, free0), 2 * r0 - len(free0))
        order = tuple(free0[i] for i in np.argsort(-x, kind="stable"))
    else:
        order = tuple(free0)

    # A node is pushed only when it holds an admissible set. The root is the one
    # node that can hold none, or have r out of range, or force two units that
    # conflict: a child includes a unit only with its partners sent out.
    # An open node: its parent's bound (the key), a tie-break that sends the
    # node pushed last first, the node, and its nearest lifted ancestor's
    # certified bound.
    Node = Tuple[float, int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...],
                 SearchSpace, float]
    heap: List[Node] = []
    dive: List[Node] = []
    pushed = [0]

    def found() -> bool:
        # a design to return: a value, or a set priced or waiting to be
        return bool(np.isfinite(best[0]) or pending or buffer)

    def push(key, I, O, F, space, inherited) -> None:
        pushed[0] += 1
        entry = (key, -pushed[0], I, O, F, space, inherited)
        if found():
            heapq.heappush(heap, entry)
        else:
            dive.append(entry)

    def pop() -> Node:
        if dive and found():
            for waiting in dive:
                heapq.heappush(heap, waiting)
            dive.clear()
        return dive.pop() if dive else heapq.heappop(heap)

    root = space_of(I0, O0)
    if root.size and not any(partners.get(u, set()) & set(I0) for u in I0):
        push(-np.inf, I0, O0, order, root, -np.inf)
    closed = True
    while heap or dive:
        if counts["work"] >= work_limit:
            closed = False
            break
        entry = pop()
        key, _, I, O, F, space, inherited = entry
        r, n = K - len(I), len(F)
        bucket = space.size.bit_length()
        p_bound = bound_rates.rate(bucket)
        if r == 0 or r == n or space.size * price * p_bound <= bound_cost(n, p_bound):
            pricing = space.size * modelled
            if counts["work"] + pricing > work_limit and not (r == 0 or r == n):
                dive.append(entry)
                closed = False
                break
            counts["work"] += pricing
            enumerate_subtree(space)
            continue
        counts["nodes"] += 1
        counts["work"] += trust_region_seconds(n)
        parts = node_parts(R, I, O, F)
        s = 2 * r - n
        U, _, m = trust_region(*parts, s)
        hit = discards(I, O, F, U, m)
        bound_rates.record(bucket, hit)
        p_lift = lift_rates.rate(bucket)
        lifted = not hit and space.size * price * p_lift > lift_cost(n, p_lift)
        if lifted:
            counts["lifts"] += 1
            goal = (4.0 * alpha / (best[0] * (1.0 - PRUNE_RTOL))
                    if np.isfinite(best[0]) else None)
            U, m, evaluations = lift(*parts, s, target=goal)
            learned.observe(evaluations)
            counts["work"] += evaluations * evaluation_seconds(n)
            hit = discards(I, O, F, U, m)
            lift_rates.record(bucket, hit)
        if hit:
            counts["pruned"] += 1
            continue
        # a lift costs far more than its certificate, which the children keep
        if lifted and m is not None:
            counts["work"] += certificate_seconds(n)
            exact = xr.certify(I, O, F, K, m)
            if exact is not None and exact > 0:
                inherited = max(inherited, round_down(four_alpha / exact))
        if np.isfinite(U) and U > 0.0:
            key = max(key, 4.0 * alpha / U)
        u, rest = F[0], F[1:]
        out = partners.get(u, set())
        for child in ((I, O + (u,), rest),
                      (I + (u,), O + tuple(k for k in rest if k in out),
                       tuple(k for k in rest if k not in out))):
            sub = space_of(child[0], child[1])
            if sub.size:
                push(key, *child, sub, inherited)

    # what the limit left open: a decided node is priced; any other keeps its
    # inherited bound, or its own trust-region bound, certified, where that is
    # higher; and none goes below the Rayleigh bound
    open_lower = np.inf
    rayleigh = engine.global_lower_bound(K)
    for _, _, I, O, F, space, inherited in heap + dive:
        r, n = K - len(I), len(F)
        if r == 0 or r == n:
            enumerate_subtree(space)
            continue
        counts["work"] += trust_region_seconds(n)
        U, _, m = trust_region(*node_parts(R, I, O, F), 2 * r - n)
        if np.isfinite(U) and U > 0.0 and 4.0 * alpha / U > inherited:
            counts["work"] += certificate_seconds(n)
            exact = xr.certify(I, O, F, K, m)
            if exact is not None and exact > 0:
                inherited = max(inherited, round_down(four_alpha / exact))
        open_lower = min(open_lower, inherited)
    flush()

    solved = 0
    if pending:
        candidates = np.concatenate([p for p, _ in pending])
        bounds = np.concatenate([q for _, q in pending])
        best[0], best[1], solved = _resolve_open(problem, candidates, bounds,
                                                 best[0], best[1])
    if best[1] is None and cutoff is not None and root.size:
        # nothing at this size beats the cutoff: every discarded node had a
        # bound, and every priced set a value, at or above it
        lower = min(floor[0], open_lower, float(cutoff))
        lower = max(lower, rayleigh) if rayleigh is not None else lower
        return BranchOutcome(
            treated=None, value=np.inf, treated_weights=None, control_weights=None,
            duality_gap=0.0, lower=float(lower), closed=closed,
            certified=bool(lower >= float(cutoff) * (1.0 - CERTIFY_RTOL)),
            nodes=counts["nodes"], lifts=counts["lifts"], pruned=counts["pruned"],
            priced=counts["priced"], solved=solved, work=counts["work"])
    if best[1] is None:
        if not closed:
            raise MlsynthEstimationError(
                f"SYNDES two-way exact backend: the branch and bound reached its "
                f"work limit ({work_limit:g} s) before any admissible design was "
                "found. Raise work_limit.")
        raise _no_feasible_design()

    value, a, b, gap = _settle(problem, engine, best[1])
    lower = min(floor[0], open_lower, value - gap)
    if rayleigh is not None:
        lower = max(lower, rayleigh)
    scale = max(abs(value), 1e-300)
    return BranchOutcome(
        treated=np.sort(np.asarray(best[1], dtype=np.int64)), value=value,
        treated_weights=a, control_weights=b, duality_gap=gap, lower=float(lower),
        closed=closed,
        # a search stopped at its limit whose bound reaches its design has
        # proved it: certification reads the bound, not whether the tree closed
        certified=bool(gap / scale <= CERTIFY_RTOL
                       and value - lower <= CERTIFY_RTOL * scale),
        nodes=counts["nodes"], lifts=counts["lifts"], pruned=counts["pruned"],
        priced=counts["priced"], solved=solved, work=counts["work"])
