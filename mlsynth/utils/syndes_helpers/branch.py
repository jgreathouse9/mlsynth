"""SYNDES two-way above the candidate limit: a certified branch and bound (#803).

Below ``CANDIDATE_LIMIT`` admissible treated sets the exact backend
(:mod:`~mlsynth.utils.syndes_helpers.exact`) prices every one under the closed
form ``lb(S) = 4 alpha / sigma'R sigma`` of :mod:`~.gram`. Above it, the sets are
searched here, and the search returns the same optimum.

Nodes. A node fixes some units into the treated set (``I``), some out of it
(``O``), and leaves ``F`` free; its subtree is every admissible treated set of
size ``K`` that contains ``I`` and avoids ``O``. A node is either priced set by
set, exactly as the enumeration prices, or bounded and split on a free unit
into a child that includes the unit and one that excludes it. The unit is
chosen by reliability branching (:func:`reliable_unit`; Achterberg, Koch and
Martin 2005): each unit's mean rise in bound per split, by side, is recorded,
a unit with too few rises is tried by its children's trust-region bounds, and
the free units start in the order of the root relaxation's maximiser (#881).
Best first: the open node whose
parent's bound is smallest is bounded next, so a search stopped at its limit
has raised the smallest open bound as far as its work allowed; between equal
bounds the node pushed last goes first, so a split dives into the child that
includes. Until it has a design to return, from a seed, a cutoff or a set
priced or waiting to be, the search dives depth first, since best first can spend its limit near
the root and price none. Each open node carries the certified bound of its nearest lifted
ancestor, valid for every set below it; a stopped search reports the smallest
bound over the nodes it left open, each node's the higher of that bound and its
own trust-region bound (Limits, below).

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
(``benchmarks/studies/syndes_global_bnb``). The lift runs BLAS on one thread:
its matrices have at most ``n + 1`` rows, where a thread pool costs more to
coordinate than it saves. On four threads a root lift took 16 s at 60 units
against 0.8 s on one, on a quiet machine (#857). Pricing keeps the caller's
threads, since its blocks of sets are large products that threads speed up.

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
and counts, never a clock, so the same arithmetic makes the same decisions on
any machine; they decide where it spends time, never what it proves.

Restrictions. Forced and forbidden units start the root's ``I`` and ``O``;
forced units and stratum quotas decide how many sets a node holds
(:class:`~.enumeration.SearchSpace`), and a node with none is dropped; a unit
included at a node sends its conflict partners out; and every set priced passes
the filter. A bound over more sets than the admissible ones is still a bound.

Limits. ``work_limit`` caps the search's work in the cost model's seconds: the
sets priced, the bounds, the lifts' evaluations and the certificates, each at its
modelled cost, so it caps a search whose bounds do not prune as well as one whose
lifts do. It reads counts, never a clock, so the same arithmetic stops at the
same node on any machine. A search that reaches it stops and returns its best
design, with the smallest certified bound over the nodes still open. Only the smallest
decides the bracket, so the open nodes are bounded lowest first
(:func:`least_open_bound`), with what the limit left and at most
``BRACKET_SHARE`` of it past it; a node the budget does not reach keeps the
bound it inherited.

The loop. The queue, the work's ledger and the handling of the limit are the
loop MAREX's search uses too (:mod:`~mlsynth.utils.search_skeleton`, #846): this
module hands it the plan for a node, the pricing and the bound. Equal keys go
to the node pushed last, the search dives until it has a design, and the limit
is read before each node as well as before each pricing.

A cutoff. Searching every size (``K=None``), a design of one size bounds the
others: a node whose bound clears the best design of any size holds nothing
better. Given that design's value as ``cutoff``, the search discards on it, and
returns no design when nothing at its size is better.

Leaves. How the sets a node holds are priced is the search's one part that
depends on the design (:class:`TwoWayLeaves`). One-way's optimum at any set is at
least two-way's, so the same tree, with its bounds and certificates, searches
one-way's designs when its leaves are priced as one-way's
(:class:`~.one_way_exact.OneWayLeaves`, #820).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import ThreadpoolController

from .enumeration import SearchSpace
from .exact import (CHUNK, CERTIFY_RTOL, WORK_LIMIT, SubsetFilter,
                    _no_feasible_design, _resolve_open, _settle, _stream_from_space,
                    with_complements)
from .gram import BoundEngine, TwoWayProblem, maxcut_candidates, signs_from_subsets
from ..persp_bound import lambda_min_lower
from ..marex_helpers.certify import Dyadic, round_down
from ...exceptions import MlsynthEstimationError
from ..search_skeleton import (Bound, Ledger, NodeQueue, Price, cheaper_to_price,
                               least_open_bound, run)

#: A node is discarded when its bound is at least the incumbent times
#: ``1 - PRUNE_RTOL``: ties go, and nothing better than the incumbent does.
PRUNE_RTOL = 1e-12

#: Restarts of the swap search whose local optima seed the incumbent.
RESTARTS = 32

#: Seconds to price one set per unit of ``N``: 2.5 µs a set at 40 units on the
#: enumeration's path, on one thread, from 24 to 50 units (#845). Below
#: ``N = T + 1`` the closed form prices a set exactly; above it, a set it
#: cannot price is fitted at the end, which this does not count.
PRICE_SECONDS_PER_UNIT = 6.0e-8

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

#: Share of the work limit a stopped search may spend past it bounding the
#: nodes it left open (:func:`least_open_bound`), on top of what the limit left.
BRACKET_SHARE = 0.05

#: Reliability branching's defaults in Achterberg, Koch and Martin (2005):
#: eight rises a side before a unit's pseudo-cost is trusted, four trials in a
#: row without a better score, and a sixth of the larger rise in the score.
RELIABLE, LOOKAHEAD, MU = 8, 4, 1.0 / 6.0

# signs of (X_ij, X_ik, X_jk) in each type; type 1 has no negative edge, the
# others one positive edge: (i, j), (i, k), (j, k)
_SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int64)


def trust_region_seconds(n: int) -> float:
    """Seconds for one trust-region bound on ``n`` free units: 0.26 ms at 10,
    0.53 ms at 40 and 0.63 ms at 60, on one thread (#845)."""
    return 1.8e-4 + 8.5e-6 * n


def certificate_seconds(n: int) -> float:
    """Seconds to certify a bound on ``n`` free units in exact arithmetic: 1.7 ms
    at 10, 13 ms at 40 and 31 ms at 60, on one thread (#845). Paid only by a
    bound that prunes."""
    return 9e-4 + 8.4e-6 * n ** 2


def evaluation_seconds(n: int) -> float:
    """Seconds the lift spends per evaluation of its dual on ``n`` free units:
    0.51 ms at 10, 1.0 ms at 40 and 1.3 ms at 60, on one thread (#845). The
    eigendecomposition of an ``(n + 1)``-row matrix is a quarter to a third of
    it; the rest is the evaluation's other arithmetic, the cut matrix, SciPy's
    L-BFGS-B and the separation of triangles (#844). Up to 60 free units it
    grows about linearly."""
    return 3.4e-4 + 1.65e-5 * n


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


@lru_cache(maxsize=None)
def _blas() -> ThreadpoolController:
    """The process's BLAS pools, found once: a fresh scan costs about 7 ms,
    and limiting a found pool about 15 us. By the first lift numpy and SciPy
    have loaded every BLAS the lift calls."""
    return ThreadpoolController()


def lift(c: float, b: np.ndarray, A: np.ndarray, s: int, *,
         target: Optional[float] = None) -> Tuple[float, Multipliers, int]:
    """The lift's bound by its regularised dual, never above the trust-region
    bound, with its multipliers and the dual evaluations it took. ``target``:
    stop once the bound is at or under it.

    BLAS runs on one thread for the call, and the caller's thread counts come
    back on the way out, whether the lift returns or raises (#857)."""
    with _blas().limit(limits=1, user_api="blas"):
        return _lift(c, b, A, s, target=target)


def _lift(c: float, b: np.ndarray, A: np.ndarray, s: int, *,
          target: Optional[float] = None) -> Tuple[float, Multipliers, int]:
    n = len(b)
    fallback, _, trm = trust_region(c, b, A, s)
    if n <= 1 or (target is not None and fallback <= target):
        return fallback, trm, 0
    # The dual runs on the free part alone, at the free part's scale, and c is
    # added at the end: c adds to every bound (X_00 = 1) and shapes none. A
    # scale taken from c grew with the fixed units, shrank the free part
    # against the alpha schedule, and stopped deep nodes at the trust region.
    unit = max(float(np.abs(b).max()), float(np.abs(A).max()))
    if unit == 0.0:
        return fallback, trm, 0
    bs, As = b / unit, A / unit
    goal = -np.inf if target is None else (target - c) / unit
    floor = (fallback - c) / unit
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
            exact, smooth, gy, gv, _ = _theta(0.0, bs, As, s, xv[:ny], cut, xv[ny:], alpha)
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
        _, smooth, gy, gv, X = _theta(0.0, bs, As, s, y, cut, v, alpha)
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
        return c + unit * best["value"], m, evaluations[0]
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


class TwoWayLeaves:
    """The two-way design's pricing of the sets the search reaches.

    A block of sets is priced by the closed form of :mod:`~.gram`, exact where
    its minimiser keeps the sign conditions; a set whose closed form breaks one
    stays open with its closed form as its bound, and the open sets under the
    incumbent are solved at the end (:func:`~.exact._resolve_open`), uncharged.

    The search asks its leaves for six things: ``seconds_per_set``, the
    modelled cost of pricing one set; ``seeds(block)``, the designs a block of
    the swap search's sets stands for; ``value(S)``, a seed's value; ``price(sets,
    best)``, a block's best exact value and set, the sets left open under the
    incumbent with their bounds, and the seconds spent past ``seconds_per_set``;
    ``resolve(sets, bounds, value, S, budget)``, the incumbent after the open
    sets are priced, how many were, the seconds spent, and the least bound left
    open; and ``settle(S)``, the design's value, weights and duality gap.
    """

    def __init__(self, problem: TwoWayProblem, engine: BoundEngine):
        self.problem, self.engine = problem, engine
        self.seconds_per_set = PRICE_SECONDS_PER_UNIT * problem.N

    def seeds(self, block: np.ndarray) -> np.ndarray:
        # at K = N / 2 a seed's complement is a design of the same value
        return with_complements(block, self.problem.N)

    def value(self, S: np.ndarray) -> float:
        return _settle(self.problem, self.engine, S)[0]

    def price(self, sets: np.ndarray, best: float):
        out = self.engine.bound(signs_from_subsets(self.problem.N, sets))
        value, S = np.inf, None
        if out.exact.any():
            j = int(np.argmin(np.where(out.exact, out.lower, np.inf)))
            value, S = float(out.lower[j]), sets[j]
        open_ = (~out.exact) & (out.lower < min(best, value))
        return value, S, sets[open_], out.lower[open_], 0.0

    def resolve(self, sets: np.ndarray, bounds: np.ndarray, value: float, S,
                budget: float):
        value, S, solved = _resolve_open(self.problem, sets, bounds, value, S)
        return value, S, solved, 0.0, np.inf

    def settle(self, S: np.ndarray):
        return _settle(self.problem, self.engine, S)


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
    trials: int = 0


# ----------------------------------------------------------------------
# the branching unit
# ----------------------------------------------------------------------


class PseudoCosts:
    """Each unit's mean rise in bound per split of it, by side (``True`` for the
    child that includes it). A fall counts as no rise. An infinite rise, from a
    child with no admissible set, is not recorded: it would hold the unit's mean
    at infinity for the rest of the search."""

    def __init__(self) -> None:
        self.total: Dict[Tuple[int, bool], float] = {}
        self.count: Dict[Tuple[int, bool], int] = {}

    def record(self, u: int, include: bool, rise: float) -> None:
        if not np.isfinite(rise):
            return
        key = (u, include)
        self.total[key] = self.total.get(key, 0.0) + max(rise, 0.0)
        self.count[key] = self.count.get(key, 0) + 1

    def reliable(self, u: int) -> bool:
        """Whether ``u`` has ``RELIABLE`` rises recorded on both sides."""
        return min(self.count.get((u, True), 0),
                   self.count.get((u, False), 0)) >= RELIABLE

    def mean(self, u: int, include: bool, default: float) -> float:
        n = self.count.get((u, include), 0)
        return self.total[(u, include)] / n if n else default

    def overall(self) -> float:
        """The mean rise over every split recorded; one before any is."""
        n = sum(self.count.values())
        return sum(self.total.values()) / n if n else 1.0


def score(down: float, up: float) -> float:
    """A split's score from its two rises, ``(1 - MU) min + MU max``: the
    smaller rise counts most, since the search must bound both children."""
    return (1.0 - MU) * min(down, up) + MU * max(down, up)


def trial_bound(R: np.ndarray, alpha: float, K: int, I: Sequence[int],
                O: Sequence[int], F: Sequence[int]) -> Tuple[float, bool]:
    """A child's bound as a trial reads it, and whether it is a trust-region
    bound, charged as one. A child with every unit decided holds one set, and
    its closed form is the bound; a child whose size cannot be reached holds
    none, and its bound is infinite. A relaxation with no positive maximum
    gives no bound, read as zero."""
    r, n = K - len(I), len(F)
    if r < 0 or r > n:
        return np.inf, False
    if r == 0 or r == n:
        sigma = -np.ones(R.shape[0])
        sigma[list(I) + (list(F) if r == n else [])] = 1.0
        q = float(sigma @ R @ sigma)
        return (4.0 * alpha / q if q > 0.0 else 0.0), False
    U, _, _ = trust_region(*node_parts(R, I, O, F), 2 * r - n)
    return (4.0 * alpha / U if np.isfinite(U) and U > 0.0 else 0.0), True


def reliable_unit(F: Sequence[int], costs: PseudoCosts, here: float,
                  trial: Callable[[int], Tuple[float, float]]) -> int:
    """The free unit a node splits on, by reliability branching (Achterberg,
    Koch and Martin 2005).

    The units are taken in order of their pseudo-costs' score, the mean over
    every split standing in for a side never split, and ``F``'s order breaks
    ties: before any split it is the root relaxation's order. A reliable unit
    is scored by its pseudo-costs. Any other is tried: ``trial(k)`` returns the
    bounds of the children excluding and including ``k``, and their rises over
    the node's bound ``here`` are recorded and scored. The search ends after
    ``LOOKAHEAD`` trials in a row without a better score, and returns the unit
    that scored best, the first among equals.
    """
    default = costs.overall()

    def estimate(k: int) -> float:
        return score(costs.mean(k, False, default), costs.mean(k, True, default))

    order = sorted(F, key=lambda k: -estimate(k))
    top, top_score, stale = order[0], -np.inf, 0
    for k in order:
        if costs.reliable(k):
            s = estimate(k)
        else:
            down, up = (b - here for b in trial(k))
            costs.record(k, False, down)
            costs.record(k, True, up)
            s = score(max(down, 0.0), max(up, 0.0))
            stale += 1
        if s > top_score:
            top, top_score, stale = k, s, 0
        elif stale >= LOOKAHEAD:
            break
    return top


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
    leaves=None,
) -> BranchOutcome:
    """The two-way design's optimum at size ``K`` by a certified branch and bound.

    ``cutoff``: the value of a design found elsewhere (another size); only a
    better one is sought. ``price_seconds`` (per set), ``bound_seconds`` and
    ``lift_seconds`` (per trust-region bound and per lift, as functions of the
    free units) replace the cost model; they change where the search spends
    time, never its answer. ``leaves`` prices the sets the search reaches,
    :class:`TwoWayLeaves` by default; a design whose value at every set is at
    least two-way's is searched by passing its own.

    Raises
    ------
    MlsynthEstimationError
        When no admissible design exists, or the work limit is reached before
        one is found.
    """
    N, R, alpha = problem.N, engine.R, engine.alpha
    leaves = TwoWayLeaves(problem, engine) if leaves is None else leaves
    xr = ExactR(engine)
    four_alpha = Fraction(4.0 * alpha)
    keep = Fraction(1) - Fraction(PRUNE_RTOL)
    modelled = leaves.seconds_per_set
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
    costs = PseudoCosts()

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
              "trials": 0}
    ledger = Ledger(work_limit)

    def price_block(sets: np.ndarray) -> None:
        value, S, open_sets, open_bounds, seconds = leaves.price(sets, best[0])
        counts["priced"] += len(sets)
        ledger.charge(seconds)
        if value < best[0]:
            best[0], best[1] = float(value), S.copy()
        if len(open_sets):
            pending.append((open_sets, open_bounds))

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
        ledger.charge(certificate_seconds(len(F)))
        exact = xr.certify(I, O, F, K, m)
        if exact is None or exact <= 0 or not four_alpha >= Fraction(best[0]) * keep * exact:
            return False
        floor[0] = min(floor[0], round_down(four_alpha / exact))
        return True

    def space_of(I, O) -> SearchSpace:
        return SearchSpace.build(N, K, forced=I, forbidden=O, strata=filt.strata)

    def split(I, O, F, u):
        """The children on ``u``: one without it, and one with it and without
        its conflict partners."""
        out = partners.get(u, set())
        rest = tuple(k for k in F if k != u)
        return ((I, O + (u,), rest),
                (I + (u,), O + tuple(k for k in rest if k in out),
                 tuple(k for k in rest if k not in out)))

    def trials_at(I, O, F):
        """The node's trial: both children's bounds on a unit, a trust-region
        bound charged as one; a decided child's closed form, one quadratic form,
        is not charged."""
        def trial(k):
            bounds = []
            for cI, cO, cF in split(I, O, F, k):
                if not space_of(cI, cO).size:
                    bounds.append(np.inf)
                    continue
                b, bounded = trial_bound(R, alpha, K, cI, cO, cF)
                if bounded:
                    counts["trials"] += 1
                    ledger.charge(trust_region_seconds(len(cF)))
                bounds.append(b)
            return bounds[0], bounds[1]
        return trial

    # the incumbent: the caller's sets and the swap search's local optima
    seeds = []
    if incumbent_sets is not None and len(np.atleast_2d(incumbent_sets)):
        seeds.append(np.atleast_2d(np.asarray(incumbent_sets, dtype=np.int64)))
    if restarts:
        seeds.append(maxcut_candidates(engine, K, n_restarts=restarts, seed=seed))
    for block in seeds:
        for S in leaves.seeds(block):
            if len(S) == K and filt.accepts(tuple(int(i) for i in S)):
                value = leaves.value(S)
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
    # An open node, keyed by its parent's bound: its units in, out and free, its
    # sets, its nearest lifted ancestor's certified bound, and the split that
    # made it (the unit, the side, and the parent's trust-region bound, from
    # which its rise is recorded). Equal keys go to the node pushed last, so a
    # split continues into the child that includes its unit.
    def found() -> bool:
        # a design to return: a value, or a set priced or waiting to be
        return bool(np.isfinite(best[0]) or pending or buffer)

    queue = NodeQueue(last_first=True, found=found)

    def bound_node(key, node, bucket):
        I, O, F, space, inherited, origin = node
        r, n = K - len(I), len(F)
        counts["nodes"] += 1
        ledger.charge(trust_region_seconds(n))
        parts = node_parts(R, I, O, F)
        s = 2 * r - n
        U, _, m = trust_region(*parts, s)
        here = 4.0 * alpha / U if np.isfinite(U) and U > 0.0 else 0.0
        if origin is not None:
            costs.record(origin[0], origin[1], here - origin[2])
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
            ledger.charge(evaluations * evaluation_seconds(n))
            hit = discards(I, O, F, U, m)
            lift_rates.record(bucket, hit)
        if hit:
            counts["pruned"] += 1
            return None
        # a lift costs far more than its certificate, which the children keep
        if lifted and m is not None:
            ledger.charge(certificate_seconds(n))
            exact = xr.certify(I, O, F, K, m)
            if exact is not None and exact > 0:
                inherited = max(inherited, round_down(four_alpha / exact))
        if np.isfinite(U) and U > 0.0:
            key = max(key, 4.0 * alpha / U)
        u = reliable_unit(F, costs, here, trials_at(I, O, F))
        children = []
        for include, child in zip((False, True), split(I, O, F, u)):
            sub = space_of(child[0], child[1])
            if sub.size:
                children.append((key, (*child, sub, inherited, (u, include, here))))
        return children

    def plan(key, node):
        """Price a decided node, or one cheaper to price than to bound
        (weighted by how often bounds have discarded nodes of its size);
        bound any other. A decided node is priced whatever the limit."""
        I, O, F, space, _, _ = node
        r, n = K - len(I), len(F)
        bucket = space.size.bit_length()
        p_bound = bound_rates.rate(bucket)
        decided = r == 0 or r == n
        if decided or cheaper_to_price(space.size, price, bound_cost(n, p_bound), p_bound):
            return Price(cost=space.size * modelled, deferrable=not decided,
                         run=lambda: enumerate_subtree(space))
        return Bound(run=lambda ledger: bound_node(key, node, bucket))

    root = space_of(I0, O0)
    if root.size and not any(partners.get(u, set()) & set(I0) for u in I0):
        queue.push(-np.inf, (I0, O0, order, root, -np.inf, None))
    closed = run(queue, ledger, plan=plan, check_before_pop=True)

    # what the limit left open: a decided node is priced; any other keeps its
    # inherited bound, or its own trust-region bound, certified, where that is
    # higher, bounded lowest first within the bracket's budget; and none goes
    # below the Rayleigh bound
    rayleigh = engine.global_lower_bound(K)
    undecided = []
    for _, _, (I, O, F, space, inherited, _) in queue.open():
        r = K - len(I)
        if r == 0 or r == len(F):
            enumerate_subtree(space)
        else:
            undecided.append((inherited, (I, O, F)))

    def own_bound(inherited: float, node) -> Tuple[float, float]:
        I, O, F = node
        r, n = K - len(I), len(F)
        cost = trust_region_seconds(n)
        U, _, m = trust_region(*node_parts(R, I, O, F), 2 * r - n)
        if not (np.isfinite(U) and U > 0.0 and 4.0 * alpha / U > inherited):
            return -np.inf, cost
        cost += certificate_seconds(n)
        exact = xr.certify(I, O, F, K, m)
        if exact is None or exact <= 0:
            return -np.inf, cost
        return round_down(four_alpha / exact), cost

    budget = ledger.left() + BRACKET_SHARE * work_limit
    open_lower, spent, _ = least_open_bound(undecided, own_bound, budget)
    ledger.charge(spent)
    flush()

    solved = 0
    if pending:
        candidates = np.concatenate([p for p, _ in pending])
        bounds = np.concatenate([q for _, q in pending])
        best[0], best[1], solved, seconds, unresolved = leaves.resolve(
            candidates, bounds, best[0], best[1], work_limit - ledger.spent)
        ledger.charge(seconds)
        if unresolved < np.inf:
            # the limit left sets open, each with its bound
            closed = False
            open_lower = min(open_lower, unresolved)
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
            priced=counts["priced"], solved=solved, work=ledger.spent,
            trials=counts["trials"])
    if best[1] is None:
        if not closed:
            raise MlsynthEstimationError(
                f"SYNDES two-way exact backend: the branch and bound reached its "
                f"work limit ({work_limit:g} s) before any admissible design was "
                "found. Raise work_limit.")
        raise _no_feasible_design()

    value, a, b, gap = leaves.settle(best[1])
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
        priced=counts["priced"], solved=solved, work=ledger.spent,
        trials=counts["trials"])
