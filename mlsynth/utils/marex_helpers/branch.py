"""MAREX's exact backend above the candidate limit: a certified branch and bound.

The exact backend prices every admissible treated set of a cluster
(:func:`~mlsynth.utils.marex_helpers.exact.search_cluster`). Above
``CANDIDATE_LIMIT`` sets that stops being affordable, and the standard,
penalized and unit-penalized designs are searched here instead. The search
returns the same optimum, and a certificate of it.

Nodes. A node fixes some markets into the treated set (``I``), some out of it
(``O``), and leaves the rest free (``F``); its subtree is every admissible
treated set of size ``m`` that contains ``I`` and avoids ``O``. A node is either
priced set by set, exactly as the enumeration prices a cluster, or bounded and
split on one free market into a child that includes it and one that excludes
it. It is bounded only when pricing its subtree would cost more than the bound
(:data:`PRICE_SECONDS` per set against :func:`lift_seconds`); the costs decide
where the search spends time, never what it proves.

The bound (:func:`node_bound`). Below any node,

    f(S) = min ||B w||^2 + a'w  +  min ||B v||^2 + b'v,

with ``B`` the centred markets, ``w`` on the simplex over ``S`` and ``v`` over
the controls. A bound over the subtree must see the products ``w w'``: the
point that spreads the weights evenly over every market is an average of
designs and scores exactly zero, since the centred markets sum to zero. The
persp+N relaxation keeps one semidefinite block per side,
``M = [[1, x'], [x, X]] >= 0`` with ``x`` the side's weights on the markets it
can use, and adds

- ``X >= 0`` entrywise and ``X 1 = x``, ``1'x = 1``: the products of
  ``x >= 0`` and of the simplex;
- for every free market, the perspective ``X_jj z_j >= x_j^2`` as the 2x2 block
  ``[[z_j, x_j], [x_j, X_jj]] >= 0`` (``1 - z_j`` on the control side) and
  ``x_j <= z_j``, with ``0 <= z <= 1`` and ``sum_F z = m - |I|``.

Every constraint holds at ``X = x x'`` for a design, so the relaxation's
minimum is a lower bound on the subtree. The semidefinite block with the
perspective blocks is the form of Dong, Chen and Linderoth, eq. (24) of Wei,
Gómez and Küçükyavuz (2022), and, with ``X 1 = x``, the primal of the
best-diagonal program of Zheng, Sun and Li (2014); the perspective is that of
Frangioni and Gentile (2006) and Günlük and Linderoth (2010). ``X >= 0`` is the
entrywise half of the doubly nonnegative relaxation of Burer's (2009)
completely positive form, which those forms lack and which carries most of the
bound. The indicators never enter a semidefinite block, so each block has one
row more than the markets its side can use. The study behind the choice is
``benchmarks/studies/marex_bnb``.

The certificate of a bound. The solver's multipliers are read, clipped to their
signs (each 2x2 multiplier made semidefinite by :func:`_psd2`), and the
Lagrangian they define is assembled in exact arithmetic from the exact Gram
matrix of the data. It is linear in the lifted variables, ``const + <S, M> +
c'z``. A design has ``tr M = 1 + |x|^2 <= 2`` per side and ``z`` in
``{0 <= z <= 1, sum z = r}``, so

    f(S) >= const + 2 min(0, lambda_min(S)) per side + (the r smallest c_j),

with ``lambda_min`` bounded below exactly by :func:`lambda_min_lower`. This is
the rigorous bound of Jansson, Chaykin and Keil (2007) for semidefinite
programs, with exact arithmetic in place of their interval arithmetic: any
multipliers give a valid bound, and accurate ones a tight one. A node whose
solve fails keeps the bound it had: its parent's, or the monotone bound
``g(I | F) + h(O | F)`` (a fit over more markets is no worse), certified the
same way as a fitted set.

Pricing and the certificate of the search. A subtree priced set by set uses the
certificate's own machinery (:class:`~mlsynth.utils.marex_helpers.certify.ExactData`):
each set's treated fit is bounded below in floating point with its rounding
bounded, plus the control fit over every market the subtree's controls can use.
A set whose bound is under the incumbent is fitted, and its objective
bracketed exactly. Everything the search discards is discarded on a certified
lower bound above a certified upper bound, so the search is exact and its
certificate comes with it: ``lower <= f* <= f(design) <= upper``, with the sets
whose exact lower bound does not clear ``upper`` listed as unresolved, as
:func:`~mlsynth.utils.marex_helpers.certify.certify_cluster` lists them.

The work limit. The search charges its work in the cost model's seconds: a
set priced at ``price_seconds``, a bound at ``lift_seconds(k)``. It does not
start a pricing or a bound that would take its work past ``work_limit``. A
search stopped there returns the best design it found, and its certificate's
lower bound is the least of the bounds of the nodes it left open, so the
certificate still brackets the optimum; ``open`` counts those nodes, and a
search with none open has proved its design optimal. Only a search stopped
before it found any design raises.

The loop. The queue, the work's ledger and the handling of the limit are the
loop SYNDES's searches use too (:mod:`~mlsynth.utils.search_skeleton`, #846):
this module hands it the plan for a node, the pricing and the bound. Equal keys
go to the node pushed first, and the limit is read before each charge.

Restrictions. Forced and forbidden units start the root's ``I`` and ``O``;
stratum quotas and forced units decide how many sets a node holds
(:func:`~mlsynth.utils.marex_helpers.exact.count_candidates`, which drops a node
with none); a market included at a node sends its conflict partners out of the
treated set; and every set priced passes the cluster's filter. Donor exclusions
shrink a design's control pool, which can only raise its control fit, so every
bound stays valid without them, and every fitted set's controls honour them.

What this does not search. The weakly-targeted design couples the two fits
through the synthetic treated unit, and the bound above splits them, so above
the limit that design is still refused.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..search_skeleton import (Bound, Deferred, Ledger, NodeQueue, Price,
                               cheaper_to_price, check_work_limit, run)
from ..syndes_helpers.exact import SubsetFilter
from .certify import ClusterCertificate, Dyadic, ExactData, round_down, round_up
from .exact import (ClusterDesign, SearchNotClosed, _TopK, _chunks,
                    count_candidates, side_fit)
from .formulation import carries_weight

#: Seconds to price one treated set; with :func:`lift_seconds` it decides where
#: the search spends time, never what it proves.
PRICE_SECONDS = 1e-6

#: The work a search may do, in the cost model's seconds, before it stops and
#: returns its best design with the bracket it reached. Every search the docs
#: report closes well inside it: Walmart with eight treated does 145 seconds of
#: it, and the BLS panel with 80 markets and six treated
#: (``benchmarks/studies/bnb_scale``) at most about 450 (11 bounds and 169
#: million sets).
WORK_LIMIT = 600.0

#: The solver of the persp+N relaxation.
SOLVER = "CLARABEL"


def lift_seconds(k: int) -> float:
    """Seconds for one persp+N bound with ``k`` markets on its larger side.

    About a second to build the problem and a cubic term in the block size,
    fitted to Clarabel on GeoLift's 40 markets.
    """
    return 1.0 + 3.0 * (k / 40.0) ** 3


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
# the node bound
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NodeBound:
    """A node's certified persp+N bound.

    ``lower`` is an exact lower bound on the objective of every design in the
    node's subtree, ``relaxation`` the solver's value of the relaxation and
    ``z`` its indicators on the free markets. All three are ``None`` when the
    solve or its certificate failed.
    """

    lower: Optional[Fraction]
    relaxation: Optional[float]
    z: Optional[Dict[int, float]]


_UNBOUNDED = NodeBound(lower=None, relaxation=None, z=None)


class _Side:
    """One side's semidefinite block and the constraints whose multipliers
    certify it."""

    def __init__(self, cp, U: List[int], z, pos: Dict[int, int], control: bool):
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
            self.cper.append(cp.bmat([
                [cp.reshape(ind, (1, 1)), cp.reshape(x[i], (1, 1))],
                [cp.reshape(x[i], (1, 1)), cp.reshape(X[i, i], (1, 1))]]) >> 0)
        self.constraints = ([self.M >> 0, self.c00, self.cnn, self.csum,
                             self.crow] + self.clin + self.cper)


def _duals_finite(sides: List[_Side]) -> bool:
    for s in sides:
        for con in [s.c00, s.cnn, s.csum, s.crow] + s.clin + s.cper:
            d = con.dual_value
            if d is None or not np.all(np.isfinite(np.asarray(d, dtype=float))):
                return False
    return True


@dataclass(frozen=True)
class _SideDuals:
    """One side's multipliers, as a solver returned them, and what they price.

    ``U`` are the markets the side can use, ``free`` the pairs ``(i, f)`` of a
    position in ``U`` and the market's position among the free markets,
    ``control`` whether the side's indicator is ``1 - z``, and ``lin`` its
    penalty vector over the cluster. ``y0``, ``alpha``, ``beta``, ``H``, ``mu``
    and ``Lam`` multiply ``M_00 = 1``, ``1'x = 1``, ``X 1 = x``, ``X >= 0``,
    ``ind_j - x_j >= 0`` and the 2x2 perspective blocks; any values will do.
    """

    U: List[int]
    free: List[Tuple[int, int]]
    control: bool
    lin: Optional[np.ndarray]
    y0: float
    alpha: float
    beta: np.ndarray
    H: np.ndarray
    mu: List[float]
    Lam: List[np.ndarray]


def _certificate(G: Dyadic, scale: Fraction, r: int, n_free: int,
                 sides: Sequence[_SideDuals]) -> Optional[Fraction]:
    """A lower bound on every design of a node, from any multipliers.

    Each multiplier is first made to have its constraint's sign (``H``, ``mu``
    clipped at zero, each 2x2 block made semidefinite), so every subtracted term
    is zero or positive at a design; the Lagrangian is then linear in the lifted
    variables, ``const + <S, M> + c'z`` per side in units of ``scale``, and is
    bounded below over the designs exactly. ``None`` when a smallest eigenvalue
    could not be bounded.
    """
    const = Fraction(0)
    cz = [Fraction(0)] * n_free
    for s in sides:
        U = s.U
        k = len(U)
        H = np.clip((s.H + s.H.T) / 2.0, 0.0, None)
        H = np.triu(H) + np.triu(H, 1).T
        const += Fraction(s.y0) + Fraction(s.alpha)
        S = [[Fraction(0)] * (k + 1) for _ in range(k + 1)]
        S[0][0] = -Fraction(s.y0)
        for a in range(k):
            for b in range(k):
                S[a + 1][b + 1] = (G.frac(U[a], U[b]) / scale - Fraction(H[a, b])
                                   - (Fraction(s.beta[a]) + Fraction(s.beta[b])) / 2)
        xc = [-Fraction(s.alpha) + Fraction(s.beta[a]) for a in range(k)]
        if s.lin is not None:
            xc = [xc[a] + Fraction(float(s.lin[U[a]])) / scale for a in range(k)]
        for (i, f), mu_raw, L in zip(s.free, s.mu, s.Lam):
            mu = Fraction(max(float(mu_raw), 0.0))
            a_, b_, c_ = (Fraction(v) for v in _psd2(np.asarray(L, float)))
            xc[i] += mu - 2 * b_
            S[i + 1][i + 1] -= c_
            if s.control:                       # the indicator is 1 - z
                const -= mu + a_
                cz[f] += mu + a_
            else:                               # the indicator is z
                cz[f] -= mu + a_
        for a in range(k):
            S[0][a + 1] = S[a + 1][0] = xc[a] / 2
        lam = lambda_min_lower(_from_fractions(S))
        if lam is None:
            return None
        const += 2 * min(Fraction(0), lam)
    if n_free:
        const += sum(sorted(cz)[:r])
    return const * scale


def node_bound(
    data: ExactData,
    m: int,
    I: Sequence[int],
    O: Sequence[int],
    F: Sequence[int],
    linear_treated: Optional[np.ndarray] = None,
    linear_control: Optional[np.ndarray] = None,
) -> NodeBound:
    """The certified persp+N bound of the node ``(I, O, F)`` with ``m`` treated.

    ``I``, ``O`` and ``F`` partition the cluster's markets; the penalty vectors
    are indexed by them.
    """
    import cvxpy as cp

    I, O, F = list(I), list(O), list(F)
    G = data.gram
    gmax = max(G.frac(i, i) for i in range(data.n))
    if not 0 < gmax < Fraction(np.finfo(float).max):
        return _UNBOUNDED
    scale = Fraction(2) ** round(math.log2(gmax))
    Gs = data.Gf / float(scale)
    r = m - len(I)
    pos = {j: i for i, j in enumerate(F)}
    z = cp.Variable(len(F)) if F else None
    cons = [] if z is None else [z >= 0, z <= 1, cp.sum(z) == r]
    sides = [_Side(cp, sorted(I + F), z, pos, control=False),
             _Side(cp, sorted(O + F), z, pos, control=True)]
    lins = [None if linear_treated is None else np.asarray(linear_treated, float),
            None if linear_control is None else np.asarray(linear_control, float)]
    obj = 0
    for s, lin in zip(sides, lins):
        X, x = s.M[1:, 1:], s.M[0, 1:]
        obj = obj + cp.trace(Gs[np.ix_(s.U, s.U)] @ X)
        if lin is not None:
            obj = obj + (lin[s.U] / float(scale)) @ x
        cons += s.constraints
    problem = cp.Problem(cp.Minimize(obj), cons)
    try:
        problem.solve(solver=SOLVER)
    except (cp.error.SolverError, ArithmeticError, ValueError,
            np.linalg.LinAlgError):
        return _UNBOUNDED
    if problem.value is None or not np.isfinite(problem.value) \
            or not _duals_finite(sides):
        return _UNBOUNDED
    duals = [_SideDuals(
        U=s.U, free=s.free, control=s.control, lin=lin,
        y0=-float(s.c00.dual_value), alpha=-float(s.csum.dual_value),
        beta=-np.asarray(s.crow.dual_value, dtype=float).reshape(s.k),
        H=np.asarray(s.cnn.dual_value, dtype=float).reshape(s.k, s.k),
        mu=[float(c.dual_value) for c in s.clin],
        Lam=[np.asarray(c.dual_value, dtype=float) for c in s.cper])
        for s, lin in zip(sides, lins)]
    lower = _certificate(G, scale, r, len(F), duals)
    if lower is None:
        return _UNBOUNDED
    zv = np.asarray(z.value, dtype=float) if z is not None else np.zeros(0)
    return NodeBound(lower=lower,
                     relaxation=float(problem.value) * float(scale),
                     z={j: float(zv[pos[j]]) for j in F})


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BranchResult:
    """What the branch and bound returns for one cluster.

    ``designs`` are the ``top_k`` best, best first; ``certificate`` brackets the
    best one; ``lifts`` counts the persp+N bounds computed and ``priced`` the
    sets priced one at a time. ``work`` is the work done in the cost model's
    seconds, the quantity ``work_limit`` caps, and ``open`` counts the nodes
    the limit left open whose bound does not clear the design: none when the
    search closed.
    """

    designs: List[ClusterDesign]
    certificate: ClusterCertificate
    lifts: int
    priced: int
    work: float
    open: int

    @property
    def closed(self) -> bool:
        """Whether the search proved its design optimal."""
        return self.open == 0


def branch_and_bound(
    B: np.ndarray,
    A: np.ndarray,
    sizes: Sequence[int],
    *,
    filt: SubsetFilter,
    linear_treated: Optional[np.ndarray] = None,
    linear_control: Optional[np.ndarray] = None,
    donor_exclusion: Sequence[Tuple[int, int]] = (),
    top_k: int = 1,
    price_seconds: float = PRICE_SECONDS,
    lift_seconds: Callable[[int], float] = lift_seconds,
    work_limit: float = WORK_LIMIT,
) -> BranchResult:
    """The ``top_k`` best designs of one cluster by a certified branch and bound.

    Takes :func:`~mlsynth.utils.marex_helpers.exact.search_cluster`'s arguments
    for the separable designs, less its candidate limit, and the costs that
    decide where it spends time.

    Raises
    ------
    MlsynthConfigError
        For a ``work_limit`` that is not a positive finite number.
    MlsynthDataError
        For a panel with a NaN or an infinite value.
    MlsynthEstimationError
        When no admissible design exists.
    SearchNotClosed
        When the work limit is reached before any design is found.
    """
    check_work_limit(work_limit)
    data = ExactData(B, A)
    B, A, n = data.B, data.A, data.n
    lin_t = None if linear_treated is None else np.asarray(linear_treated, float)
    lin_c = None if linear_control is None else np.asarray(linear_control, float)
    excluded: Dict[int, set] = {}
    for i, j in donor_exclusion:
        excluded.setdefault(int(i), set()).add(int(j))
    partners: Dict[int, set] = {}
    for i, j in filt.conflict_pairs:
        partners.setdefault(int(i), set()).add(int(j))
        partners.setdefault(int(j), set()).add(int(i))

    best = _TopK(top_k)
    brackets: Dict[Tuple[int, ...], Tuple[Fraction, Fraction]] = {}
    floors: Dict[Tuple[int, ...], Fraction] = {}
    refused: set = set()
    state = {"lifts": 0, "priced": 0, "admissible": False}
    ledger = Ledger(work_limit)

    def pool(S: Sequence[int]) -> List[int]:
        drop = set(S)
        for j in S:
            drop |= excluded.get(j, set())
        return [j for j in range(n) if j not in drop]

    def fit_lower(cols: Sequence[int], linear) -> Tuple[Fraction, np.ndarray]:
        cols = list(cols)
        _, w = side_fit(B, A, cols, None if linear is None else linear[cols])
        return data.interval(cols, w, linear)[0], w

    def control_floor(cols: Tuple[int, ...]) -> Fraction:
        """A certified lower bound on every control fit inside ``cols``."""
        if cols not in floors:
            floors[cols] = fit_lower(cols, lin_c)[0]
        return floors[cols]

    def threshold() -> Optional[Fraction]:
        """A certified upper bound on every design kept, once ``top_k`` are.

        A set whose certified lower bound is above it is worse than ``top_k``
        designs already found, so it is not among the ``top_k`` best; and it is
        strictly worse than the best design the search will return, whose
        objective can only be lower than the best one's now.
        """
        if len(best.items) < best.k:
            return None
        return max(brackets[d.treated][1] for d in best.items)

    def clears(bound: Fraction) -> bool:
        cut = threshold()
        return cut is not None and bound > cut

    def fit(S: Tuple[int, ...], h_floor: Fraction) -> None:
        if S in brackets or S in refused:
            return
        cols = list(S)
        g, w = side_fit(B, A, cols, None if lin_t is None else lin_t[cols])
        if not carries_weight(w):
            # A set treats the markets it weights: one its fit gives no weight
            # treats fewer than it holds and is not a design of its size.
            refused.add(S)
            return
        lo_g, hi_g = data.interval(cols, w, lin_t)
        if clears(lo_g + h_floor):
            return
        control = pool(S)
        if not control:
            return
        h, v = side_fit(B, A, control, None if lin_c is None else lin_c[control])
        lo_h, hi_h = data.interval(control, v, lin_c)
        brackets[S] = (lo_g + lo_h, hi_g + hi_h)
        best.offer(ClusterDesign(value=g + h, treated=S, control=tuple(control),
                                 w=w, v=v))

    def price(m: int, sub: SubsetFilter, O: Tuple[int, ...], F: Tuple[int, ...]
              ) -> None:
        """Every set of a subtree, priced and fitted under the incumbent."""
        h_floor = control_floor(tuple(sorted(O + F)))
        h_down = round_down(h_floor)
        for block in _chunks(n, [m], sub):
            state["priced"] += len(block)
            priced = np.nextafter(data.treated_bounds(block, lin_t) + h_down, -np.inf)
            for i in np.argsort(priced, kind="stable"):
                cut = threshold()
                if cut is not None and priced[i] > round_up(cut):
                    break
                fit(tuple(int(j) for j in block[i]), h_floor)

    def conflicted(I: Sequence[int]) -> bool:
        chosen = set(I)
        return any(partners.get(j, set()) & chosen for j in chosen)

    def bound_node(bound, m, I, O, F, sub, k):
        h_floor = control_floor(tuple(sorted(O + F)))
        g_floor, w = fit_lower(sorted(I + F), lin_t)
        bound = max(bound, g_floor + h_floor)
        if clears(bound):
            return None
        charge = lift_seconds(k)
        if ledger.passes(charge):
            return Deferred(bound)
        ledger.charge(charge)
        nb = node_bound(data, m, I, O, F, lin_t, lin_c)
        state["lifts"] += 1
        if nb.lower is not None:
            bound = max(bound, nb.lower)
        U = sorted(I + F)
        r = m - len(I)
        score = nb.z if nb.z is not None else {j: float(w[U.index(j)]) for j in F}
        guess = tuple(sorted(I + tuple(sorted(F, key=lambda k: (-score[k], k))[:r])))
        if sub.accepts(guess):
            fit(guess, h_floor)
        if clears(bound):
            return None
        j = max(F, key=lambda k: (score[k], -k))
        out = partners.get(j, set())
        return [(bound, (m, tuple(sorted(I + (j,))),
                         tuple(sorted(O + tuple(k for k in F if k in out))),
                         tuple(k for k in F if k != j and k not in out))),
                (bound, (m, I, tuple(sorted(O + (j,))), tuple(k for k in F if k != j)))]

    def over_budget(I, F, r) -> bool:
        """Whether even the ``r`` cheapest free markets put the node's forced
        markets over the total-cost budget, so no set below it is admissible."""
        if filt.costs is None:
            return False
        least = float(filt.costs[list(I)].sum()) + float(np.sort(filt.costs[list(F)])[:r].sum())
        return least > filt.budget + 1e-9

    def plan(bound, node):
        """Drop a node its key clears or that holds no admissible design;
        price one cheaper to price than to bound; bound any other."""
        m, I, O, F = node
        if clears(bound):
            return None
        r = m - len(I)
        if r < 0 or r > len(F) or conflicted(I):
            return None
        if over_budget(I, F, r):
            return None
        sub = dataclasses.replace(filt, forced_in=frozenset(I), forbidden=frozenset(O))
        count = count_candidates(n, [m], sub)
        if not count:
            return None
        state["admissible"] = True
        if len(O) + len(F) - r < 1:              # every set treats every market
            return None
        k = max(len(I) + len(F), len(O) + len(F))
        if count < 2 or cheaper_to_price(count, price_seconds, lift_seconds(k)):
            return Price(cost=count * price_seconds, deferrable=True,
                         run=lambda: price(m, sub, O, F))
        return Bound(run=lambda ledger: bound_node(bound, m, I, O, F, sub, k))

    # equal keys go to the node pushed first: each size's root in order, and a
    # split's child that includes its market before the one that excludes it
    queue = NodeQueue(last_first=False)
    forced = tuple(sorted(filt.forced_in))
    forbidden = tuple(sorted(filt.forbidden))
    if not set(forced) & set(forbidden):
        rest = tuple(j for j in range(n) if j not in filt.forced_in
                     and j not in filt.forbidden)
        for m in sizes:
            queue.push(Fraction(0), (int(m), forced, forbidden, rest))
    run(queue, ledger, plan=plan, check_before_pop=False)

    left = [entry[0] for entry in queue.open()]   # the bounds of the nodes left open
    if left and not best.items:
        raise SearchNotClosed(
            f"The branch and bound reached its work limit ({work_limit:g} modelled "
            f"seconds) before it found an admissible design. Narrow the design "
            f"with restrictions, clusters or a smaller treated count, or raise "
            f"work_limit.")
    if not best.items:
        if not state["admissible"]:
            raise MlsynthEstimationError(
                "No admissible treated set exists in this cluster under the "
                "cardinality and the active restrictions.")
        raise MlsynthEstimationError(
            "No admissible design exists in this cluster: every treated set "
            "the restrictions allow leaves no control market or has a market "
            "its fit gives no weight. A treated market must carry weight; "
            "Abadie and Zhao count the nonzero treated weights.")
    top = best.items[0]
    lower, upper = brackets[top.treated]
    unresolved = []
    for S, (lo, _) in brackets.items():
        if S != top.treated and lo <= upper:
            unresolved.append(S)
            lower = min(lower, lo)
    still_open = [b for b in left if not clears(b)]
    lower = min([lower] + still_open)
    certificate = ClusterCertificate(lower=round_down(lower), upper=round_up(upper),
                                     unresolved=tuple(sorted(unresolved)),
                                     fitted=len(brackets))
    return BranchResult(designs=list(best.items), certificate=certificate,
                        lifts=state["lifts"], priced=state["priced"],
                        work=ledger.spent, open=len(still_open))
