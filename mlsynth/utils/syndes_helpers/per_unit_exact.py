"""The exact per-unit SYNDES backend: a branch and bound on units.

Doudchenko et al. (2021, Appendix A.3) observe that the per-unit problem is ``K``
one-way problems, one per treated unit, each with every control unit as a donor.
In ``mlsynth``'s formulation a treated unit's weight on another treated unit is
forced to zero and the penalty falls on the weights, so for a treated set ``S``
of size ``K``

    f(S) = (1/K) sum_{i in S} phi_i(S^c),
    phi_i(P) = min over v on the simplex over P of (1/T)||y_i - Y_P v||^2 + lam ||v||^2,

where ``P`` also lacks the donors a restriction excludes for ``i``. Naming the
treated set leaves ``K`` simplex fits and no binary at all.

``phi_i`` can only rise as its pool shrinks. A node of the search fixes a set
``F`` of treated units and a set ``E`` of controls, and every completion ``S``
contains ``F``, so every pool in it lacks ``F``:

    f(S) >= (1/K) [ sum_{i in F} phi_i(N \\ F)
                    + the K - |F| smallest phi_u(N \\ (F + u)), u undecided ].

The bound tightens as ``F`` grows and equals ``f(S)`` at a leaf, which is what
the separable bound over root fits alone cannot do. On 156 study runs
(``benchmarks/studies/syndes_per_unit_exact/``) this search closed every one,
in at most 28 s, where the order of the separable bound left five open after
120 s.

The fits run on the panel centred within each period and divided by its root
mean square, with the penalty divided by the square: a unit's residual is a
simplex combination, so a level common to the units cancels from it, and the
change of units is exact. Values are reported in the outcome's units. Without
it the active-set solver's absolute tolerances decide the fits on a panel
measured in small units: at a scale of ``1e-5`` the design was 1% worse than the
optimum.

Two reuses keep the fits cheap, and both are exact. A unit's root fit stays
optimal on a smaller pool whenever the removed donors carried no weight in it.
A fit with only the root's support removed stays optimal for any larger removal
that misses its own support. Every fit is ``solve_simplex_qp``, its convergence
read and its Frank-Wolfe gap recorded; each is warm-started from the unit's root
weights.

When the search closes, every pruned node's bound was at or above the incumbent,
so the incumbent is the optimum. The search counts its work in modelled seconds,
as the two-way search does: each fit at :func:`fit_seconds` of its matrix and each
node at :data:`NODE_SECONDS_PER_UNIT` per unit, the cost of reading and sorting
the undecided units' fits. It reads ``work_limit`` between nodes, so the work
passes the limit by at most the last node's. A search stopped there returns its
incumbent with ``certified=False``, the smallest bound left open as the lower
bound, and a :class:`~.optimization.SYNDESSolveWarning`.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from time import perf_counter
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from ..fast_scm_helpers.structure import IndexSet
from ..solvers.active_set import solve_simplex_qp
from .exact import SubsetFilter, _NoFeasibleDesign
from .optimization import SYNDESSolveWarning, estimate_lambda
from .structures import SYNDESDesign, SYNDESSolveOutcome

#: Modelled seconds for one simplex fit: a fixed cost and a cost per entry of
#: its matrix (rows, the periods and one per donor for the penalty, by the
#: donors), fitted to the library's 540,262 fits on the per-unit study's panels.
FIT_SECONDS = 8.1e-5
FIT_SECONDS_PER_ENTRY = 2.44e-7

#: Modelled seconds a node spends beyond its fits, per unit of the panel.
NODE_SECONDS_PER_UNIT = 5.1e-5

#: The modelled seconds a search may spend before it stops and reports an
#: uncertified incumbent. Every run of the per-unit study closes inside it: the
#: hardest, half of twenty units treated on a panel with no structure, does 74
#: modelled seconds (31,083 nodes and 203,903 fits), 72 to 75 on the clock.
WORK_LIMIT = 120.0


def fit_seconds(rows: int, cols: int) -> float:
    """Modelled seconds for one fit with a ``rows`` by ``cols`` matrix."""
    return FIT_SECONDS + FIT_SECONDS_PER_ENTRY * rows * cols


class UnitFits:
    """``phi_i`` on any pool, with exact reuse of solved fits.

    The fits run on ``Y`` centred within each period and divided by its root
    mean square ``scale``, with ``lam / scale**2``; values come back multiplied
    by ``scale**2``, in the outcome's units.

    Parameters
    ----------
    Y : np.ndarray
        ``(T, N)`` pre-period panel.
    lam : float
        Penalty on the weights, in the outcome's units.
    excluded : dict, optional
        Unit ``i`` to the donors it may never use.
    """

    def __init__(self, Y: np.ndarray, lam: float,
                 excluded: Optional[Dict[int, FrozenSet[int]]] = None):
        Y = np.asarray(Y, dtype=float)
        centred = Y - Y.mean(axis=1, keepdims=True)
        scale = float(np.sqrt(np.mean(centred ** 2)))
        self.scale = scale if scale > 0.0 else 1.0
        self.Y = centred / self.scale
        self.T, self.N = self.Y.shape
        self.lam = float(lam) / self.scale ** 2
        self.excluded = {int(i): frozenset(int(j) for j in js)
                         for i, js in (excluded or {}).items()}
        self.n_fits = 0
        self.work = 0.0
        self.max_rel_gap = 0.0
        self._cache: Dict[Tuple[int, FrozenSet[int]],
                          Tuple[float, Optional[np.ndarray]]] = {}
        self.root_value = np.empty(self.N)
        self.root_weights: List[Optional[np.ndarray]] = []
        self.root_support: List[FrozenSet[int]] = []
        for i in range(self.N):
            value, w = self._solve(i, frozenset())
            self.root_value[i] = value
            self.root_weights.append(w)
            self.root_support.append(_support(w))

    def _solve(self, i: int, removed: FrozenSet[int]
               ) -> Tuple[float, Optional[np.ndarray]]:
        """``phi_i`` and its weights (dense, length ``N``), from a fresh fit."""
        key = (i, removed)
        if key in self._cache:
            return self._cache[key]
        banned = removed | self.excluded.get(i, frozenset()) | {i}
        pool = [j for j in range(self.N) if j not in banned]
        if not pool:
            self._cache[key] = (math.inf, None)
            return self._cache[key]

        root_t = math.sqrt(self.T)
        B = self.Y[:, pool] / root_t
        A = self.Y[:, i] / root_t
        if self.lam > 0.0:
            B = np.vstack([B, math.sqrt(self.lam) * np.eye(len(pool))])
            A = np.concatenate([A, np.zeros(len(pool))])
        warm = None
        if i < len(self.root_weights) and self.root_weights[i] is not None:
            start = self.root_weights[i][pool]
            if start.sum() > 0.0:
                warm = start / start.sum()
        v, info = solve_simplex_qp(B, A, warm_start=warm, return_info=True)
        self.work += fit_seconds(*B.shape)
        if not info["converged"]:
            raise MlsynthEstimationError(
                f"SYNDES per-unit exact backend: the fit of unit {i} on a pool of "
                f"{len(pool)} donors did not converge, so its value bounds nothing.")
        r = A - B @ v
        value = float(r @ r)
        grad = -2.0 * (B.T @ r)
        if value > 0.0:
            self.max_rel_gap = max(self.max_rel_gap,
                                   abs(float(grad @ v - grad.min())) / value)
        self.n_fits += 1
        w = np.zeros(self.N)
        w[pool] = v
        self._cache[key] = (value * self.scale ** 2, w)
        return self._cache[key]

    def fit(self, i: int, removed: FrozenSet[int]
            ) -> Tuple[float, Optional[np.ndarray]]:
        """``phi_i`` on the pool lacking ``i``, ``removed`` and ``i``'s exclusions."""
        removed = frozenset(removed) - {i}
        hit = removed & self.root_support[i]
        if not hit:
            return float(self.root_value[i]), self.root_weights[i]
        value, w = self._solve(i, hit)
        if w is None or not (_support(w) & removed):
            return value, w
        return self._solve(i, removed)

    def phi(self, i: int, removed: FrozenSet[int]) -> float:
        return self.fit(i, removed)[0]


def _support(w: Optional[np.ndarray]) -> FrozenSet[int]:
    if w is None:
        return frozenset()
    return frozenset(int(j) for j in np.flatnonzero(w > 0.0))


@dataclasses.dataclass
class _Outcome:
    treated: Tuple[int, ...]
    value: float
    weights: np.ndarray
    lower_bound: float
    closed: bool
    nodes: int
    fits: int
    max_rel_gap: float
    seconds: float
    work: float


def _search(fits: UnitFits, K: int, filt: SubsetFilter, work_limit: float
            ) -> _Outcome:
    """Depth-first branch and bound over which units are treated."""
    t0 = perf_counter()
    N = fits.N
    cost = filt.costs
    node_seconds = NODE_SECONDS_PER_UNIT * N

    def can_add(F: FrozenSet[int], u: int) -> bool:
        if any((u == a and b in F) or (u == b and a in F)
               for a, b in filt.conflict_pairs):
            return False
        for members, _, upper in filt.strata:
            if upper is not None and u in members and \
                    len(F.intersection(members)) + 1 > upper:
                return False
        if cost is not None and float(cost[list(F | {u})].sum()) > filt.budget + 1e-9:
            return False
        return True

    best, best_set = math.inf, None
    never = frozenset(i for i in range(N) if math.isinf(fits.root_value[i]))
    start_F, start_E = frozenset(filt.forced_in), frozenset(filt.forbidden) | never
    # Seed the incumbent with the forced units and the best root fits, when that
    # set is admissible: it prunes from the first node, and a search stopped at
    # its work limit then has a design to return.
    free = [u for u in np.argsort(fits.root_value, kind="stable")
            if u not in start_F and u not in start_E]
    if K - len(start_F) >= 0 and len(free) >= K - len(start_F):
        seed = tuple(sorted(start_F | {int(u) for u in free[:K - len(start_F)]}))
        if filt.accepts(seed):
            value = _price(fits, seed)
            if not math.isinf(value):
                best, best_set = value, seed
    # Each open node carries its parent's bound, a valid bound on everything
    # below it; the smallest of them is the search's lower bound if it stops.
    stack: List[Tuple[FrozenSet[int], FrozenSet[int], float]] = []
    if not (start_F & start_E):
        stack.append((start_F, start_E, -math.inf))
    nodes = 0
    closed = True
    while stack:
        if fits.work + nodes * node_seconds >= work_limit:
            closed = False
            break
        F, E, _ = stack.pop()
        nodes += 1
        need = K - len(F)
        U = [u for u in range(N) if u not in F and u not in E]
        if need < 0 or need > len(U):
            continue
        fixed = math.fsum(fits.phi(i, F) for i in F)
        terms = sorted((fits.phi(u, F), u) for u in U)
        cheapest = [v for v, _ in terms[:need]]
        if math.isinf(fixed) or any(math.isinf(v) for v in cheapest):
            continue
        bound = (fixed + math.fsum(cheapest)) / K
        if bound >= best:
            continue
        if need == 0 or need == len(U):
            S = tuple(sorted(F | set(U))) if need else tuple(sorted(F))
            if filt.accepts(S):
                value = bound if need == 0 else _price(fits, S)
                if value < best:
                    best, best_set = value, S
            continue
        u = terms[0][1]
        stack.append((F, E | {u}, bound))
        if can_add(F, u):
            stack.append((F | {u}, E, bound))

    if best_set is None:
        if not closed:
            raise MlsynthEstimationError(
                "SYNDES per-unit exact backend: the work limit was reached before "
                "any admissible design was found. Raise work_limit.")
        raise _NoFeasibleDesign(
            "SYNDES found no feasible design: the active restrictions "
            "(forced/forbidden units, donor exclusions, conflicts, stratum "
            "quotas, cost budget) are jointly unsatisfiable for the requested K. "
            "Relax the restrictions or reduce K.")
    lower = best if closed else min([best] + [b for _, _, b in stack])
    Q = np.zeros((N, N))
    removed = frozenset(best_set)
    for i in best_set:
        Q[i] = fits.fit(i, removed)[1]
    return _Outcome(treated=best_set, value=best, weights=Q, lower_bound=lower,
                    closed=closed, nodes=nodes, fits=fits.n_fits,
                    max_rel_gap=fits.max_rel_gap, seconds=perf_counter() - t0,
                    work=fits.work + nodes * node_seconds)


def _price(fits: UnitFits, S: Sequence[int]) -> float:
    removed = frozenset(S)
    return math.fsum(fits.phi(i, removed) for i in S) / len(S)


def solve_per_unit_exact(
    Y: np.ndarray,
    K: int,
    *,
    lam: Optional[float] = None,
    unit_index: Optional[IndexSet] = None,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    restrictions: Optional[Any] = None,
    forbidden_sets: Optional[Sequence[np.ndarray]] = None,
    work_limit: float = WORK_LIMIT,
) -> SYNDESDesign:
    """Solve the per-unit design by a branch and bound on units.

    Returns the same :class:`SYNDESDesign` as
    :func:`~.optimization.solve_synthetic_design` with ``mode="per_unit"``, so
    the two backends are interchangeable downstream, and fills ``solve_outcome``
    with the search's own bound.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix of shape ``(T, N)``.
    K : int
        Number of treated units, in ``1 .. N - 1``.
    lam : float, optional
        Penalty on the weights; ``estimate_lambda(Y)`` when ``None``, matching
        the MIP path.
    unit_index : IndexSet, optional
        Label mapping, used to populate the design's label fields.
    costs, budget : optional
        Per-unit treatment cost and a cap on their total over the treated set.
    restrictions : DesignRestrictions, optional
        Assignment restrictions. Donor exclusions shrink the treated units'
        pools; the rest restrict which units are treated.
    forbidden_sets : sequence of arrays, optional
        Treated sets to exclude, which is how the pool asks for the next design.
    work_limit : float, optional
        Modelled seconds the search may spend before it stops uncertified.

    Raises
    ------
    MlsynthDataError
        If ``Y`` is not a two-dimensional array.
    MlsynthConfigError
        For an out-of-range ``K``, a negative ``lam``, malformed costs, or a
        ``work_limit`` that is not a positive finite number.
    MlsynthEstimationError
        When no admissible design exists, or a fit does not converge.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise MlsynthDataError("Y must be a two-dimensional T x N matrix.")
    T, N = Y.shape
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or not 1 <= K <= N - 1:
        raise MlsynthConfigError(
            f"K must be an integer in 1..{N - 1} for a per-unit design; got {K!r}.")
    K = int(K)
    if (isinstance(work_limit, bool) or not isinstance(work_limit, (int, float, np.number))
            or not np.isfinite(work_limit) or work_limit <= 0):
        raise MlsynthConfigError(
            f"work_limit must be a positive finite number of seconds; got {work_limit!r}.")
    lam_value = float(estimate_lambda(Y)) if lam is None else float(lam)
    if not lam_value >= 0.0:
        raise MlsynthConfigError(f"lam must be nonnegative; got {lam!r}.")

    excluded: Dict[int, set] = {}
    treated_rules = restrictions
    if restrictions is not None and getattr(restrictions, "donor_exclusion", None):
        for i, j in restrictions.donor_exclusion:
            excluded.setdefault(int(i), set()).add(int(j))
        treated_rules = dataclasses.replace(restrictions, donor_exclusion=[])
    filt = SubsetFilter.build(treated_rules, costs, budget, forbidden_sets, N=N)

    t0 = perf_counter()
    fits = UnitFits(Y, lam_value, {i: frozenset(js) for i, js in excluded.items()})
    outcome = _search(fits, K, filt, float(work_limit))
    seconds = perf_counter() - t0

    gap = ((outcome.value - outcome.lower_bound) / outcome.value
           if outcome.value > 0.0 else 0.0)
    status = "optimal" if outcome.closed else "work_limit"
    if not outcome.closed:
        warnings.warn(
            f"SYNDES per-unit exact backend stopped at its work limit "
            f"({work_limit:g} modelled seconds, {outcome.nodes} nodes) with a gap "
            f"of {gap:.3g} to its lower bound; the design is the best found, not a "
            "proven optimum. Raise work_limit.",
            SYNDESSolveWarning, stacklevel=2)
    solve_outcome = SYNDESSolveOutcome(
        status=status, solver_status="exact", stopped_at_limit=not outcome.closed,
        objective=outcome.value, solver_objective=outcome.value,
        dual_bound=outcome.lower_bound, gap=gap, solve_time=seconds,
        limits_applied=True)
    return _to_design(Y, outcome, lam_value, K, unit_index, solve_outcome)


def solve_per_unit_exact_pool(
    Y: np.ndarray,
    K: int,
    top_K: int = 1,
    **kwargs: Any,
) -> List[SYNDESDesign]:
    """The ``top_K`` best distinct per-unit designs, ranked by objective.

    Re-searches with each chosen treated set forbidden. Stops early, returning
    fewer than ``top_K``, once the admissible region is exhausted; any other
    failure raises.

    Raises
    ------
    MlsynthConfigError
        If ``top_K`` is not a positive integer.
    """
    if isinstance(top_K, bool) or not isinstance(top_K, (int, np.integer)) or top_K < 1:
        raise MlsynthConfigError(f"top_K must be a positive integer; got {top_K!r}.")
    excluded = list(kwargs.pop("forbidden_sets", None) or [])
    designs: List[SYNDESDesign] = []
    for _ in range(int(top_K)):
        try:
            design = solve_per_unit_exact(Y, K, forbidden_sets=excluded, **kwargs)
        except _NoFeasibleDesign:
            break
        designs.append(design)
        excluded.append(np.asarray(design.selected_unit_indices, dtype=np.int64))
    return designs


def _to_design(Y: np.ndarray, outcome: _Outcome, lam: float, K: int,
               unit_index: Optional[IndexSet],
               solve_outcome: SYNDESSolveOutcome) -> SYNDESDesign:
    """Package a search outcome in the shared design container.

    Mirrors ``optimization._extract_weights``' per-unit convention: row ``i`` of
    ``q`` is treated unit ``i``'s control weights, and the contrast is the mean
    of the unit contrasts, ``(D - q.sum(0)) / K``.
    """
    N = Y.shape[1]
    treated = np.asarray(outcome.treated, dtype=np.int64)
    assignment = np.zeros(N, dtype=int)
    assignment[treated] = 1
    if unit_index is not None:
        selected_labels = unit_index.get_labels(treated)
        all_labels = unit_index.labels
    else:
        selected_labels = treated
        all_labels = np.arange(N)
    Q = outcome.weights
    contrast_weights = (assignment - Q.sum(axis=0)) / K
    contrast_series = Y @ contrast_weights
    return SYNDESDesign(
        mode="per_unit",
        objective_value=outcome.value,
        lambda_value=lam,
        assignment=assignment,
        selected_unit_indices=treated,
        selected_unit_labels=np.asarray(selected_labels),
        assignment_by_unit={
            label: int(val) for label, val in zip(all_labels, assignment)},
        w=Q,
        q=Q,
        z=None,
        treated_weights=Q,
        control_weights=None,
        contrast_weights=contrast_weights,
        contrast_series=contrast_series,
        pre_fit_rmse=float(np.sqrt(np.mean(contrast_series ** 2))),
        raw_results={
            "status": solve_outcome.status,
            "solver": "exact",
            "certified": outcome.closed,
            "lower_bound": outcome.lower_bound,
            "n_nodes": outcome.nodes,
            "n_fits": outcome.fits,
            "work": outcome.work,
            "max_fit_gap": outcome.max_rel_gap,
        },
        solve_outcome=solve_outcome,
    )
