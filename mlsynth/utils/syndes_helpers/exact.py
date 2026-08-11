"""The exact two-way SYNDES backend: search treated sets, not branch-and-bound.

:mod:`mlsynth.utils.syndes_helpers.gram` shows that naming the treated set
removes every binary from the two-way global MIP, leaving a convex program in
``G = Y'Y / T``, and gives a closed form for a relaxation of it,

    lb(S) = 4 alpha / (sigma' R sigma),

which lower-bounds ``f(S)`` always and equals it whenever the dropped sign
conditions turn out to be slack. :mod:`mlsynth.utils.syndes_helpers.partition`
solves ``f(S)`` for a batch of treated sets when they do bind.

This module is the outer search those two make cheap. One matrix product scores
every candidate design; the rows the closed form settles are solved outright and
supply an incumbent; every design whose bound clears that incumbent is discarded
soundly; only the survivors reach the iterative solver. On the panels used to
develop it, that leaves nought or one design out of thousands.

What the search buys over branch-and-bound is a change in the price of a
candidate, not in the complexity class. Choosing ``S`` is still hard -- ranking
designs by the bound is a cardinality-constrained max-cut. But a candidate costs
one row of a matrix product instead of a node with an LP relaxation, and the
panel length drops out entirely after the Gram step, so a long panel costs no
more per design than a short one. Above ``candidate_limit`` admissible designs,
enumeration stops being affordable and the search falls back to max-cut candidates
polished on the objective, reported against the Rayleigh bound.

Restrictions on the assignment are predicates on the treated set, and they split
in two. Forced units, forbidden units and stratum quotas are structural: they
decide which candidates exist, so
:mod:`mlsynth.utils.syndes_helpers.enumeration` builds candidates inside them and
counts the result exactly. That count is what ``candidate_limit`` is measured
against, so an instance with most of its units forbidden is enumerated on the
designs it has instead of the designs it would have had. Conflict pairs, a cost
budget and the pool's no-good sets do not decompose over a group, so they stay
tests on a finished candidate and cannot lower the count.

``donor_exclusion`` is refused outright: it ties the control weights to which
units are treated, so the closed form's matrix would differ per candidate and the
batch product would not hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..fast_scm_helpers.structure import IndexSet
from .enumeration import SearchSpace
from .gram import (
    BoundEngine,
    TwoWayProblem,
    maxcut_candidates,
    signs_from_subsets,
)
from .partition import solve_partition_batch, split_indices
from .structures import SYNDESDesign

#: Above this many candidate designs, enumeration stops being affordable and the
#: search falls back to max-cut candidates reported against the Rayleigh bound.
CANDIDATE_LIMIT = 20_000_000

#: Designs generated per chunk while streaming. Bounds peak memory without
#: changing any result.
CHUNK = 200_000


# ----------------------------------------------------------------------
# candidate generation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetFilter:
    """Predicates on the treated set, applied while candidates are generated.

    Every restriction the two-way design supports is a statement about which
    units are treated, so each is a test on a candidate, not a constraint handed
    to a solver.
    """

    forced_in: frozenset = frozenset()
    forbidden: frozenset = frozenset()
    conflict_pairs: Tuple[Tuple[int, int], ...] = ()
    strata: Tuple[Tuple[Tuple[int, ...], Optional[int], Optional[int]], ...] = ()
    costs: Optional[np.ndarray] = None
    budget: Optional[float] = None
    forbidden_sets: Tuple[frozenset, ...] = ()

    @classmethod
    def build(
        cls,
        restrictions: Optional[Any] = None,
        costs: Optional[np.ndarray] = None,
        budget: Optional[float] = None,
        forbidden_sets: Optional[Sequence[np.ndarray]] = None,
        N: Optional[int] = None,
    ) -> "SubsetFilter":
        """Assemble the predicate set, validating cost inputs.

        Raises
        ------
        MlsynthConfigError
            If ``costs`` and ``budget`` are not supplied together, if ``costs``
            has the wrong length or a negative entry, if ``budget`` is not
            positive, or if ``restrictions`` carries a donor exclusion.
        """
        if restrictions is not None and getattr(restrictions, "donor_exclusion", None):
            raise MlsynthConfigError(
                "donor exclusions are not supported by the exact two-way backend: "
                "they tie the control weights to which units are treated, so the "
                "design cannot be scored as a function of the treated set alone. "
                "Drop the donor rules, or use mode='one_way_global' / 'per_unit'."
            )
        if (costs is None) != (budget is None):
            raise MlsynthConfigError(
                "costs and budget must be supplied together (or both None).")

        cost_vec = None
        if costs is not None:
            cost_vec = np.asarray(costs, dtype=float).reshape(-1)
            if N is not None and cost_vec.shape[0] != N:
                raise MlsynthConfigError(
                    f"costs must have length N={N}; got {cost_vec.shape[0]}.")
            if np.any(cost_vec < 0):
                raise MlsynthConfigError("costs must be non-negative.")
            if float(budget) <= 0:
                raise MlsynthConfigError("budget must be strictly positive.")

        return cls(
            forced_in=frozenset(int(i) for i in getattr(restrictions, "forced_in", ())),
            forbidden=frozenset(int(i) for i in getattr(restrictions, "forbidden", ())),
            conflict_pairs=tuple(
                (int(i), int(j)) for i, j in getattr(restrictions, "conflict_pairs", ())),
            strata=tuple(
                (tuple(int(m) for m in members), lo, hi)
                for members, lo, hi in getattr(restrictions, "strata", ())),
            costs=cost_vec,
            budget=None if budget is None else float(budget),
            forbidden_sets=tuple(
                frozenset(int(i) for i in np.asarray(fs).reshape(-1))
                for fs in (forbidden_sets or []) if np.asarray(fs).size),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.forced_in or self.forbidden or self.conflict_pairs
                    or self.strata or self.costs is not None or self.forbidden_sets)

    def accepts(self, subset: Tuple[int, ...]) -> bool:
        """Whether a candidate treated set satisfies every predicate."""
        chosen = frozenset(subset)
        if not self.forced_in <= chosen:
            return False
        if chosen & self.forbidden:
            return False
        for i, j in self.conflict_pairs:
            if i in chosen and j in chosen:
                return False
        for members, lower, upper in self.strata:
            inside = len(chosen.intersection(members))
            if lower is not None and inside < lower:
                return False
            if upper is not None and inside > upper:
                return False
        if self.costs is not None:
            if float(self.costs[list(subset)].sum()) > self.budget + 1e-9:
                return False
        if chosen in self.forbidden_sets:
            return False
        return True


def _search_space(N: int, K: int, filt: SubsetFilter) -> SearchSpace:
    """The structural half of ``filt``, resolved into a space to walk.

    Forced units, forbidden units and stratum quotas shape which candidates
    exist, so they are handed to the generator. What remains is applied to
    finished candidates.
    """
    return SearchSpace.build(N, K, forced=filt.forced_in,
                             forbidden=filt.forbidden, strata=filt.strata)


def _stream_from_space(space: SearchSpace, filt: SubsetFilter, chunk: int):
    """Yield ``(B, K)`` arrays of admissible treated sets, ``B <= chunk``.

    The walk already honours the structural restrictions, so ``filt`` is consulted
    here for the ones that do not decompose over a group: conflict pairs, the cost
    budget, and the pool's no-good sets. Blocks fill to ``chunk`` from whatever
    survives, so a long rejected run delays a yield instead of shrinking one.
    """
    block: List[Tuple[int, ...]] = []
    for subset in space.iter_subsets():
        if not filt.accepts(subset):
            continue
        block.append(subset)
        if len(block) == chunk:
            yield np.array(block, dtype=np.int64)
            block = []
    if block:
        yield np.array(block, dtype=np.int64)


def _stream_subsets(N: int, K: int, filt: SubsetFilter, chunk: int):
    """Yield ``(B, K)`` arrays of admissible treated sets, ``B <= chunk``."""
    yield from _stream_from_space(_search_space(N, K, filt), filt, chunk)


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------


@dataclass
class SearchOutcome:
    """One design plus how the search arrived at it."""

    treated: np.ndarray
    value: float
    treated_weights: np.ndarray
    control_weights: np.ndarray
    duality_gap: float
    lower_bound: Optional[float]
    certified: bool
    n_designs: int = 0
    n_closed_form: int = 0
    n_solved: int = 0


def _settle(problem: TwoWayProblem, engine: BoundEngine, treated: np.ndarray):
    """Best available value and weights for one treated set.

    Prefers the closed form when its minimiser respects the sign conditions --
    there it is the optimum at machine precision. The iterative solver loses
    relative accuracy as ``lam`` approaches zero, and the closed form does not.
    """
    treated = np.sort(np.asarray(treated, dtype=np.int64))
    t_idx, c_idx = split_indices(problem.N, treated[None, :])

    if engine.reliable:
        closed = engine.bound(signs_from_subsets(problem.N, treated[None, :]))
        if bool(closed.exact[0]):
            u = closed.contrast[0]
            return (float(closed.lower[0]), np.maximum(u, 0.0),
                    np.maximum(-u, 0.0), 0.0)

    solved = solve_partition_batch(problem, t_idx, c_idx,
                                   max_iter=20_000, tol=1e-15)
    a = np.zeros(problem.N); a[t_idx[0]] = solved.treated_weights[0]
    b = np.zeros(problem.N); b[c_idx[0]] = solved.control_weights[0]
    return float(solved.value[0]), a, b, float(solved.gap[0])


def _search(
    problem: TwoWayProblem,
    K: int,
    filt: SubsetFilter,
    *,
    candidate_limit: int,
    incumbent_sets: Optional[np.ndarray],
    seed: int,
) -> SearchOutcome:
    """Enumerate and prune, or fall back to max-cut candidates when too large."""
    engine = BoundEngine.from_problem(problem)
    N = problem.N
    space = _search_space(N, K, filt)

    if space.size > candidate_limit or not engine.reliable:
        return _search_by_candidates(problem, engine, K, filt, seed=seed,
                                     n_designs=space.size)

    best_value, best_set = np.inf, None
    if incumbent_sets is not None and len(np.atleast_2d(incumbent_sets)):
        for seed_set in np.atleast_2d(np.asarray(incumbent_sets, dtype=np.int64)):
            if filt.accepts(tuple(int(i) for i in np.sort(seed_set))):
                value, _, _, _ = _settle(problem, engine, seed_set)
                if value < best_value:
                    best_value, best_set = value, np.sort(seed_set)

    kept: List[np.ndarray] = []
    kept_bounds: List[np.ndarray] = []
    n_designs = 0
    n_closed_form = 0

    for block in _stream_from_space(space, filt, CHUNK):
        n_designs += block.shape[0]
        scored = engine.bound(signs_from_subsets(N, block))
        if scored.exact.any():
            n_closed_form += int(scored.exact.sum())
            local = int(np.argmin(np.where(scored.exact, scored.lower, np.inf)))
            if scored.lower[local] < best_value:
                best_value, best_set = float(scored.lower[local]), block[local]
        alive = np.flatnonzero((~scored.exact) & (scored.lower < best_value))
        if alive.size:
            kept.append(block[alive])
            kept_bounds.append(scored.lower[alive])

    if n_designs == 0:
        raise MlsynthEstimationError(
            "SYNDES found no feasible design: the active restrictions "
            "(forced/forbidden units, spillover/adjacency conflict, stratum "
            "quotas, cost budget) are jointly unsatisfiable for the requested K. "
            "Relax the restrictions or reduce K.")

    n_solved = 0
    if kept:
        candidates = np.concatenate(kept)
        bounds = np.concatenate(kept_bounds)
        alive = bounds < best_value
        candidates = candidates[alive]
        n_solved = int(candidates.shape[0])
        if n_solved:
            t_idx, c_idx = split_indices(N, candidates)
            solved = solve_partition_batch(problem, t_idx, c_idx,
                                           max_iter=4000, tol=1e-14)
            j = int(np.argmin(solved.value))
            if float(solved.value[j]) < best_value:
                best_value, best_set = float(solved.value[j]), candidates[j]

    value, a, b, gap = _settle(problem, engine, best_set)
    scale = max(abs(value), 1e-300)
    return SearchOutcome(
        treated=np.sort(np.asarray(best_set, dtype=np.int64)),
        value=value, treated_weights=a, control_weights=b,
        duality_gap=gap, lower_bound=engine.global_lower_bound(K),
        certified=bool(gap / scale <= 1e-8),
        n_designs=n_designs, n_closed_form=n_closed_form, n_solved=n_solved,
    )


def _search_by_candidates(
    problem: TwoWayProblem,
    engine: BoundEngine,
    K: int,
    filt: SubsetFilter,
    *,
    seed: int,
    n_designs: int,
) -> SearchOutcome:
    """Max-cut candidates polished on the objective, for instances too large to enumerate."""
    if not engine.reliable:
        raise MlsynthEstimationError(
            "the two-way Gram matrix is too ill-conditioned for the exact backend "
            f"(cond(M) = {engine.condition:.3e}), and the instance is too large to "
            "enumerate. Raise lam, add pre-treatment periods, or use the MIP backend.")
    if not filt.is_empty:
        raise MlsynthConfigError(
            f"the exact backend cannot apply design restrictions to an instance "
            f"with {n_designs} candidate designs, which is past the enumeration "
            "limit. Reduce K, drop the restrictions, or raise candidate_limit.")

    candidates = maxcut_candidates(engine, K, seed=seed)
    t_idx, c_idx = split_indices(problem.N, candidates)
    ranked = solve_partition_batch(problem, t_idx, c_idx, max_iter=2000, tol=1e-13)
    j = int(np.argmin(ranked.value))

    value, a, b, gap = _settle(problem, engine, candidates[j])
    return SearchOutcome(
        treated=np.sort(candidates[j]), value=value,
        treated_weights=a, control_weights=b, duality_gap=gap,
        lower_bound=engine.global_lower_bound(K), certified=False,
        n_designs=n_designs, n_closed_form=0, n_solved=int(candidates.shape[0]),
    )


# ----------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------


def solve_synthetic_design_exact(
    Y: np.ndarray,
    K: Optional[int],
    *,
    lam: Optional[float] = None,
    unit_index: Optional[IndexSet] = None,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    restrictions: Optional[Any] = None,
    forbidden_sets: Optional[Sequence[np.ndarray]] = None,
    incumbent_sets: Optional[np.ndarray] = None,
    candidate_limit: int = CANDIDATE_LIMIT,
    seed: int = 0,
) -> SYNDESDesign:
    """Solve the two-way global design by searching treated sets.

    Returns the same :class:`SYNDESDesign` as
    :func:`mlsynth.utils.syndes_helpers.optimization.solve_synthetic_design` with
    ``mode="global_2way"``, so the two backends are interchangeable downstream.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix of shape ``(T, N)``.
    K : int or None
        Number of treated units. ``None`` searches every size from ``1`` to
        ``N - 1``, which the paper allows for the global modes.
    lam : float, optional
        Penalty on the weights; estimated from ``Y`` when ``None``, matching the
        MIP path so both minimise the same objective.
    unit_index : IndexSet, optional
        Label mapping, used to populate the design's label fields.
    costs, budget : optional
        Per-unit treatment cost and a cap on their total over the treated set.
    restrictions : DesignRestrictions, optional
        Assignment restrictions. Donor exclusions are refused.
    forbidden_sets : sequence of arrays, optional
        Treated sets to exclude, which is how the pool asks for the next-best
        distinct design.
    incumbent_sets : np.ndarray, optional
        Treated sets to seed the incumbent with, so more designs prune on their
        bound alone. A poor seed costs time and nothing else.
    candidate_limit : int, optional
        Above this many candidate designs the search stops enumerating and falls
        back to max-cut candidates reported against the Rayleigh bound.
    seed : int, optional
        Seed for the fallback's restarts.

    Raises
    ------
    MlsynthConfigError
        For an out-of-range ``K``, malformed cost inputs, or donor exclusions.
    MlsynthEstimationError
        When no admissible design exists under the restrictions.
    """
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    N = problem.N
    filt = SubsetFilter.build(restrictions, costs, budget, forbidden_sets, N=N)

    if K is None:
        sizes = range(1, N)
    else:
        if not 1 <= int(K) <= N - 1:
            raise MlsynthConfigError(
                f"K must lie in 1..{N - 1} for a two-way design; got {K}.")
        sizes = [int(K)]

    best: Optional[SearchOutcome] = None
    failures: List[str] = []
    for size in sizes:
        try:
            outcome = _search(problem, size, filt,
                              candidate_limit=candidate_limit,
                              incumbent_sets=incumbent_sets, seed=seed)
        except MlsynthEstimationError as exc:
            failures.append(str(exc))
            continue
        if best is None or outcome.value < best.value:
            best = outcome

    if best is None:
        raise MlsynthEstimationError(
            failures[0] if failures else
            "SYNDES found no feasible design under the active restrictions.")

    return _to_design(problem, Y, best, unit_index)


def solve_synthetic_design_exact_pool(
    Y: np.ndarray,
    K: int,
    top_K: int = 1,
    **kwargs: Any,
) -> List[SYNDESDesign]:
    """The ``top_K`` best distinct designs, ranked by objective.

    Re-searches with each chosen treated set added to ``forbidden_sets``, so the
    next call returns the next-best distinct design. Stops early, returning fewer
    than ``top_K``, once the admissible region is exhausted.

    Raises
    ------
    MlsynthConfigError
        If ``top_K`` is not a positive integer.
    """
    if not isinstance(top_K, (int, np.integer)) or top_K < 1:
        raise MlsynthConfigError(f"top_K must be a positive integer; got {top_K!r}.")

    excluded = list(kwargs.pop("forbidden_sets", None) or [])
    designs: List[SYNDESDesign] = []
    for _ in range(int(top_K)):
        try:
            design = solve_synthetic_design_exact(
                Y, K, forbidden_sets=excluded, **kwargs)
        except MlsynthEstimationError:
            break
        designs.append(design)
        excluded.append(np.asarray(design.selected_unit_indices, dtype=np.int64))
    return designs


def _to_design(
    problem: TwoWayProblem,
    Y: np.ndarray,
    outcome: SearchOutcome,
    unit_index: Optional[IndexSet],
) -> SYNDESDesign:
    """Package a search outcome in the shared design container.

    Mirrors ``optimization._extract_weights``' two-way convention so the two
    backends are interchangeable downstream: ``q`` is the treated block, ``w - q``
    the control block, and the contrast is their difference.
    """
    N = problem.N
    assignment = np.zeros(N, dtype=int)
    assignment[outcome.treated] = 1

    if unit_index is not None:
        selected_labels = unit_index.get_labels(outcome.treated)
        all_labels = unit_index.labels
    else:
        selected_labels = outcome.treated
        all_labels = np.arange(N)

    a, b = outcome.treated_weights, outcome.control_weights
    contrast_weights = a - b
    contrast_series = np.asarray(Y, dtype=float) @ contrast_weights
    pre_fit_rmse = float(np.sqrt(np.mean(contrast_series ** 2)))
    return SYNDESDesign(
        mode="global_2way",
        objective_value=outcome.value,
        lambda_value=problem.lam,
        assignment=assignment,
        selected_unit_indices=outcome.treated,
        selected_unit_labels=np.asarray(selected_labels),
        assignment_by_unit={
            label: int(val) for label, val in zip(all_labels, assignment)},
        w=a + b,
        q=a,
        z=None,
        treated_weights=a,
        control_weights=b,
        contrast_weights=contrast_weights,
        contrast_series=contrast_series,
        pre_fit_rmse=pre_fit_rmse,
        raw_results={
            "status": "optimal" if outcome.certified else "feasible",
            "solver": "exact",
            "n_designs": outcome.n_designs,
            "n_closed_form": outcome.n_closed_form,
            "n_solved": outcome.n_solved,
            "duality_gap": outcome.duality_gap,
            "lower_bound": outcome.lower_bound,
            "certified": outcome.certified,
        },
    )
