"""The exact one-way SYNDES backend: enumeration under a ridge bound.

Doudchenko et al. (2021, Section 2 and Appendix A.2) define the one-way design as
the two-way design with the treated weights fixed at ``1/K``. Naming the treated
set ``S`` leaves one simplex fit,

    f(S) = lam/K + min over c on the simplex over P of
           (1/T)||ybar_S - Y_P c||^2 + lam ||c||^2,

where ``ybar_S`` is the treated units' average and ``P`` is ``S``'s complement
less every donor a restriction excludes for a unit of ``S``.

Written on contrasts ``u`` (``u_S = 1/K``, ``u_P = -c``) this is
``min u'(G + lam I)u`` subject to ``u_S = 1/K``, ``1'u = 0`` and
``u_F <= 0`` off ``S``, with ``G`` the Gram matrix of
:mod:`~.gram`. Dropping only the controls' sign conditions (and the donor
exclusions, which only shrink the pool) leaves two blocks of equalities and a
closed form. With ``H = (G + lam I)^{-1}``, ``C = [E_S, 1]`` and
``d = [1/K; 0]``,

    ridge(S) = d' (C'HC)^{-1} d,

one ``(K + 1) x (K + 1)`` solve per set, batched. Its constraints imply the
two-way relaxation's, so it is at least :mod:`~.gram`'s ``lb(S)``. On 132 study
runs (``benchmarks/studies/syndes_one_way_exact/``) it equalled the optimum to
rounding on 126 and was within 0.2% on the rest: with the treated weights fixed,
the best equality-constrained control is already nonnegative, or nearly so.

The search enumerates the admissible treated sets through the two-way backend's
:class:`~.enumeration.SearchSpace` and :class:`~.exact.SubsetFilter`, bounds them
in blocks, prices each block's best-bounded set as it streams so that later
blocks keep only the sets whose bound is below that incumbent, and then prices
the survivors in order of their bound until the next bound reaches the
incumbent. Every set left unpriced has a bound at or above the incumbent, so the
incumbent is the optimum, to the rounding of the bound. On the study's runs that
took one fit. Where ``G + lam I`` is singular to working precision (``lam = 0``
on a panel with fewer periods than units, or a unit that is an exact combination
of others), no bound can be formed; ``f(S) >= 0`` is then the bound, and the
search prices every admissible set, up to :data:`PRICE_ALL_LIMIT`.

Pricing runs on the panel centred within each period: the treated average and
the control combination both carry weights that sum to one, so a level common to
the units cancels from the residual. On the uncentred panel at a level of
``1e8``, 222 of 3,328 fits did not converge; centred, every one did, within the
rounding of the stored data. The panel is not rescaled. Cold-started, as here,
the active-set fits agreed with fits on a unit-scale panel to ``7e-16`` relative
at every scale from ``1e-10`` to ``1e10``, so a change of units would buy
nothing. Every fit is ``solve_simplex_qp``, its convergence read and its
Frank-Wolfe gap recorded.

Above ``candidate_limit`` admissible sets, enumeration stops being affordable.
The search then prices the two-way backend's swap-search candidates
(:func:`~.gram.maxcut_candidates`) as one-way designs and reports the best
against the Rayleigh bound over every size-``K`` set, which bounds one-way as
well (``one-way >= two-way >= Rayleigh``), with ``certified=False`` and a
:class:`~.optimization.SYNDESSolveWarning`. Treated-side restrictions are refused
there, since the candidates do not respect them; donor exclusions apply.
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
from .exact import (
    CANDIDATE_LIMIT,
    SubsetFilter,
    _NoFeasibleDesign,
    _search_space,
    _stream_from_space,
)
from .gram import BoundEngine, TwoWayProblem, maxcut_candidates
from .optimization import SYNDESSolveWarning, estimate_lambda
from .structures import SYNDESDesign, SYNDESSolveOutcome

#: Treated sets bounded per block while streaming. The design does not depend
#: on it; 20,000 keeps a block's arrays in cache (1.1 µs a set at 40 units,
#: against 1.3 µs at 200,000).
CHUNK = 20_000

#: Admissible treated sets the search prices one by one when ``G + lam I`` is
#: singular to working precision and no bound can be formed. Fits on such a
#: panel are degenerate and took 1 to 10 ms each at 20 to 100 units, so this is
#: about 20 s at 40.
PRICE_ALL_LIMIT = 10_000


class OneWayFits:
    """``f(S)`` for any treated set, with its control weights.

    The fits run on ``Y`` centred within each period, in the outcome's units.

    Parameters
    ----------
    Y : np.ndarray
        ``(T, N)`` pre-period panel.
    lam : float
        Penalty on the weights, in the outcome's units.
    excluded : dict, optional
        Unit ``i`` to the donors no design treating ``i`` may use.
    """

    def __init__(self, Y: np.ndarray, lam: float,
                 excluded: Optional[Dict[int, FrozenSet[int]]] = None):
        Y = np.asarray(Y, dtype=float)
        self.Y = Y - Y.mean(axis=1, keepdims=True)
        self.T, self.N = self.Y.shape
        self.lam = float(lam)
        self.excluded = {int(i): frozenset(int(j) for j in js)
                         for i, js in (excluded or {}).items()}
        self.n_fits = 0
        self.max_rel_gap = 0.0
        self._cache: Dict[Tuple[int, ...], Tuple[float, Optional[np.ndarray]]] = {}

    def price(self, S: Sequence[int]) -> Tuple[float, Optional[np.ndarray]]:
        """``f(S)`` and the control weights (dense, length ``N``).

        ``(inf, None)`` when the exclusions leave ``S`` no control.

        Raises
        ------
        MlsynthEstimationError
            If the fit does not converge.
        """
        key = tuple(sorted(int(i) for i in S))
        if key in self._cache:
            return self._cache[key]
        banned = set(key)
        for i in key:
            banned |= self.excluded.get(i, frozenset())
        pool = [j for j in range(self.N) if j not in banned]
        if not pool:
            self._cache[key] = (math.inf, None)
            return self._cache[key]

        K = len(key)
        root_t = math.sqrt(self.T)
        B = self.Y[:, pool] / root_t
        A = self.Y[:, list(key)].mean(axis=1) / root_t
        if self.lam > 0.0:
            B = np.vstack([B, math.sqrt(self.lam) * np.eye(len(pool))])
            A = np.concatenate([A, np.zeros(len(pool))])
        c, info = solve_simplex_qp(B, A, return_info=True)
        if not info["converged"]:
            raise MlsynthEstimationError(
                f"SYNDES one-way exact backend: the fit of treated set {key} on a "
                f"pool of {len(pool)} controls did not converge, so its value "
                "certifies nothing.")
        r = A - B @ c
        value = float(r @ r)
        grad = -2.0 * (B.T @ r)
        if value > 0.0:
            self.max_rel_gap = max(self.max_rel_gap,
                                   abs(float(grad @ c - grad.min())) / value)
        self.n_fits += 1
        weights = np.zeros(self.N)
        weights[pool] = c
        self._cache[key] = (value + self.lam / K, weights)
        return self._cache[key]


def ridge_bound(engine: BoundEngine, sets: np.ndarray) -> np.ndarray:
    """``d'(C'HC)^{-1}d`` for each row of a ``(B, K)`` batch of treated sets.

    The minimum of ``u'(G + lam I)u`` over ``u_S = 1/K`` and ``1'u = 0``: the
    one-way program without the controls' sign conditions. ``C'HC`` is
    ``[[H_SS, p_S], [p_S', alpha]]`` with ``p = H1`` and ``alpha = 1'H1``,
    positive definite whenever ``K < N``.
    """
    sets = np.atleast_2d(np.asarray(sets, dtype=np.int64))
    B, K = sets.shape
    A = np.empty((B, K + 1, K + 1))
    A[:, :K, :K] = engine.H[sets[:, :, None], sets[:, None, :]]
    A[:, :K, K] = engine.p[sets]
    A[:, K, :K] = engine.p[sets]
    A[:, K, K] = engine.alpha
    d = np.zeros((B, K + 1))
    d[:, :K] = 1.0 / K
    x = np.linalg.solve(A, d[..., None])[..., 0]
    return np.einsum("bi,bi->b", d, x)


@dataclasses.dataclass
class _Outcome:
    treated: Tuple[int, ...]
    value: float
    control_weights: np.ndarray
    lower_bound: Optional[float]
    certified: bool
    n_designs: int
    n_survivors: int
    n_priced: int


def _search(fits: OneWayFits, engine: BoundEngine, K: int, filt: SubsetFilter,
            candidate_limit: int, seed: int) -> _Outcome:
    """Enumerate and price in bound order, or fall back above the limit."""
    N = fits.N
    space = _search_space(N, K, filt)
    if not engine.reliable and space.size > min(candidate_limit, PRICE_ALL_LIMIT):
        raise _too_singular(engine, space.size, min(candidate_limit, PRICE_ALL_LIMIT))
    if space.size > candidate_limit:
        return _search_by_candidates(fits, engine, K, filt, seed=seed,
                                     n_designs=space.size)

    best, best_set = math.inf, None
    kept: List[np.ndarray] = []
    kept_bounds: List[np.ndarray] = []
    n_designs = 0
    priced = set()
    for block in _stream_from_space(space, filt, CHUNK):
        n_designs += block.shape[0]
        # f(S) >= 0, so a bound of zero is valid where none can be formed, and
        # the search then prices every set.
        bounds = (ridge_bound(engine, block) if engine.reliable
                  else np.zeros(block.shape[0]))
        j = int(np.argmin(bounds))
        if bounds[j] < best:
            S = tuple(int(i) for i in block[j])
            value = fits.price(S)[0]
            priced.add(S)
            if value < best:
                best, best_set = value, S
        alive = bounds < best
        if alive.any():
            kept.append(block[alive])
            kept_bounds.append(bounds[alive])

    if n_designs == 0:
        raise _NoFeasibleDesign(
            "SYNDES found no feasible design: the active restrictions "
            "(forced/forbidden units, conflicts, stratum quotas, cost budget) "
            "are jointly unsatisfiable for the requested K. Relax the "
            "restrictions or reduce K.")

    n_survivors = 0
    if kept:
        sets = np.concatenate(kept)
        bounds = np.concatenate(kept_bounds)
        n_survivors = int(sets.shape[0])
        for k in np.argsort(bounds, kind="stable"):
            if bounds[k] >= best:
                break
            S = tuple(int(i) for i in sets[k])
            value = fits.price(S)[0]
            priced.add(S)
            if value < best:
                best, best_set = value, S

    if best_set is None:
        raise _NoFeasibleDesign(
            "SYNDES found no feasible design: the donor exclusions leave every "
            "admissible treated set without a control. Relax the restrictions "
            "or reduce K.")
    return _Outcome(treated=best_set, value=best,
                    control_weights=fits.price(best_set)[1], lower_bound=best,
                    certified=True, n_designs=n_designs, n_survivors=n_survivors,
                    n_priced=len(priced))


def _too_singular(engine: BoundEngine, n_sets: int,
                  limit: int) -> MlsynthEstimationError:
    return MlsynthEstimationError(
        "SYNDES one-way exact backend: G + lam I is singular to working "
        f"precision (cond = {engine.condition:.3e}), so no bound can be formed, "
        f"and {n_sets} admissible treated sets are more than the {limit} it "
        "prices one by one. This happens when lam is zero on a panel with fewer "
        "pre-periods than units, or with a unit that is an exact combination of "
        "others. Raise lam, add pre-treatment periods, or use backend='mip'.")


def _search_by_candidates(fits: OneWayFits, engine: BoundEngine, K: int,
                          filt: SubsetFilter, *, seed: int,
                          n_designs: int) -> _Outcome:
    """Swap-search candidates priced as one-way designs, for instances too large
    to enumerate."""
    if not filt.is_empty:
        raise MlsynthConfigError(
            f"the exact one-way backend cannot apply treated-side restrictions to "
            f"an instance with {n_designs} candidate designs, which is past the "
            "enumeration limit. Reduce K, drop the restrictions, raise "
            "candidate_limit, or use backend='mip'.")
    candidates = maxcut_candidates(engine, K, seed=seed)
    values = [fits.price(S)[0] for S in candidates]
    j = int(np.argmin(values))
    if math.isinf(values[j]):
        raise MlsynthEstimationError(
            "SYNDES one-way exact backend: the donor exclusions leave every "
            "candidate of the swap search without a control. Raise "
            "candidate_limit or use backend='mip'.")
    S = tuple(int(i) for i in candidates[j])
    return _Outcome(treated=S, value=float(values[j]),
                    control_weights=fits.price(S)[1],
                    lower_bound=engine.global_lower_bound(K), certified=False,
                    n_designs=n_designs, n_survivors=0,
                    n_priced=int(candidates.shape[0]))


def solve_one_way_exact(
    Y: np.ndarray,
    K: int,
    *,
    lam: Optional[float] = None,
    unit_index: Optional[IndexSet] = None,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    restrictions: Optional[Any] = None,
    forbidden_sets: Optional[Sequence[np.ndarray]] = None,
    candidate_limit: int = CANDIDATE_LIMIT,
    seed: int = 0,
) -> SYNDESDesign:
    """Solve the one-way design by enumerating treated sets under a ridge bound.

    Returns the same :class:`SYNDESDesign` as
    :func:`~.optimization.solve_synthetic_design` with
    ``mode="global_equal_weights"``, so the two backends are interchangeable
    downstream, and fills ``solve_outcome`` with the search's own bound.

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
        Assignment restrictions. Donor exclusions shrink the control pool of
        every set that treats the excluding unit; the rest restrict which units
        are treated.
    forbidden_sets : sequence of arrays, optional
        Treated sets to exclude, which is how the pool asks for the next design.
    candidate_limit : int, optional
        Above this many admissible treated sets the search prices the swap
        search's candidates instead, uncertified.
    seed : int, optional
        Seed for the swap search's restarts.

    Raises
    ------
    MlsynthDataError
        If ``Y`` is not a two-dimensional array.
    MlsynthConfigError
        For an out-of-range ``K``, a negative ``lam``, malformed costs, or
        treated-side restrictions on an instance past ``candidate_limit``.
    MlsynthEstimationError
        When no admissible design exists, a fit does not converge, or the
        problem on contrasts is singular to working precision and has more
        admissible sets than :data:`PRICE_ALL_LIMIT`.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise MlsynthDataError("Y must be a two-dimensional T x N matrix.")
    T, N = Y.shape
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or not 1 <= K <= N - 1:
        raise MlsynthConfigError(
            f"K must be an integer in 1..{N - 1} for a one-way design; got {K!r}.")
    K = int(K)
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
    engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=lam_value))
    fits = OneWayFits(Y, lam_value, {i: frozenset(js) for i, js in excluded.items()})
    outcome = _search(fits, engine, K, filt, int(candidate_limit), int(seed))
    seconds = perf_counter() - t0

    lower = outcome.lower_bound
    gap = ((outcome.value - lower) / outcome.value
           if lower is not None and outcome.value > 0.0 else 0.0)
    status = "optimal" if outcome.certified else "candidate_limit"
    if not outcome.certified:
        warnings.warn(
            f"SYNDES one-way exact backend: {outcome.n_designs} admissible "
            f"treated sets exceed the candidate limit ({candidate_limit}), so the "
            "design is the best of the swap search's candidates, with a gap of "
            f"{gap:.3g} to the Rayleigh bound; it is not a proven optimum. Raise "
            "candidate_limit to search every set.",
            SYNDESSolveWarning, stacklevel=2)
    solve_outcome = SYNDESSolveOutcome(
        status=status, solver_status="exact",
        stopped_at_limit=not outcome.certified, objective=outcome.value,
        solver_objective=outcome.value, dual_bound=lower, gap=gap,
        solve_time=seconds, limits_applied=True)
    return _to_design(Y, outcome, lam_value, K, unit_index, solve_outcome,
                      fits.n_fits, fits.max_rel_gap)


def solve_one_way_exact_pool(
    Y: np.ndarray,
    K: int,
    top_K: int = 1,
    **kwargs: Any,
) -> List[SYNDESDesign]:
    """The ``top_K`` best distinct one-way designs, ranked by objective.

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
            design = solve_one_way_exact(Y, K, forbidden_sets=excluded, **kwargs)
        except _NoFeasibleDesign:
            break
        designs.append(design)
        excluded.append(np.asarray(design.selected_unit_indices, dtype=np.int64))
    return designs


def _to_design(Y: np.ndarray, outcome: _Outcome, lam: float, K: int,
               unit_index: Optional[IndexSet],
               solve_outcome: SYNDESSolveOutcome, n_fits: int,
               max_fit_gap: float) -> SYNDESDesign:
    """Package a search outcome in the shared design container.

    Mirrors ``optimization._extract_weights``' one-way convention: treated
    weights ``1/K`` on the treated set, ``control_weights`` the fitted ``c``, and
    the contrast their difference. ``w``, ``q`` and ``z`` are ``None``, as on the
    MIP path.
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
    treated_weights = assignment / K
    control_weights = outcome.control_weights
    contrast_weights = treated_weights - control_weights
    contrast_series = Y @ contrast_weights
    return SYNDESDesign(
        mode="global_equal_weights",
        objective_value=outcome.value,
        lambda_value=lam,
        assignment=assignment,
        selected_unit_indices=treated,
        selected_unit_labels=np.asarray(selected_labels),
        assignment_by_unit={
            label: int(val) for label, val in zip(all_labels, assignment)},
        w=None,
        q=None,
        z=None,
        treated_weights=treated_weights,
        control_weights=control_weights,
        contrast_weights=contrast_weights,
        contrast_series=contrast_series,
        pre_fit_rmse=float(np.sqrt(np.mean(contrast_series ** 2))),
        raw_results={
            "status": solve_outcome.status,
            "solver": "exact",
            "certified": outcome.certified,
            "lower_bound": outcome.lower_bound,
            "n_designs": outcome.n_designs,
            "n_survivors": outcome.n_survivors,
            "n_priced": outcome.n_priced,
            "n_fits": n_fits,
            "max_fit_gap": max_fit_gap,
        },
        solve_outcome=solve_outcome,
    )
