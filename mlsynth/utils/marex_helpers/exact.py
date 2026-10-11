"""MAREX's exact backend: the design as a bounded search over treated sets.

MAREX's MIQP chooses, per cluster, a treated set ``S``, treated weights ``w`` on
the simplex over ``S`` and control weights ``v`` on the simplex over the rest of
the cluster. For the ``standard``, ``penalized`` and ``unit_penalized`` designs
the objective is a sum of a treated-side term and a control-side term,

    ||A - B w||^2 + a'w   +   ||A - B v||^2 + b'v,

with ``A`` the cluster's predictor mean, the columns of ``B`` its markets, and
``a``, ``b`` non-negative penalty vectors (zero for ``standard``). Once ``S`` is
named the two terms share nothing, so

    f(S) = g(S) + h(C(S)),

two simplex-constrained least-squares fits, each solved exactly by
:func:`mlsynth.utils.solvers.active_set.solve_simplex_qp`. ``C(S)`` is the
cluster less ``S`` and less any donor a treated market excludes. The design is
a search over treated sets with an exact oracle, not a mixed-integer program.

The search. Both terms are non-negative, so ``f(S) >= g(S) >= lb(S)`` for any
lower bound ``lb`` on the treated fit. Every admissible candidate is priced by
``lb`` in batches, and only candidates whose bound is under the best total found
reach the oracle. A single streaming pass keeps the ``BUFFER`` best-bounded
candidates and evaluates them in bound order, which sets the incumbent; if more
candidates than the buffer held have a bound under it, a second pass collects
and evaluates those. Every candidate is therefore either solved exactly or
discarded on a bound that clears the incumbent, and the result is the optimum.

The bound. Because the weights sum to one, subtracting the target from every
column changes nothing: ``||A - B w|| = ||(B - A 1') w||``. On those centred
columns the target is zero, and dropping only ``w >= 0`` leaves a closed form,

    min { w'G w + a'w : 1'w = 1 } = (2 + t)^2 / (4 s) - u / 4,

with ``G = B~'B~`` restricted to ``S``, ``s = 1'G^-1 1``, ``t = 1'G^-1 a`` and
``u = a'G^-1 a`` (``1 / s`` when ``a = 0``). It equals ``g(S)`` whenever its
minimiser is already non-negative. SYNDES's exact backend bounds its own
objective the same way (:mod:`mlsynth.utils.syndes_helpers.gram`).

Restrictions. Forced units, forbidden units and stratum quotas decide which
candidates exist, so :class:`~mlsynth.utils.syndes_helpers.enumeration.SearchSpace`
builds candidates inside them and counts them exactly; conflict pairs are a
test on a finished candidate (:class:`~mlsynth.utils.syndes_helpers.exact.SubsetFilter`).
Donor exclusions shrink each candidate's control pool, which the bound never
sees, since it reads only the treated side. Clusters are searched independently,
so a conflict pair or a stratum that spans two clusters couples their searches
and is refused.

What this backend does not solve. ``design="weakly_targeted"`` couples the two
sides; a cost budget constrains the treated weights; ``relaxed`` and the
branch-and-bound knobs (``solver``, ``warm_start``, ``time_limit``) belong to the
MIQP. The config refuses each with ``backend="exact"``. Above
``CANDIDATE_LIMIT`` candidates in a cluster, enumeration stops being affordable
and the search raises with the MIQP as the way out.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthEstimationError
from ..solvers.active_set import solve_simplex_qp
from ..syndes_helpers.enumeration import SearchSpace
from ..syndes_helpers.exact import SubsetFilter
from .formulation import (
    build_membership_mask,
    compute_cluster_means_members,
    get_per_cluster_param,
    per_unit_reproducibility,
    precompute_distances,
    prepare_clusters,
    prepare_fit_slices,
    validate_scm_inputs,
)
from .optimization import _augment_fit, finish_design

#: Designs whose objective splits into a treated term and a control term.
SEPARABLE_DESIGNS = frozenset({"standard", "penalized", "unit_penalized"})

#: Above this many candidates in one cluster the search refuses and names the MIQP.
CANDIDATE_LIMIT = 20_000_000

#: Candidates priced per batch.
CHUNK = 20_000

#: Best-bounded candidates kept from the first pass.
BUFFER = 512

#: A candidate is discarded only when its bound clears the cutoff by this much,
#: relative; it covers the floating-point error of the closed-form bound.
PRUNE_MARGIN = 1e-6


@dataclass(frozen=True)
class ClusterDesign:
    """One cluster's design: the treated set, its fits and its weights.

    Indices are local to the cluster: positions in the ``B`` the search was
    given.
    """

    value: float
    treated: Tuple[int, ...]
    control: Tuple[int, ...]
    w: np.ndarray
    v: np.ndarray


# ----------------------------------------------------------------------
# the oracle and the bound
# ----------------------------------------------------------------------


def side_fit(B: np.ndarray, A: np.ndarray, cols: Sequence[int],
             linear: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """``min ||A - B[:, cols] w||^2 + linear'w`` over the simplex, exactly.

    Returns the minimum and the weights, in the order of ``cols``.
    """
    cols = list(cols)
    w = solve_simplex_qp(B[:, cols], A, linear=linear)
    r = A - B[:, cols] @ w
    value = float(r @ r)
    if linear is not None:
        value += float(np.dot(linear, w))
    return value, w


def treated_fit_bound(B: np.ndarray, A: np.ndarray, subsets: np.ndarray,
                      linear: Optional[np.ndarray] = None) -> np.ndarray:
    """A lower bound on the treated-side fit of every row of ``subsets``.

    ``min { w'G w + a'w : 1'w = 1 }`` on the centred columns, in closed form
    (see the module docstring). A block whose solve does not reproduce its
    right-hand side to ``1e-8`` relative, or whose ``1'G^-1 1`` is not positive, gets
    the bound ``0``, which never discards anything; so does a negative value,
    since the fit itself is non-negative.
    """
    subsets = np.asarray(subsets, dtype=np.int64)
    n_sub, m = subsets.shape
    Bc = B - A[:, None]
    G = Bc.T @ Bc
    lin = None if linear is None else np.asarray(linear, dtype=float)
    out = np.zeros(n_sub)
    for lo in range(0, n_sub, CHUNK):
        idx = subsets[lo:lo + CHUNK]
        b = len(idx)
        Gs = G[idx[:, :, None], idx[:, None, :]]
        rhs = np.ones((b, m, 1 if lin is None else 2))
        if lin is not None:
            rhs[:, :, 1] = lin[idx]
        try:
            x = np.linalg.solve(Gs, rhs)
        except np.linalg.LinAlgError:
            x = np.full_like(rhs, np.nan)
            for i in range(b):
                try:
                    x[i] = np.linalg.solve(Gs[i], rhs[i])
                except np.linalg.LinAlgError:
                    pass
        resid = np.abs(Gs @ x - rhs).reshape(b, -1).max(axis=1)
        scale = np.maximum(1.0, np.abs(rhs).reshape(b, -1).max(axis=1))
        s = x[:, :, 0].sum(axis=1)
        good = (np.isfinite(s) & np.isfinite(resid) & (s > 0)
                & (resid <= 1e-8 * scale))
        val = np.zeros(b)
        if lin is None:
            val[good] = 1.0 / s[good]
        else:
            t = x[:, :, 1].sum(axis=1)
            u = np.einsum("bi,bi->b", rhs[:, :, 1], x[:, :, 1])
            val[good] = (2.0 + t[good]) ** 2 / (4.0 * s[good]) - u[good] / 4.0
        val[~np.isfinite(val)] = 0.0
        out[lo:lo + b] = np.maximum(val, 0.0)
    return out


# ----------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------


def _space(n: int, size: int, filt: SubsetFilter) -> SearchSpace:
    return SearchSpace.build(n, size, forced=filt.forced_in,
                             forbidden=filt.forbidden, strata=filt.strata)


def count_candidates(n: int, sizes: Sequence[int], filt: SubsetFilter) -> int:
    """Admissible treated sets over ``sizes``, before conflict pairs remove any."""
    return sum(_space(n, s, filt).size for s in sizes)


def _chunks(n: int, sizes: Sequence[int], filt: SubsetFilter
            ) -> Iterator[np.ndarray]:
    """Admissible treated sets as ``(b, size)`` arrays, ``b <= CHUNK``.

    The walk already honours forced units, forbidden units and disjoint strata.
    Conflict pairs, and strata the walk could not resolve into groups, are
    tested on each finished candidate; with none of them the candidates go
    straight into an array without a per-candidate call.
    """
    check = bool(filt.conflict_pairs or filt.strata or filt.forbidden_sets)
    for size in sizes:
        space = _space(n, size, filt)
        if not space.size:
            continue
        walk = space.iter_subsets()
        if not check:
            while True:
                flat = np.fromiter(
                    itertools.chain.from_iterable(itertools.islice(walk, CHUNK)),
                    dtype=np.int64)
                if not flat.size:
                    break
                yield flat.reshape(-1, size)
            continue
        block: List[Tuple[int, ...]] = []
        for subset in walk:
            if not filt.accepts(subset):
                continue
            block.append(subset)
            if len(block) == CHUNK:
                yield np.asarray(block, dtype=np.int64)
                block = []
        if block:
            yield np.asarray(block, dtype=np.int64)


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------


class _TopK:
    """The ``k`` best designs so far, and the cutoff they set."""

    def __init__(self, k: int):
        self.k = int(k)
        self.items: List[ClusterDesign] = []

    @property
    def cutoff(self) -> float:
        return self.items[-1].value if len(self.items) == self.k else np.inf

    def offer(self, design: ClusterDesign) -> None:
        if design.value >= self.cutoff:
            return
        self.items.append(design)
        self.items.sort(key=lambda d: (d.value, d.treated))
        del self.items[self.k:]


def search_cluster(
    B: np.ndarray,
    A: np.ndarray,
    sizes: Sequence[int],
    *,
    filt: SubsetFilter,
    linear_treated: Optional[np.ndarray] = None,
    linear_control: Optional[np.ndarray] = None,
    donor_exclusion: Sequence[Tuple[int, int]] = (),
    top_k: int = 1,
    candidate_limit: Optional[int] = None,
) -> List[ClusterDesign]:
    """The ``top_k`` best designs of one cluster, best first.

    Parameters
    ----------
    B : np.ndarray, shape (T, n)
        The cluster's markets as columns, over the fit window.
    A : np.ndarray, shape (T,)
        The cluster's target.
    sizes : sequence of int
        Treated-set sizes to search.
    filt : SubsetFilter
        Forced, forbidden, conflict and stratum rules, in local indices.
    linear_treated, linear_control : np.ndarray, optional
        Non-negative penalty vectors over the cluster's markets.
    donor_exclusion : sequence of (i, j)
        ``j`` may not be a control when ``i`` is treated.
    top_k : int
        How many designs to return.
    candidate_limit : int, optional
        Refuse above this many candidates; defaults to ``CANDIDATE_LIMIT``.

    Raises
    ------
    MlsynthConfigError
        Above the candidate limit.
    MlsynthEstimationError
        When no admissible design exists.
    """
    n = B.shape[1]
    limit = CANDIDATE_LIMIT if candidate_limit is None else int(candidate_limit)
    total = count_candidates(n, sizes, filt)
    if total > limit:
        raise MlsynthConfigError(
            f"The exact backend would enumerate {total:,} treated sets in one "
            f"cluster, above its limit of {limit:,}. Use backend='mip', or "
            f"narrow the design with restrictions or a smaller treated count.")

    excluded: Dict[int, set] = {}
    for i, j in donor_exclusion:
        excluded.setdefault(int(i), set()).add(int(j))
    lin_t = None if linear_treated is None else np.asarray(linear_treated, float)
    lin_c = None if linear_control is None else np.asarray(linear_control, float)

    best = _TopK(top_k)
    seen: set = set()

    def margin(cut: float) -> float:
        return PRUNE_MARGIN * (1.0 + abs(cut))

    def consider(S: Tuple[int, ...]) -> None:
        seen.add(S)
        cols = list(S)
        g, w = side_fit(B, A, cols, None if lin_t is None else lin_t[cols])
        if g >= best.cutoff:
            return
        drop = set(S)
        for i in S:
            drop |= excluded.get(i, set())
        control = [j for j in range(n) if j not in drop]
        if not control:
            return
        h, v = side_fit(B, A, control, None if lin_c is None else lin_c[control])
        best.offer(ClusterDesign(value=g + h, treated=tuple(S),
                                 control=tuple(control), w=w, v=v))

    # Pass 1: price every candidate, keep the BUFFER best-bounded.
    buf_lb = np.empty(0)
    buf_sets: List[Tuple[int, ...]] = []
    for block in _chunks(n, sizes, filt):
        lb = treated_fit_bound(B, A, block, lin_t)
        held = buf_lb.size
        all_lb = np.concatenate([buf_lb, lb])
        if all_lb.size > BUFFER:
            part = np.argpartition(all_lb, BUFFER - 1)[:BUFFER]
        else:
            part = np.arange(all_lb.size)
        buf_sets = [buf_sets[i] if i < held
                    else tuple(int(j) for j in block[i - held]) for i in part]
        buf_lb = all_lb[part]
    if not buf_sets:
        raise MlsynthEstimationError(
            "No admissible treated set exists in this cluster under the "
            "cardinality and the active restrictions.")
    for i in np.argsort(buf_lb, kind="stable"):
        cut = best.cutoff
        if buf_lb[i] >= cut + margin(cut):
            break
        consider(buf_sets[i])

    # Pass 2, only when the buffer cannot certify the result: it was full, and
    # either fewer than top_k designs were found or its largest bound is still
    # under the cutoff, so a candidate outside it could still enter.
    cut = best.cutoff
    if len(buf_sets) == BUFFER and (
            not np.isfinite(cut) or buf_lb.max() < cut + margin(cut)):
        for block in _chunks(n, sizes, filt):
            lb = treated_fit_bound(B, A, block, lin_t)
            for i in np.argsort(lb, kind="stable"):
                cut = best.cutoff
                if lb[i] >= cut + margin(cut):
                    break
                S = tuple(int(j) for j in block[i])
                if S not in seen:
                    consider(S)

    if not best.items:
        raise MlsynthEstimationError(
            "No admissible design exists in this cluster: every treated set "
            "the restrictions allow leaves no control market.")
    return list(best.items)


# ----------------------------------------------------------------------
# MAREX's program, solved by the search
# ----------------------------------------------------------------------


def _local_rules(restrictions: Any, members: np.ndarray, cluster_of: np.ndarray
                 ) -> Tuple[SubsetFilter, Tuple[Tuple[int, int], ...]]:
    """One cluster's restrictions in local indices, and its donor exclusions."""
    pos = {int(g): i for i, g in enumerate(members)}
    if restrictions is None or restrictions.is_empty:
        return SubsetFilter(), ()
    forced = frozenset(pos[j] for j in restrictions.forced_in if j in pos)
    forbidden = frozenset(pos[j] for j in restrictions.forbidden if j in pos)
    conflicts = tuple((pos[i], pos[j]) for i, j in restrictions.conflict_pairs
                      if i in pos and j in pos)
    strata = tuple((tuple(pos[s] for s in mem if s in pos), lo, hi)
                   for mem, lo, hi in restrictions.strata
                   if any(s in pos for s in mem))
    donors = tuple((pos[i], pos[j]) for i, j in restrictions.donor_exclusion
                   if i in pos and j in pos)
    filt = SubsetFilter(forced_in=forced, forbidden=forbidden,
                        conflict_pairs=conflicts, strata=strata)
    return filt, donors


def _refuse_spanning_rules(restrictions: Any, cluster_of: np.ndarray) -> None:
    """Clusters are searched apart, so no rule may tie two of them together."""
    if restrictions is None or restrictions.is_empty:
        return
    for i, j in restrictions.conflict_pairs:
        if cluster_of[i] != cluster_of[j]:
            raise MlsynthConfigError(
                "A border conflict links markets in different clusters. The "
                "exact backend searches each cluster on its own, so it cannot "
                "honour a rule that spans two of them; use backend='mip'.")
    for members, lo, hi in restrictions.strata:
        if len({int(cluster_of[s]) for s in members}) > 1 and (
                lo is not None or hi is not None):
            raise MlsynthConfigError(
                "A stratum quota counts markets in more than one cluster. The "
                "exact backend searches each cluster on its own, so it cannot "
                "honour a quota that spans two of them; use backend='mip'.")


def _sizes(n: int, lab: Any, m_eq, m_min, m_max) -> List[int]:
    me = get_per_cluster_param(m_eq, lab)
    if me is not None:
        return [int(me)] if 1 <= int(me) <= n - 1 else []
    mn = get_per_cluster_param(m_min, lab)
    mx = get_per_cluster_param(m_max, lab)
    lo = max(1, int(mn) if mn is not None else 1)
    hi = min(n - 1, int(mx) if mx is not None else n - 1)
    return list(range(lo, hi + 1))


def _penalties(design: str, k_idx: int, members: np.ndarray, X_fit: np.ndarray,
               D1: np.ndarray, lambda1: float, lambda2: float, xi: float,
               lambda1_unit: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The linear terms ``build_objective`` adds, restricted to one cluster."""
    if design == "penalized":
        a = lambda1 * D1[members, k_idx] if lambda1 > 0 else None
        b = lambda2 * D1[members, k_idx] if lambda2 > 0 else None
        return a, b
    if design == "unit_penalized":
        a = np.zeros(len(members))
        if xi > 0:
            a += xi * per_unit_reproducibility(X_fit, members)[members]
        if lambda1_unit > 0:
            a += lambda1_unit * D1[members, k_idx]
        return (a if (xi > 0 or lambda1_unit > 0) else None), None
    return None, None


def _k_best_combinations(lists: List[List[ClusterDesign]], k: int
                         ) -> List[Tuple[float, Tuple[int, ...]]]:
    """The ``k`` smallest sums choosing one design per cluster, best first."""
    start = tuple(0 for _ in lists)
    heap = [(sum(lst[0].value for lst in lists), start)]
    seen = {start}
    out = []
    while heap and len(out) < k:
        total, pick = heapq.heappop(heap)
        out.append((total, pick))
        for c, lst in enumerate(lists):
            if pick[c] + 1 < len(lst):
                nxt = pick[:c] + (pick[c] + 1,) + pick[c + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (total - lst[pick[c]].value
                                          + lst[pick[c] + 1].value, nxt))
    return out


def _designs(
    Y_full, T0, clusters, blank_periods=0, m_eq=None, m_min=None, m_max=None,
    exclusive=True, design="standard", beta=1e-6, lambda1=0.0, lambda2=0.0,
    xi=0.0, lambda1_unit=0.0, lambda2_unit=0.0, costs=None, budget=None,
    covariates=None, covariate_weight=1.0, standardize=False,
    restrictions=None, top_k=1,
) -> List[Dict[str, Any]]:
    if design not in SEPARABLE_DESIGNS:
        raise MlsynthConfigError(
            f"The exact backend solves the separable designs "
            f"({', '.join(sorted(SEPARABLE_DESIGNS))}); got design={design!r}.")
    if costs is not None or budget is not None:
        raise MlsynthConfigError(
            "The exact backend does not take a cost budget, which constrains "
            "the treated weights; use backend='mip'.")
    if design == "unit_penalized" and lambda2_unit > 0:
        raise MlsynthConfigError(
            "lambda2_unit pairs the treated weights against the control weights "
            "and leaves no convex program to solve; use lambda1_unit.")
    validate_scm_inputs(Y_full, T0, blank_periods, design, beta, lambda1,
                        lambda2, xi, lambda1_unit, lambda2_unit)
    Y_full_np, clusters, N, cluster_labels, K, label_to_k = prepare_clusters(
        Y_full, clusters)
    Y_fit, Y_blank, T_fit = prepare_fit_slices(Y_full_np, T0, blank_periods)
    X_fit = _augment_fit(Y_fit, covariates, covariate_weight, standardize)
    M = build_membership_mask(clusters, label_to_k, N, K)
    Xbar_clusters, cluster_members = compute_cluster_means_members(
        X_fit, M, cluster_labels)
    D1, _ = precompute_distances(X_fit, Xbar_clusters, cluster_members)
    cluster_of = np.array([int(np.argmax(M[j])) for j in range(N)])
    _refuse_spanning_rules(restrictions, cluster_of)

    per_cluster: List[List[ClusterDesign]] = []
    for k_idx, lab in enumerate(cluster_labels):
        members = np.asarray(cluster_members[k_idx])
        sizes = _sizes(len(members), lab, m_eq, m_min, m_max)
        if not sizes:
            raise MlsynthEstimationError(
                f"MAREX design is infeasible in cluster {lab!r}: the treated "
                f"count leaves no control market (m_eq={m_eq}, m_min={m_min}, "
                f"m_max={m_max}, cluster size {len(members)}).")
        filt, donors = _local_rules(restrictions, members, cluster_of)
        a, b = _penalties(design, k_idx, members, X_fit, D1, lambda1, lambda2,
                          xi, lambda1_unit)
        try:
            designs = search_cluster(
                X_fit[members].T, Xbar_clusters[k_idx], sizes, filt=filt,
                linear_treated=a, linear_control=b, donor_exclusion=donors,
                top_k=top_k)
        except MlsynthEstimationError as exc:
            raise MlsynthEstimationError(
                f"MAREX design is infeasible under the active restrictions "
                f"(forced/forbidden units, border conflicts, size band) and the "
                f"cluster cardinality (m_min/m_max) in cluster {lab!r}: {exc}"
            ) from exc
        per_cluster.append(designs)

    raws = []
    for total, pick in _k_best_combinations(per_cluster, top_k):
        w_opt = np.zeros((N, K))
        v_opt = np.zeros((N, K))
        z_opt = np.zeros((N, K))
        for k_idx, choice in enumerate(pick):
            d = per_cluster[k_idx][choice]
            members = np.asarray(cluster_members[k_idx])
            treated = members[list(d.treated)]
            control = members[list(d.control)]
            w_opt[treated, k_idx] = d.w
            v_opt[control, k_idx] = d.v
            z_opt[treated, k_idx] = 1.0
        raws.append(finish_design(
            Y_full, Y_full_np, w_opt, v_opt, z_opt, Xbar_clusters,
            cluster_labels, cluster_members, N, K, T0, blank_periods, T_fit,
            Y_fit, Y_blank, design, clusters, float(total)))
    return raws


def solve_design_exact(Y_full, T0, clusters, **kwargs) -> Dict[str, Any]:
    """MAREX's design by bounded search; the same dict as ``solve_design``.

    Takes ``solve_design``'s arguments less the branch-and-bound knobs
    (``solver``, ``verbose``, ``warm_start``, ``time_limit``, ``forbidden``).
    """
    return _designs(Y_full, T0, clusters, top_k=1, **kwargs)[0]


def solve_design_exact_pool(Y_full, T0, clusters, *, top_K=1, **kwargs
                            ) -> List[Dict[str, Any]]:
    """The ``top_K`` best distinct designs, best first; ``solve_design_pool``'s output.

    The MIQP pool re-solves with a no-good cut per design already found. Here
    each cluster's search keeps its ``top_K`` best treated sets, and the global
    designs are the ``top_K`` smallest sums of one choice per cluster.
    """
    return _designs(Y_full, T0, clusters, top_k=int(top_K), **kwargs)
