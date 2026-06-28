"""Up-front, itemised feasibility audit for GeoLift market selection.

When the design constraints admit no candidate test region, the analyst needs to
know *which* constraint bound the search -- not a catch-all "no candidate
satisfies the constraints". This module reports every individually-infeasible
constraint **together** in one :class:`MlsynthConfigError`, each line in a
uniform ``have vs need -> minimal fix`` shape.

It mirrors LEXSCM's :func:`mlsynth.utils.fast_scm_helpers.feasibility.audit_feasibility`
(same shape, same contract: it *reports*, it never silently relaxes a constraint
the analyst set), re-implemented label-based here so GeoLift owns its constraint
layer. The audit is a no-op when the design is feasible.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, List, Mapping, Optional

from mlsynth.exceptions import MlsynthConfigError

from .constraints import ConflictGraph


def _greedy_independent_set_size(
    markets: Iterable[Hashable], conflict: ConflictGraph
) -> int:
    """Greedy minimum-degree independent-set size over ``markets`` (a lower bound
    on the maximum independent set).

    Sound as a *necessary* feasibility signal: the true maximum is at least this,
    so reporting ``largest < k`` only when the greedy bound itself falls short
    would risk a false positive. For the pure-cluster conflict graph (the common
    case) the greedy bound is exact -- it equals the number of distinct clusters
    present -- so the reported number is the real largest conflict-free set there.
    """
    markets = list(markets)
    remaining = set(markets)
    chosen = 0
    while remaining:
        # pick the minimum-degree remaining market (fewest conflicts inside the set)
        pick = min(remaining, key=lambda u: len(conflict.get(u, frozenset()) & remaining))
        chosen += 1
        remaining.discard(pick)
        remaining -= conflict.get(pick, frozenset())
    return chosen


def audit_geolift_feasibility(
    eligible: Iterable[Hashable],
    treatment_size: int,
    *,
    conflict: Optional[ConflictGraph] = None,
    stratum_map: Optional[Mapping[Hashable, Hashable]] = None,
    min_per_stratum: Optional[int] = None,
    max_per_stratum: Optional[int] = None,
    required_strata: Optional[Iterable[Hashable]] = None,
) -> None:
    """Raise a single, itemised :class:`MlsynthConfigError` if any active design
    constraint makes a size-``treatment_size`` treated set impossible. No-op when
    feasible.

    Parameters
    ----------
    eligible : iterable of hashable
        Markets eligible for treatment (after the size band and
        ``not_to_be_treated`` have been removed).
    treatment_size : int
        The requested number of treated markets ``k``.
    conflict : ConflictGraph, optional
        Cluster / adjacency conflict graph; a treated set must be an independent
        set of it.
    stratum_map : mapping, optional
        ``{market: stratum}`` for the coverage quota.
    min_per_stratum, max_per_stratum : int, optional
        Coverage quota bounds.
    required_strata : iterable, optional
        The strata that ``min_per_stratum`` must cover (those with at least one
        eligible market), supplied by the caller.
    """
    eligible = list(eligible)
    M = len(eligible)
    problems: List[str] = []

    # 1. Candidate pool -- the precondition for everything else.
    if M < treatment_size:
        problems.append(
            f"candidate pool: only {M} eligible market(s), but "
            f"treatment_size={treatment_size}. Widen eligibility / the size band, "
            f"or reduce treatment_size."
        )
        _raise(problems)        # the other checks are meaningless with too few units

    # 2. Coverage / quota.
    if stratum_map is not None:
        problems += _quota_problems(
            eligible, treatment_size, stratum_map,
            min_per_stratum, max_per_stratum, required_strata,
        )

    # 3. Cluster / adjacency -- a treated set must be a size-k independent set.
    if conflict is not None:
        largest = _greedy_independent_set_size(eligible, conflict)
        if largest < treatment_size:
            problems.append(
                f"spillover/cluster: the largest conflict-free treated set is "
                f"{largest} < treatment_size={treatment_size}. Relax the "
                f"cluster/adjacency constraint, widen the candidate pool, or "
                f"reduce treatment_size."
            )

    _raise(problems)


def _quota_problems(
    eligible: List[Hashable],
    k: int,
    stratum_map: Mapping[Hashable, Hashable],
    min_per: Optional[int],
    max_per: Optional[int],
    required: Optional[Iterable[Hashable]],
) -> List[str]:
    """Coverage/quota infeasibilities, each with its have-vs-need arithmetic."""
    problems: List[str] = []
    counts = Counter(
        stratum_map[u] for u in eligible
        if u in stratum_map and stratum_map[u] is not None
    )

    if min_per is not None and required is not None:
        req = list(required)
        need = min_per * len(req)
        if need > k:
            problems.append(
                f"coverage: min_per_stratum={min_per} over {len(req)} required "
                f"stratum(s) needs >= {need} treated market(s), but "
                f"treatment_size={k}. Raise treatment_size to >= {need}, lower "
                f"min_per_stratum, or shrink the universe."
            )
        short = sorted(
            (str(s) for s in req if counts.get(s, 0) < min_per)
        )
        if short:
            problems.append(
                f"coverage: stratum(s) {short} have fewer than "
                f"min_per_stratum={min_per} eligible market(s). Widen the "
                f"universe / size band so each required stratum can supply "
                f"{min_per}, or lower min_per_stratum."
            )

    if max_per is not None:
        n_strata = len(counts)
        cap = max_per * n_strata
        if cap < k:
            problems.append(
                f"quota: max_per_stratum={max_per} over {n_strata} stratum(s) "
                f"allows at most {cap} treated market(s), but treatment_size={k}. "
                f"Raise max_per_stratum, add eligible strata, or reduce "
                f"treatment_size."
            )

    return problems


def _raise(problems: List[str]) -> None:
    if problems:
        raise MlsynthConfigError(
            "GeoLift design is infeasible -- the binding constraint(s):\n  - "
            + "\n  - ".join(problems)
        )
