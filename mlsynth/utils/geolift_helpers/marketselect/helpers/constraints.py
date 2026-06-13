"""Design-constraint primitives for GeoLift market selection.

GeoLift's upstream surface restricts the treated side only through **hard market
lists** (``to_be_treated`` / ``not_to_be_treated``). This module adds the
*rule-based* constraint vocabulary -- cluster / spillover non-interference,
coverage quotas, and treated-unit size bands -- so the candidate nomination and
donor pool can encode geography and other design considerations.

Every constraint reduces to *where the optimizer may look*: it is either a
**treatment criterion** (a filter on which candidate test regions are
admissible -- independent set, quota, size band) or a **control criterion** (a
restriction on a candidate's donor pool -- spillover exclusion). None of them
touch the inner weight solve.

The logic mirrors LEXSCM's design constraints (see ``docs/lexscm.rst``), but is
**re-implemented here** so GeoLift owns its constraint layer; LEXSCM is not
modified. Cross-family consolidation into a shared module is a deliberate later
step. All helpers are pure and **label-based**: they operate on the ``Ywide``
market labels and the candidate :class:`frozenset`\\s directly, never on
positional indices.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Optional

import pandas as pd

from mlsynth.exceptions import MlsynthConfigError

# A conflict graph maps each market label to the frozenset of labels it
# conflicts with (interferes with). Symmetric, with no self-loops.
ConflictGraph = Dict[Hashable, frozenset]


def build_conflict_graph(
    units: Iterable[Hashable],
    *,
    cluster_map: Optional[Mapping[Hashable, Hashable]] = None,
    adjacency: Optional[pd.DataFrame] = None,
    spillover_threshold: float = 0.0,
) -> ConflictGraph:
    """Build the symmetric market conflict (interference) graph.

    Two markets *conflict* when treating both would create interference. The
    graph is assembled from either or both of:

    * ``cluster_map`` -- a ``{market: cluster}`` mapping; markets in the **same**
      cluster conflict (e.g. two geos in one DMA / state). A market absent from
      the map simply has no cluster conflicts.
    * ``adjacency`` -- a square :class:`~pandas.DataFrame` of pairwise spillover
      strengths indexed and columned by market label; markets ``i, j`` conflict
      when ``adjacency.loc[i, j] > spillover_threshold`` (the diagonal is
      ignored).

    When both are supplied the two conflict sets are combined by logical **OR**.
    With neither, every market's conflict set is empty (the unconstrained case).

    Parameters
    ----------
    units : iterable of hashable
        The market labels the graph is defined over (the ``Ywide`` columns).
    cluster_map : mapping, optional
        ``{market: cluster}``; same-cluster markets conflict.
    adjacency : pandas.DataFrame, optional
        Square spillover matrix; its index/columns must cover ``units``.
    spillover_threshold : float, default 0.0
        Off-diagonal entries strictly above this mark a conflict.

    Returns
    -------
    ConflictGraph
        ``{market: frozenset(conflicting markets)}`` for every market in
        ``units``, symmetric and self-free.

    Raises
    ------
    MlsynthConfigError
        If ``adjacency`` does not cover every unit on both axes.
    """
    units = list(units)
    neighbours: Dict[Hashable, set] = {u: set() for u in units}

    if cluster_map is not None:
        # Group units by cluster; all pairs within a (non-null) cluster conflict.
        by_cluster: Dict[Hashable, List[Hashable]] = {}
        for u in units:
            label = cluster_map.get(u)
            if label is None or (isinstance(label, float) and pd.isna(label)):
                continue
            by_cluster.setdefault(label, []).append(u)
        for members in by_cluster.values():
            for u in members:
                neighbours[u].update(m for m in members if m != u)

    if adjacency is not None:
        missing = [u for u in units
                   if u not in adjacency.index or u not in adjacency.columns]
        if missing:
            raise MlsynthConfigError(
                "adjacency matrix must cover every unit on both axes; missing "
                f"{sorted(map(str, missing))}."
            )
        sub = adjacency.reindex(index=units, columns=units)
        for u in units:
            row = sub.loc[u]
            for v in units:
                if u != v and float(row[v]) > spillover_threshold:
                    neighbours[u].add(v)
                    neighbours[v].add(u)         # enforce symmetry

    return {u: frozenset(neigh) for u, neigh in neighbours.items()}


def is_independent_set(markets: Iterable[Hashable], conflict: ConflictGraph) -> bool:
    """True iff no two ``markets`` conflict (the set is conflict-free).

    The Stage-1 "no interference" treatment criterion: a candidate test region
    must be an *independent set* of the conflict graph.
    """
    markets = list(markets)
    for i, u in enumerate(markets):
        neigh = conflict.get(u, frozenset())
        for v in markets[i + 1:]:
            if v in neigh:
                return False
    return True


def conflict_neighbors(
    markets: Iterable[Hashable], conflict: ConflictGraph
) -> frozenset:
    """The conflict-neighbours ``A(S)`` of a treated set, **excluding** ``S``.

    The Stage-2 "exclusion restriction" control criterion: these markets are a
    treated geo's spillover neighbours and must be dropped from its donor pool.
    """
    markets = frozenset(markets)
    out: set = set()
    for u in markets:
        out.update(conflict.get(u, frozenset()))
    return frozenset(out - markets)


def eligible_by_size(
    units: Iterable[Hashable],
    size_map: Mapping[Hashable, float],
    *,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
) -> frozenset:
    """The markets eligible for **treatment** under a size band ``[min, max]``.

    A treated-unit size band (both bounds inclusive): the floor is a power /
    operational minimum, the ceiling encodes synthesizability (a market far
    larger than the donors cannot sit inside their convex hull -- GeoLift's
    scaled-L2 imbalance would blow up). Markets outside the band stay available
    as **donors**; only their *treatment* eligibility is removed. A market with
    no recorded size is treated as ineligible.
    """
    out: set = set()
    for u in units:
        if u not in size_map:
            continue
        s = size_map[u]
        if min_size is not None and s < min_size:
            continue
        if max_size is not None and s > max_size:
            continue
        out.add(u)
    return frozenset(out)


def satisfies_quota(
    markets: Iterable[Hashable],
    stratum_map: Mapping[Hashable, Hashable],
    *,
    min_per_stratum: Optional[int] = None,
    max_per_stratum: Optional[int] = None,
    required_strata: Optional[Iterable[Hashable]] = None,
) -> bool:
    """True iff a candidate meets the coverage quota over strata.

    ``max_per_stratum`` caps the treated count in **any** stratum present;
    ``min_per_stratum`` requires at least that many treated markets in **every**
    ``required_stratum`` (the strata that contain at least one eligible market,
    supplied by the caller). With neither bound set, always ``True``.
    """
    markets = list(markets)
    counts: Dict[Hashable, int] = {}
    for u in markets:
        s = stratum_map.get(u)
        if s is None:
            continue
        counts[s] = counts.get(s, 0) + 1

    if max_per_stratum is not None:
        if any(c > max_per_stratum for c in counts.values()):
            return False

    if min_per_stratum is not None and required_strata is not None:
        for s in required_strata:
            if counts.get(s, 0) < min_per_stratum:
                return False

    return True


def admissible_candidates(
    candidates: List[frozenset],
    *,
    conflict: Optional[ConflictGraph] = None,
    stratum_map: Optional[Mapping[Hashable, Hashable]] = None,
    min_per_stratum: Optional[int] = None,
    max_per_stratum: Optional[int] = None,
    required_strata: Optional[Iterable[Hashable]] = None,
) -> List[frozenset]:
    """Filter candidate test regions to those satisfying the treatment criteria.

    Keeps a candidate iff it is an independent set of ``conflict`` (when given)
    **and** satisfies the stratum quota (when given). Order is preserved. This
    composes the Stage-1 treatment criteria; the size band is applied earlier as
    a restriction on the eligible nomination pool, and the spillover exclusion is
    a Stage-2 control criterion (:func:`conflict_neighbors`), so neither belongs
    here. Returns the (possibly empty) admissible list; the caller decides
    whether an empty result is an infeasibility to report.
    """
    has_quota = min_per_stratum is not None or max_per_stratum is not None
    out: List[frozenset] = []
    for cand in candidates:
        if conflict is not None and not is_independent_set(cand, conflict):
            continue
        if has_quota and stratum_map is not None and not satisfies_quota(
            cand, stratum_map, min_per_stratum=min_per_stratum,
            max_per_stratum=max_per_stratum, required_strata=required_strata,
        ):
            continue
        out.append(cand)
    return out
