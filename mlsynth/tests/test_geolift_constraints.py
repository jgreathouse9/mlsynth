"""Design-constraint primitives for GeoLift market selection.

Self-contained constraint layer (GeoLift owns it; LEXSCM is untouched). These
are pure, label-based helpers that decide which candidate test regions are
admissible (treatment criteria: cluster non-interference, coverage quotas, size
bands) and which donors a candidate may use (control criterion: spillover
exclusion). Tested test-first per the repo contract.
"""

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.geolift_helpers.marketselect.helpers.constraints import (
    build_conflict_graph,
    conflict_neighbors,
    is_independent_set,
    eligible_by_size,
    satisfies_quota,
    admissible_candidates,
)


# === build_conflict_graph ===

def test_conflict_graph_from_clusters():
    units = ["a", "b", "c", "d"]
    cluster_map = {"a": "NE", "b": "NE", "c": "S", "d": "S"}
    g = build_conflict_graph(units, cluster_map=cluster_map)
    assert g["a"] == frozenset({"b"})           # same cluster -> conflict
    assert g["c"] == frozenset({"d"})
    assert "a" not in g["c"]                     # different cluster -> no conflict
    assert "a" not in g["a"]                     # never self


def test_conflict_graph_singleton_cluster_has_no_conflicts():
    units = ["a", "b", "c"]
    g = build_conflict_graph(units, cluster_map={"a": "X", "b": "Y", "c": "Z"})
    assert all(g[u] == frozenset() for u in units)


def test_conflict_graph_missing_cluster_label_no_conflict():
    """A unit absent from the cluster map simply has no cluster conflicts."""
    units = ["a", "b", "c"]
    g = build_conflict_graph(units, cluster_map={"a": "X", "b": "X"})  # c missing
    assert g["a"] == frozenset({"b"})
    assert g["c"] == frozenset()


def test_conflict_graph_from_adjacency_threshold_and_symmetry():
    units = ["a", "b", "c"]
    adj = pd.DataFrame(
        [[0.0, 0.9, 0.1],
         [0.9, 0.0, 0.4],
         [0.1, 0.4, 0.0]],
        index=units, columns=units,
    )
    g = build_conflict_graph(units, adjacency=adj, spillover_threshold=0.5)
    assert g["a"] == frozenset({"b"})           # 0.9 > 0.5
    assert g["b"] == frozenset({"a"})           # symmetric
    assert g["c"] == frozenset()                # 0.1, 0.4 both <= 0.5


def test_conflict_graph_clusters_and_adjacency_combine_by_or():
    units = ["a", "b", "c"]
    cluster_map = {"a": "X", "b": "Y", "c": "X"}   # a~c by cluster
    adj = pd.DataFrame(
        [[0.0, 0.8, 0.0],
         [0.8, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        index=units, columns=units,
    )                                              # a~b by adjacency
    g = build_conflict_graph(units, cluster_map=cluster_map, adjacency=adj,
                             spillover_threshold=0.5)
    assert g["a"] == frozenset({"b", "c"})         # union of both sources


def test_conflict_graph_no_inputs_is_empty():
    units = ["a", "b"]
    g = build_conflict_graph(units)
    assert g == {"a": frozenset(), "b": frozenset()}


def test_conflict_graph_adjacency_must_cover_units():
    units = ["a", "b", "c"]
    adj = pd.DataFrame([[0.0, 0.9], [0.9, 0.0]], index=["a", "b"], columns=["a", "b"])
    with pytest.raises(MlsynthConfigError, match="adjacency"):
        build_conflict_graph(units, adjacency=adj)


# === is_independent_set / conflict_neighbors ===

def _graph():
    units = ["a", "b", "c", "d"]
    # a~b (cluster NE), c~d (cluster S)
    return build_conflict_graph(units, cluster_map={"a": "NE", "b": "NE",
                                                    "c": "S", "d": "S"})


def test_is_independent_set_true_when_conflict_free():
    g = _graph()
    assert is_independent_set(frozenset({"a", "c"}), g)      # different clusters
    assert is_independent_set(frozenset({"a"}), g)           # singleton
    assert is_independent_set(frozenset(), g)                # empty


def test_is_independent_set_false_when_members_conflict():
    g = _graph()
    assert not is_independent_set(frozenset({"a", "b"}), g)  # same cluster


def test_conflict_neighbors_union_minus_members():
    g = _graph()
    # neighbours of {a} are {b}; exclude the treated members themselves
    assert conflict_neighbors(frozenset({"a"}), g) == frozenset({"b"})
    assert conflict_neighbors(frozenset({"a", "c"}), g) == frozenset({"b", "d"})


def test_conflict_neighbors_excludes_treated_members():
    g = _graph()
    # b is a neighbour of a, but b is itself treated -> not in A(S)
    assert conflict_neighbors(frozenset({"a", "b"}), g) == frozenset()


def test_conflict_neighbors_empty_graph():
    g = build_conflict_graph(["a", "b", "c"])
    assert conflict_neighbors(frozenset({"a"}), g) == frozenset()


# === eligible_by_size ===

def test_eligible_by_size_both_bounds_inclusive():
    size_map = {"a": 10, "b": 50, "c": 100, "d": 200}
    elig = eligible_by_size(size_map.keys(), size_map, min_size=50, max_size=100)
    assert elig == frozenset({"b", "c"})            # bounds inclusive


def test_eligible_by_size_min_only_and_max_only():
    size_map = {"a": 10, "b": 50, "c": 100}
    assert eligible_by_size(size_map, size_map, min_size=50) == frozenset({"b", "c"})
    assert eligible_by_size(size_map, size_map, max_size=50) == frozenset({"a", "b"})


def test_eligible_by_size_no_bounds_is_all():
    size_map = {"a": 10, "b": 50}
    assert eligible_by_size(size_map, size_map) == frozenset({"a", "b"})


def test_eligible_by_size_unit_missing_size_is_ineligible():
    size_map = {"a": 10, "b": 50}
    assert eligible_by_size(["a", "b", "c"], size_map, min_size=0) == frozenset({"a", "b"})


# === satisfies_quota ===

def _strata():
    return {"a": "NE", "b": "NE", "c": "S", "d": "W"}


def test_satisfies_quota_max_per_stratum():
    sm = _strata()
    # at most 1 per stratum: {a,b} both NE -> violates
    assert not satisfies_quota(frozenset({"a", "b"}), sm, max_per_stratum=1)
    assert satisfies_quota(frozenset({"a", "c"}), sm, max_per_stratum=1)


def test_satisfies_quota_min_per_required_stratum():
    sm = _strata()
    required = {"NE", "S", "W"}
    # must cover every required stratum at least once
    assert not satisfies_quota(frozenset({"a", "c"}), sm, min_per_stratum=1,
                               required_strata=required)          # misses W
    assert satisfies_quota(frozenset({"a", "c", "d"}), sm, min_per_stratum=1,
                           required_strata=required)


def test_satisfies_quota_both_bounds():
    sm = _strata()
    required = {"NE", "S"}
    ok = frozenset({"a", "c"})          # 1 NE, 1 S
    assert satisfies_quota(ok, sm, min_per_stratum=1, max_per_stratum=1,
                           required_strata=required)
    bad = frozenset({"a", "b", "c"})    # 2 NE > max
    assert not satisfies_quota(bad, sm, min_per_stratum=1, max_per_stratum=1,
                               required_strata=required)


def test_satisfies_quota_no_quota_is_true():
    sm = _strata()
    assert satisfies_quota(frozenset({"a", "b", "c", "d"}), sm)


def test_satisfies_quota_ignores_market_without_stratum():
    """A treated market absent from the stratum map is not counted in any
    stratum (it contributes to no quota)."""
    sm = {"a": "NE", "b": "NE"}                  # c, d have no stratum
    assert satisfies_quota(frozenset({"a", "c", "d"}), sm, max_per_stratum=1)
    assert not satisfies_quota(frozenset({"a", "b", "c"}), sm, max_per_stratum=1)


# === admissible_candidates (composition) ===

def test_admissible_candidates_filters_conflicts_and_quota():
    g = build_conflict_graph(["a", "b", "c", "d"],
                             cluster_map={"a": "NE", "b": "NE", "c": "S", "d": "W"})
    sm = {"a": "NE", "b": "NE", "c": "S", "d": "W"}
    cands = [
        frozenset({"a", "b"}),      # conflict (both NE) -> drop
        frozenset({"a", "c"}),      # ok
        frozenset({"c", "d"}),      # ok
    ]
    out = admissible_candidates(cands, conflict=g, stratum_map=sm, max_per_stratum=1)
    assert frozenset({"a", "b"}) not in out
    assert frozenset({"a", "c"}) in out and frozenset({"c", "d"}) in out


def test_admissible_candidates_drops_on_quota_alone():
    """A conflict-free candidate that violates the quota is still dropped."""
    sm = {"a": "NE", "b": "NE", "c": "S"}
    cands = [frozenset({"a", "b"}), frozenset({"a", "c"})]
    out = admissible_candidates(cands, stratum_map=sm, max_per_stratum=1)
    assert out == [frozenset({"a", "c"})]       # {a,b} both NE -> dropped


def test_admissible_candidates_no_constraints_returns_all():
    cands = [frozenset({"a", "b"}), frozenset({"c"})]
    assert admissible_candidates(cands) == cands
