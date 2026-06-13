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

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.geolift_helpers.marketselect.helpers.constraints import (
    build_conflict_graph,
    conflict_neighbors,
    is_independent_set,
    eligible_by_size,
    satisfies_quota,
    admissible_candidates,
    unit_attribute_map,
)
from mlsynth.utils.geolift_helpers.marketselect.helpers.shaping import donor_matrix
from mlsynth.utils.geolift_helpers.config import GeoLiftConfig
from mlsynth.utils.geolift_helpers.marketselect.orchestration import run_design


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


# === unit_attribute_map ===

def test_unit_attribute_map_constant_per_unit():
    df = pd.DataFrame({"unit": ["a", "a", "b", "b"], "t": [0, 1, 0, 1],
                       "g": ["X", "X", "Y", "Y"]})
    assert unit_attribute_map(df, "unit", "g") == {"a": "X", "b": "Y"}


def test_unit_attribute_map_varying_raises():
    df = pd.DataFrame({"unit": ["a", "a"], "t": [0, 1], "g": ["X", "Y"]})
    with pytest.raises(MlsynthDataError, match="constant per unit"):
        unit_attribute_map(df, "unit", "g")


def test_unit_attribute_map_missing_col_raises():
    df = pd.DataFrame({"unit": ["a"], "t": [0]})
    with pytest.raises(MlsynthConfigError, match="nope"):
        unit_attribute_map(df, "unit", "nope")


# === donor_matrix spillover exclusion ===

def _wide(units, T=20, seed=1):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    return pd.DataFrame({u: base + rng.normal(scale=1.0, size=T) + i
                         for i, u in enumerate(units)})


def test_donor_matrix_exclude_drops_extra_units():
    Yw = _wide(["a", "b", "c", "d"])
    dm = donor_matrix(Yw, frozenset({"a"}), exclude={"b"})
    assert list(dm.columns) == ["c", "d"]            # candidate a + excluded b gone


def test_donor_matrix_exclude_none_is_unchanged():
    Yw = _wide(["a", "b", "c", "d"])
    assert list(donor_matrix(Yw, frozenset({"a"})).columns) == ["b", "c", "d"]


def test_donor_matrix_exclude_emptying_pool_raises():
    Yw = _wide(["a", "b"])
    with pytest.raises(MlsynthDataError):
        donor_matrix(Yw, frozenset({"a"}), exclude={"b"})


# === orchestration: cluster / spillover wiring (end-to-end) ===

def _clustered_long(seed=3, T=24):
    """6 units in 3 clusters of 2: u0,u1 -> C0; u2,u3 -> C1; u4,u5 -> C2."""
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    rows = []
    for i in range(6):
        s = base + rng.normal(scale=1.0, size=T) + i
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(s[t]),
                         "cluster": f"C{i // 2}"})
    return pd.DataFrame(rows)


_CLUSTER_OF = {f"u{i}": f"C{i // 2}" for i in range(6)}


def test_run_design_cluster_constraint_yields_independent_sets():
    cfg = GeoLiftConfig(df=_clustered_long(), outcome="Y", unitid="unit", time="time",
                        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                        lookback_window=1, ns=20, seed=0, cluster_col="cluster")
    res = run_design(cfg)
    assert len(res.search.candidates) >= 1
    for cd in res.search.candidates:
        cl = [_CLUSTER_OF[str(u)] for u in cd.candidate]
        assert len(cl) == len(set(cl))               # never two from one cluster


def test_run_design_spillover_excludes_same_cluster_donors():
    cfg = GeoLiftConfig(df=_clustered_long(), outcome="Y", unitid="unit", time="time",
                        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                        lookback_window=1, ns=20, seed=0, cluster_col="cluster")
    res = run_design(cfg)
    for cd in res.search.candidates:
        treated_clusters = {_CLUSTER_OF[str(u)] for u in cd.candidate}
        for donor in cd.weights.donor_weights:       # nonzero donors only
            assert _CLUSTER_OF[str(donor)] not in treated_clusters


def test_run_design_infeasible_cluster_raises():
    """Treat 4 with one-per-cluster but only 3 clusters -> no admissible region."""
    with pytest.raises(MlsynthConfigError, match="constraint"):
        run_design(GeoLiftConfig(
            df=_clustered_long(), outcome="Y", unitid="unit", time="time",
            treatment_size=4, durations=[4], effect_sizes=[0.0],
            lookback_window=1, ns=20, seed=0, cluster_col="cluster"))


def test_run_design_adjacency_excludes_spillover_donor():
    """A thresholded adjacency edge u0~u1 forbids them together and drops u1
    from u0's donor pool."""
    units = [f"u{i}" for i in range(6)]
    adj = pd.DataFrame(0.0, index=units, columns=units)
    adj.loc["u0", "u1"] = adj.loc["u1", "u0"] = 0.9   # only this pair conflicts
    cfg = GeoLiftConfig(df=_clustered_long(), outcome="Y", unitid="unit", time="time",
                        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                        lookback_window=1, ns=20, seed=0,
                        adjacency=adj, spillover_threshold=0.5)
    res = run_design(cfg)
    cands = {cd.candidate for cd in res.search.candidates}
    assert frozenset({"u0", "u1"}) not in cands       # adjacency forbids the pair
    for cd in res.search.candidates:
        if "u0" in cd.candidate:
            assert "u1" not in cd.weights.donor_weights   # u1 excluded as donor


def test_run_design_no_cluster_col_unchanged():
    """Without constraint fields the design is identical to the baseline run."""
    common = dict(outcome="Y", unitid="unit", time="time", treatment_size=2,
                  durations=[4], effect_sizes=[0.0, 0.5], lookback_window=1,
                  ns=20, seed=0)
    df = _clustered_long()
    base = run_design(GeoLiftConfig(df=df, **common))
    again = run_design(GeoLiftConfig(df=df, **common))
    assert {cd.candidate for cd in base.search.candidates} == \
           {cd.candidate for cd in again.search.candidates}


# === orchestration: coverage quotas + size bands (end-to-end) ===

_REGION_OF = {"u0": "R1", "u1": "R1", "u2": "R2", "u3": "R2", "u4": "R3", "u5": "R3"}
_SIZE_OF = {"u0": 10.0, "u1": 50.0, "u2": 60.0, "u3": 70.0, "u4": 500.0, "u5": 80.0}


def _stratified_long(seed=5, T=24):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    rows = []
    for i in range(6):
        u = f"u{i}"
        s = base + rng.normal(scale=1.0, size=T) + i
        for t in range(T):
            rows.append({"unit": u, "time": t, "Y": float(s[t]),
                         "region": _REGION_OF[u], "size": _SIZE_OF[u]})
    return pd.DataFrame(rows)


def test_run_design_size_band_restricts_treatment_not_donors():
    cfg = GeoLiftConfig(df=_stratified_long(), outcome="Y", unitid="unit", time="time",
                        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                        lookback_window=1, ns=20, seed=0,
                        size_col="size", min_size=40.0, max_size=100.0)
    res = run_design(cfg)
    treated_ever = set()
    for cd in res.search.candidates:
        treated_ever |= {str(u) for u in cd.candidate}
    assert "u0" not in treated_ever and "u4" not in treated_ever   # out of band
    # but the out-of-band markets remain available as donors somewhere
    donors_seen = set()
    for cd in res.search.candidates:
        donors_seen |= {str(d) for d in cd.weights.donor_weights}
    assert "u0" in donors_seen or "u4" in donors_seen


def test_run_design_max_per_stratum():
    cfg = GeoLiftConfig(df=_stratified_long(), outcome="Y", unitid="unit", time="time",
                        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                        lookback_window=1, ns=20, seed=0,
                        stratum_col="region", max_per_stratum=1)
    res = run_design(cfg)
    for cd in res.search.candidates:
        regions = [_REGION_OF[str(u)] for u in cd.candidate]
        assert len(regions) == len(set(regions))         # never two from one region


def test_run_design_min_per_stratum_covers_every_region():
    cfg = GeoLiftConfig(df=_stratified_long(), outcome="Y", unitid="unit", time="time",
                        treatment_size=3, durations=[4], effect_sizes=[0.0, 0.5],
                        lookback_window=1, ns=20, seed=0,
                        stratum_col="region", min_per_stratum=1)
    res = run_design(cfg)
    assert len(res.search.candidates) >= 1
    for cd in res.search.candidates:
        assert {_REGION_OF[str(u)] for u in cd.candidate} == {"R1", "R2", "R3"}


def test_run_design_min_per_stratum_infeasible_raises():
    """Cover 3 regions but only treat 2 -> impossible -> reported."""
    with pytest.raises(MlsynthConfigError, match="constraint"):
        run_design(GeoLiftConfig(
            df=_stratified_long(), outcome="Y", unitid="unit", time="time",
            treatment_size=2, durations=[4], effect_sizes=[0.0],
            lookback_window=1, ns=20, seed=0,
            stratum_col="region", min_per_stratum=1))


def test_run_design_size_band_too_tight_raises():
    with pytest.raises(MlsynthConfigError, match="size"):
        run_design(GeoLiftConfig(
            df=_stratified_long(), outcome="Y", unitid="unit", time="time",
            treatment_size=2, durations=[4], effect_sizes=[0.0],
            lookback_window=1, ns=20, seed=0,
            size_col="size", min_size=1000.0))


# === config validation for the new constraint fields ===

@pytest.mark.parametrize("kwargs,msg", [
    ({"stratum_col": "region", "min_per_stratum": 0}, "min_per_stratum"),
    ({"stratum_col": "region", "max_per_stratum": 0}, "max_per_stratum"),
    ({"min_per_stratum": 1}, "stratum_col"),               # quota without stratum_col
    ({"max_per_stratum": 1}, "stratum_col"),
    ({"min_size": 5.0}, "size_col"),                       # band without size_col
    ({"max_size": 5.0}, "size_col"),
    ({"size_col": "size", "min_size": 100.0, "max_size": 10.0}, "max_size"),
])
def test_constraint_config_validation(kwargs, msg):
    base = dict(df=_stratified_long(), outcome="Y", unitid="unit", time="time",
                treatment_size=2, durations=[4], effect_sizes=[0.0])
    base.update(kwargs)
    with pytest.raises(MlsynthConfigError, match=msg):
        GeoLiftConfig(**base)
