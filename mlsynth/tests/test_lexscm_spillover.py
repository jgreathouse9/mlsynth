"""Exhaustive tests for LEXSCM spillover / interference exclusions.

Validates the two Vives-i-Bastida (2022) exclusion criteria end to end:

* **Stage 1 -- "No interference":** the selected treated ``m``-tuple is an
  *independent set* of the conflict graph (no two treated units in the same
  cluster / sharing a spillover edge).
* **Stage 2 -- "Exclusion restriction":** a treated unit's conflict neighbours
  ``N(S)`` are excluded from its donor pool.

Layers covered: the IndexSet-aligned conflict-graph builder, the graph
utilities, the Stage-1 search (exact enumeration *and* multi-start heuristic),
the Stage-2 control QP, feasibility errors on both stages, and the full LEXSCM
estimator for both specification surfaces (``cluster_col``, ``adjacency``) and
both unit-id types (``str``, ``int``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.lexscm import LEXSCM
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.fast_scm_helpers.conflict import (
    build_conflict_matrix,
    is_independent,
    max_independent_set_size_at_least,
    neighbours,
)
from mlsynth.utils.fast_scm_helpers.fast_scm_control_helpers import solve_control_qp
from mlsynth.utils.fast_scm_helpers.lexsearch import select_treated_designs


# =========================================================================
# Fixtures / helpers
# =========================================================================

# clusters for 15 units: 8 candidates paired into A,B,C,D; the 7 non-candidates Z
CLUSTERS = ["A", "A", "B", "B", "C", "C", "D", "D"] + ["Z"] * 7

BASE = dict(
    outcome="y", unitid="unitid", time="time", candidate_col="candidate",
    m=2, post_col="post", top_K=4, n_sims=30, verbose=False,
)


def make_panel(*, clusters=None, n_units=15, T=40, T_post=12, n_cand=8, L=2,
               seed=0, str_ids=False):
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal((n_units, L))
    nu = rng.standard_normal((T, L))
    Y = 100.0 + nu @ gamma.T + 0.1 * rng.standard_normal((T, n_units))
    rows = []
    for i in range(n_units):
        uid = f"u{i:02d}" if str_ids else i
        for t in range(T):
            r = {"unitid": uid, "time": t, "y": float(Y[t, i]),
                 "post": int(t >= T - T_post), "candidate": int(i < n_cand),
                 "cost": 1.0 + i}
            if clusters is not None:
                r["state"] = clusters[i]
            rows.append(r)
    return pd.DataFrame(rows)


def _small_gram(J=10, T=8, seed=0):
    rng = np.random.default_rng(seed)
    Xt = rng.normal(size=(T, J))
    return Xt.T @ Xt


# clique conflict for clusters 0-3=A, 4-6=B, 7-9=C (3 cliques)
def _clique_conflict(codes):
    codes = np.asarray(codes)
    A = (codes[:, None] == codes[None, :])
    np.fill_diagonal(A, False)
    return A


# =========================================================================
# Conflict-graph builder
# =========================================================================

class TestConflictBuilder:

    def test_cluster_is_clique_per_cluster(self):
        ui = IndexSet.from_labels([0, 1, 2, 3, 4])
        A = build_conflict_matrix(ui, cluster_of={0: "A", 1: "A", 2: "B", 3: "B", 4: "C"})
        assert A[0, 1] and A[1, 0] and A[2, 3]
        assert not A[0, 2] and not A[0, 4]
        assert not A[4].any()                      # singleton cluster conflicts with no one

    def test_diagonal_false_and_symmetric(self):
        ui = IndexSet.from_labels(list(range(6)))
        A = build_conflict_matrix(ui, cluster_of={i: i % 2 for i in range(6)})
        assert not np.any(np.diag(A))
        assert np.array_equal(A, A.T)

    def test_cluster_missing_value_raises(self):
        ui = IndexSet.from_labels([0, 1, 2])
        with pytest.raises(MlsynthDataError, match="no cluster value"):
            build_conflict_matrix(ui, cluster_of={0: "A", 1: "A"})   # 2 missing

    def test_cluster_nan_conflicts_with_no_one(self):
        ui = IndexSet.from_labels([0, 1, 2])
        A = build_conflict_matrix(ui, cluster_of={0: np.nan, 1: np.nan, 2: "B"})
        assert not A.any()                          # NaN never matches, even another NaN

    def test_adjacency_array(self):
        ui = IndexSet.from_labels([0, 1, 2, 3])
        M = np.zeros((4, 4)); M[0, 2] = 1.0
        A = build_conflict_matrix(ui, adjacency=M, spillover_threshold=0.5)
        assert A[0, 2] and A[2, 0]                   # symmetrised
        assert not A[0, 1]

    def test_adjacency_threshold(self):
        ui = IndexSet.from_labels([0, 1, 2])
        M = np.array([[0, 0.2, 0.9], [0.2, 0, 0], [0.9, 0, 0]])
        A = build_conflict_matrix(ui, adjacency=M, spillover_threshold=0.5)
        assert A[0, 2] and not A[0, 1]               # 0.2 < 0.5, 0.9 > 0.5

    def test_adjacency_dataframe_realigns_to_indexset(self):
        # DataFrame given in SCRAMBLED label order must be reindexed to the IndexSet.
        ui = IndexSet.from_labels([10, 20, 30])
        df = pd.DataFrame(0.0, index=[30, 10, 20], columns=[30, 10, 20])
        df.loc[10, 30] = 1.0; df.loc[30, 10] = 1.0
        A = build_conflict_matrix(ui, adjacency=df, spillover_threshold=0.5)
        # unit 10 is row 0, unit 30 is row 2 in the IndexSet
        assert A[0, 2] and A[2, 0] and not A[0, 1]

    def test_adjacency_wrong_shape_raises(self):
        ui = IndexSet.from_labels([0, 1, 2, 3])
        with pytest.raises(MlsynthConfigError, match="shape"):
            build_conflict_matrix(ui, adjacency=np.zeros((3, 3)))

    def test_adjacency_dataframe_missing_label_raises(self):
        ui = IndexSet.from_labels([0, 1, 2])
        df = pd.DataFrame(0.0, index=[0, 1], columns=[0, 1])   # missing unit 2
        with pytest.raises(MlsynthDataError, match="missing"):
            build_conflict_matrix(ui, adjacency=df)

    def test_adjacency_nonfinite_raises(self):
        ui = IndexSet.from_labels([0, 1])
        M = np.array([[0.0, np.inf], [np.inf, 0.0]])
        with pytest.raises(MlsynthDataError, match="finite"):
            build_conflict_matrix(ui, adjacency=M)

    def test_combined_cluster_and_adjacency_or(self):
        ui = IndexSet.from_labels([0, 1, 2, 3])
        M = np.zeros((4, 4)); M[0, 3] = 1.0
        A = build_conflict_matrix(ui, cluster_of={0: "A", 1: "A", 2: "B", 3: "B"},
                                  adjacency=M, spillover_threshold=0.5)
        assert A[0, 1]     # from clusters
        assert A[0, 3]     # from adjacency
        assert A[2, 3]     # from clusters

    def test_none_when_no_inputs(self):
        ui = IndexSet.from_labels([0, 1, 2])
        assert build_conflict_matrix(ui) is None


# =========================================================================
# Graph utilities
# =========================================================================

class TestGraphUtils:

    def test_neighbours_union_and_excludes_self(self):
        A = _clique_conflict([0, 0, 0, 1, 1, 2])     # {0,1,2}, {3,4}, {5}
        assert set(neighbours(A, [0]).tolist()) == {1, 2}
        assert set(neighbours(A, [0, 3]).tolist()) == {1, 2, 4}
        assert neighbours(A, [5]).size == 0          # isolated
        assert neighbours(A, []).size == 0

    def test_is_independent(self):
        A = _clique_conflict([0, 0, 1, 2])
        assert is_independent(None, [0, 1, 2])       # no graph -> always independent
        assert is_independent(A, [0])                # singleton
        assert is_independent(A, [0, 2, 3])          # one per clique
        assert not is_independent(A, [0, 1])         # same clique

    def test_feasibility_cluster_exact(self):
        A = _clique_conflict([0, 0, 0, 1, 1, 2])     # 3 cliques
        assert max_independent_set_size_at_least(A, range(6), 3)
        assert not max_independent_set_size_at_least(A, range(6), 4)

    def test_feasibility_none_graph(self):
        assert max_independent_set_size_at_least(None, range(6), 6)


# =========================================================================
# Stage 1 -- independent-set constraint in the search
# =========================================================================

class TestStage1:

    @pytest.mark.parametrize("method", ["enumerate", "heuristic"])
    def test_returns_only_independent_sets(self, method):
        G = _small_gram(10)
        A = _clique_conflict([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])   # cliques A,B,C
        res = select_treated_designs(G, list(range(10)), m=3, top_K=5,
                                     method=method, conflict=A, random_state=1)
        designs = res["top_designs"]
        assert len(designs) > 0
        for d in designs:
            assert is_independent(A, d.indices)

    @pytest.mark.parametrize("method", ["enumerate", "heuristic"])
    def test_empty_graph_equals_unconstrained(self, method):
        G = _small_gram(10)
        empty = np.zeros((10, 10), dtype=bool)
        r0 = select_treated_designs(G, list(range(10)), m=3, top_K=5,
                                    method=method, random_state=1)
        rA = select_treated_designs(G, list(range(10)), m=3, top_K=5,
                                    method=method, conflict=empty, random_state=1)
        assert r0["top_designs"][0].indices == rA["top_designs"][0].indices

    def test_infeasible_raises(self):
        G = _small_gram(10)
        A = _clique_conflict([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])   # only 3 cliques
        with pytest.raises(MlsynthConfigError, match="spillover|conflict-free"):
            select_treated_designs(G, list(range(10)), m=4, method="enumerate", conflict=A)

    def test_conflict_composes_with_budget(self):
        G = _small_gram(10)
        A = _clique_conflict([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])   # 5 cliques
        costs = np.arange(10, dtype=float)
        res = select_treated_designs(G, list(range(10)), m=2, top_K=5,
                                     method="enumerate", conflict=A,
                                     unit_costs=costs, budget=6.0)
        for d in res["top_designs"]:
            assert is_independent(A, d.indices)
            assert costs[d.indices].sum() <= 6.0 + 1e-9


# =========================================================================
# Stage 2 -- donor exclusion in the control QP
# =========================================================================

class TestStage2:

    def test_exclude_idx_zeros_neighbours(self):
        rng = np.random.default_rng(0)
        X_E = rng.normal(size=(8, 10))
        treated_idx = [0]
        treated_vec = X_E[:, 0]
        v = solve_control_qp(X_E, treated_vec, treated_idx, lambda_penalty=0.1,
                             exclude_idx=[1, 2])
        assert v is not None
        assert np.allclose(v[[0, 1, 2]], 0.0, atol=1e-8)     # treated + neighbours excluded
        assert v[3:].sum() > 0                                # remaining donors carry the weight

    def test_empty_donor_pool_returns_none(self):
        rng = np.random.default_rng(0)
        X_E = rng.normal(size=(8, 4))
        # exclude everything but one donor's complement -> exclude all donors
        v = solve_control_qp(X_E, X_E[:, 0], treated_idx=[0],
                             exclude_idx=[1, 2, 3])
        assert v is None

    def test_no_exclude_matches_baseline(self):
        rng = np.random.default_rng(1)
        X_E = rng.normal(size=(8, 6))
        v0 = solve_control_qp(X_E, X_E[:, 0], treated_idx=[0])
        v1 = solve_control_qp(X_E, X_E[:, 0], treated_idx=[0], exclude_idx=None)
        assert np.allclose(v0, v1)


# =========================================================================
# End-to-end LEXSCM
# =========================================================================

class TestLEXSCMSpilloverEndToEnd:

    @pytest.mark.parametrize("str_ids", [False, True])
    def test_cluster_treated_in_distinct_clusters(self, str_ids):
        df = make_panel(clusters=CLUSTERS, str_ids=str_ids)
        res = LEXSCM({"df": df, **BASE, "cluster_col": "state"}).fit()
        cmap = {(f"u{i:02d}" if str_ids else i): CLUSTERS[i] for i in range(15)}
        sel_clusters = [cmap[u if str_ids else int(u)] for u in res.selected_units]
        assert len(set(sel_clusters)) == len(sel_clusters)    # all distinct

    @pytest.mark.parametrize("str_ids", [False, True])
    def test_cluster_donors_exclude_treated_clusters(self, str_ids):
        df = make_panel(clusters=CLUSTERS, str_ids=str_ids)
        res = LEXSCM({"df": df, **BASE, "cluster_col": "state"}).fit()
        cmap = {(f"u{i:02d}" if str_ids else i): CLUSTERS[i] for i in range(15)}
        treated_clusters = {cmap[u if str_ids else int(u)] for u in res.selected_units}
        donor_clusters = {cmap[d if str_ids else int(d)]
                          for d in res.design_weights.donor_weights}
        assert treated_clusters.isdisjoint(donor_clusters)

    def test_adjacency_mode_respects_conflict(self):
        df = make_panel()
        adj = pd.DataFrame(0.0, index=range(15), columns=range(15))
        adj.loc[0, 1] = adj.loc[1, 0] = 1.0          # forbid co-treating 0 & 1
        res = LEXSCM({"df": df, **BASE, "adjacency": adj,
                      "spillover_threshold": 0.5}).fit()
        sel = {int(u) for u in res.selected_units}
        assert not ({0, 1} <= sel)

    def test_baseline_unchanged_without_spillover(self):
        # Empty conflict (each unit its own cluster) must reproduce the no-spillover pick.
        df = make_panel()
        res0 = LEXSCM({"df": df, **BASE}).fit()
        df_c = make_panel(clusters=[f"c{i}" for i in range(15)])   # all distinct
        res1 = LEXSCM({"df": df_c, **BASE, "cluster_col": "state"}).fit()
        assert list(res0.selected_units) == list(res1.selected_units)

    def test_infeasible_m_exceeds_clusters_raises(self):
        # 8 candidates but only 4 candidate-clusters (A,B,C,D); m=5 is infeasible.
        df = make_panel(clusters=CLUSTERS)
        with pytest.raises(MlsynthConfigError, match="spillover|conflict-free"):
            LEXSCM({"df": df, **{**BASE, "m": 5}, "cluster_col": "state"}).fit()

    @pytest.mark.parametrize("str_ids", [False, True])
    def test_label_invariant_holds_with_spillover(self, str_ids):
        # The returned labels are still keys in treated_weights (the #27 invariant),
        # now under the spillover constraint.
        df = make_panel(clusters=CLUSTERS, str_ids=str_ids)
        res = LEXSCM({"df": df, **BASE, "cluster_col": "state"}).fit()
        tw = res.design_weights.summary_stats["treated_weights"]
        for label in res.selected_units:
            assert label in tw
