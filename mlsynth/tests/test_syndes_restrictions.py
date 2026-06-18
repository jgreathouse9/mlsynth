"""TDD for SYNDES design restrictions (geography / clustering / size / forcing).

SYNDES selects the treated set by a MIP over a binary assignment vector ``D``.
The GEOLIFT/LEXSCM restriction vocabulary maps to linear constraints on ``D``:

* ``to_be_treated``      -> ``D_i = 1``      (force a unit in)
* ``not_to_be_treated``  -> ``D_i = 0``      (forbid treatment; stays a donor)
* ``size_col`` band      -> ``D_i = 0`` for out-of-band units
* ``cluster_col`` /
  ``adjacency`` /
  ``spillover_threshold`` -> ``D_i + D_j <= 1`` for every conflicting pair
* ``stratum_col`` quota  -> ``min <= sum_{i in stratum} D_i <= max``

These tests pin the pure restriction builder/applier (Layer 1), the estimator
enforcement that the solved design honours every restriction (Layer 3), and the
config/failure semantics (translated ``MlsynthConfigError``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SYNDES
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.syndes_helpers.restrictions import (
    DesignRestrictions,
    build_restrictions,
    apply_restrictions,
)

_MARKETS = Path(__file__).resolve().parents[2] / "basedata" / "markets"


_CLUSTER = {f"u{j}": f"c{j // 2}" for j in range(8)}      # u0,u1->c0 ; u2,u3->c1 ; ...
_REGION = {f"u{j}": ("east" if j < 4 else "west") for j in range(8)}
_SIZE = {f"u{j}": float(j + 1) for j in range(8)}          # 1..8


def _panel(N=8, T=14, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    rows = []
    for j in range(N):
        u = f"u{j}"
        for t in range(T):
            rows.append({"unit": u, "time": t, "Y": float(Y[t, j]),
                         "post": int(t >= T - n_post),
                         "cluster": _CLUSTER[u], "region": _REGION[u],
                         "size": _SIZE[u]})
    return pd.DataFrame(rows)


def _uindex(N=8):
    return IndexSet.from_labels([f"u{j}" for j in range(N)])


# ----------------------------------------------------------------------
# Layer 1 -- build_restrictions
# ----------------------------------------------------------------------

class TestBuildRestrictions:
    def test_forced_and_forbidden_to_indices(self):
        r = build_restrictions(_panel(), "unit", _uindex(),
                               to_be_treated=["u1"], not_to_be_treated=["u5", "u7"])
        assert r.forced_in == [1]
        assert r.forbidden == [5, 7]

    def test_size_band_forbids_out_of_band(self):
        # sizes are 1..8; band [3, 6] keeps u2..u5 (sizes 3..6) treatable.
        r = build_restrictions(_panel(), "unit", _uindex(),
                               size_col="size", min_size=3.0, max_size=6.0)
        assert r.forbidden == [0, 1, 6, 7]               # sizes 1,2,7,8 excluded

    def test_cluster_conflict_pairs(self):
        r = build_restrictions(_panel(), "unit", _uindex(), cluster_col="cluster")
        # same-cluster pairs (i<j): (0,1),(2,3),(4,5),(6,7)
        assert set(r.conflict_pairs) == {(0, 1), (2, 3), (4, 5), (6, 7)}

    def test_adjacency_conflict_pairs(self):
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        A.loc["u0", "u3"] = A.loc["u3", "u0"] = 0.9       # one spillover edge
        r = build_restrictions(_panel(), "unit", _uindex(),
                               adjacency=A, spillover_threshold=0.5)
        assert set(r.conflict_pairs) == {(0, 3)}

    def test_stratum_min_max_groups(self):
        r = build_restrictions(_panel(), "unit", _uindex(),
                               stratum_col="region", min_per_stratum=1,
                               max_per_stratum=2)
        groups = {tuple(sorted(m)): (lo, hi) for m, lo, hi in r.strata}
        assert groups == {(0, 1, 2, 3): (1, 2), (4, 5, 6, 7): (1, 2)}

    def test_forced_forbidden_overlap_raises(self):
        with pytest.raises(MlsynthConfigError):
            build_restrictions(_panel(), "unit", _uindex(),
                               to_be_treated=["u1"], not_to_be_treated=["u1"])

    def test_empty_when_no_restrictions(self):
        r = build_restrictions(_panel(), "unit", _uindex())
        assert r.is_empty

    def test_unknown_label_raises(self):
        with pytest.raises(MlsynthConfigError):
            build_restrictions(_panel(), "unit", _uindex(),
                               to_be_treated=["ghost"])

    def test_missing_stratum_value_skipped(self):
        df = _panel()
        df.loc[df["unit"] == "u0", "region"] = np.nan      # u0 has no stratum
        r = build_restrictions(df, "unit", _uindex(),
                               stratum_col="region", max_per_stratum=1)
        members = sorted(sum((list(m) for m, _, _ in r.strata), []))
        assert 0 not in members                            # u0 excluded
        assert set(members) == {1, 2, 3, 4, 5, 6, 7}


# ----------------------------------------------------------------------
# Layer 1 -- apply_restrictions (constraint count, exercised via a tiny solve)
# ----------------------------------------------------------------------

class TestApplyRestrictions:
    def test_builds_one_constraint_per_rule(self):
        import cvxpy as cp
        D = cp.Variable(8, boolean=True)
        r = DesignRestrictions(forced_in=[1], forbidden=[5],
                               conflict_pairs=[(0, 2)],
                               strata=[((0, 1, 2, 3), 1, 2)])
        cons = apply_restrictions(D, r)
        # 1 forced + 1 forbidden + 1 conflict + (1 lower + 1 upper) = 5
        assert len(cons) == 5


# ----------------------------------------------------------------------
# Layer 3 -- the solved design honours every restriction
# ----------------------------------------------------------------------

class TestSYNDESRestrictionsEnforced:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 mode="two_way_global", run_inference=False, solver="SCIP",
                 gap_limit=0.2, time_limit=8.0)

    def _treated(self, **over):
        res = SYNDES({"df": _panel(), **self._BASE, **over}).fit()
        return set(map(str, np.asarray(res.selected_unit_labels).tolist()))

    def test_not_to_be_treated_excluded(self):
        treated = self._treated(K=2, not_to_be_treated=["u0", "u1", "u2"])
        assert treated.isdisjoint({"u0", "u1", "u2"})

    def test_to_be_treated_forced_in(self):
        treated = self._treated(K=3, to_be_treated=["u6"])
        assert "u6" in treated

    def test_cluster_no_two_treated_same_cluster(self):
        treated = self._treated(K=3, cluster_col="cluster")
        clusters = [_CLUSTER[u] for u in treated]
        assert len(clusters) == len(set(clusters))

    def test_adjacency_no_conflicting_pair_treated(self):
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        for a, b in [("u0", "u1"), ("u2", "u3")]:
            A.loc[a, b] = A.loc[b, a] = 1.0
        res = SYNDES({"df": _panel(), **self._BASE, "K": 2,
                      "adjacency": A, "spillover_threshold": 0.5}).fit()
        treated = set(map(str, np.asarray(res.selected_unit_labels).tolist()))
        assert treated not in ({"u0", "u1"}, {"u2", "u3"})

    def test_size_band_excludes_out_of_band(self):
        treated = self._treated(K=2, size_col="size", min_size=3.0, max_size=6.0)
        assert treated <= {"u2", "u3", "u4", "u5"}

    def test_stratum_max_per_stratum(self):
        treated = self._treated(K=2, stratum_col="region", max_per_stratum=1)
        regions = [_REGION[u] for u in treated]
        assert regions.count("east") <= 1 and regions.count("west") <= 1

    def test_stratum_min_per_stratum(self):
        treated = self._treated(K=2, stratum_col="region", min_per_stratum=1)
        regions = {_REGION[u] for u in treated}
        assert regions == {"east", "west"}              # at least one each

    def test_restrictions_flow_through_holdout(self):
        treated = self._treated(K=2, not_to_be_treated=["u0", "u1"],
                                top_K=4, holdout_frac=0.3)
        assert treated.isdisjoint({"u0", "u1"})

    def test_restrictions_flow_through_ic(self):
        treated = self._treated(K=2, not_to_be_treated=["u0", "u1"],
                                top_K=4, selection="ic")
        assert treated.isdisjoint({"u0", "u1"})


# ----------------------------------------------------------------------
# Config / failure semantics
# ----------------------------------------------------------------------

class TestSYNDESRestrictionsConfig:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 mode="two_way_global", run_inference=False)

    def _make(self, **over):
        return SYNDES({"df": _panel(), **self._BASE, **over})

    def test_unknown_forced_unit(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, to_be_treated=["nope"]).fit()

    def test_forced_forbidden_disjoint(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, to_be_treated=["u1"], not_to_be_treated=["u1"])

    def test_too_many_forced(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, to_be_treated=["u1", "u2", "u3"])

    def test_stratum_quota_requires_col(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, min_per_stratum=1)

    def test_size_band_requires_col(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, min_size=2.0)

    def test_cluster_col_must_be_in_df(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, cluster_col="not_a_column")

    @pytest.mark.parametrize("field", ["min_per_stratum", "max_per_stratum"])
    def test_stratum_bound_must_be_positive(self, field):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, stratum_col="region", **{field: 0})

    def test_min_size_le_max_size(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, size_col="size", min_size=6.0, max_size=3.0)

    def test_restrictions_reject_arm(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, arm="region", not_to_be_treated=["u0"])

    def test_restrictions_reject_annealed(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=2, mode="two_way_global_annealed",
                       not_to_be_treated=["u0"])

    def test_forbidden_leaves_too_few_treatable(self):
        with pytest.raises(MlsynthConfigError):
            self._make(K=3, not_to_be_treated=[f"u{j}" for j in range(6)]).fit()


# ----------------------------------------------------------------------
# Infeasibility: over-constrained restrictions return a translated, informative
# error (not a leaked solver INFEASIBLE).
# ----------------------------------------------------------------------

class TestSYNDESRestrictionsInfeasible:
    _BASE = dict(outcome="Y", unitid="unit", time="time", post_col="post",
                 mode="two_way_global", run_inference=False, solver="SCIP",
                 gap_limit=0.2, time_limit=8.0)

    def test_adjacency_clique_is_infeasible_and_reported(self):
        # u0..u3 form a conflict clique (all mutually adjacent); u4..u7 forbidden.
        # Only the clique is treatable, but no two of its members can both be
        # treated -> K=3 is infeasible. The error must be translated and name the
        # restrictions, not leak a bare solver status.
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        for a in range(4):
            for b in range(a + 1, 4):
                A.loc[labels[a], labels[b]] = A.loc[labels[b], labels[a]] = 1.0
        with pytest.raises(MlsynthEstimationError) as ei:
            SYNDES({"df": _panel(), **self._BASE, "K": 3,
                    "adjacency": A, "spillover_threshold": 0.5,
                    "not_to_be_treated": ["u4", "u5", "u6", "u7"]}).fit()
        assert "restriction" in str(ei.value).lower()


# ----------------------------------------------------------------------
# Real geography: validate against the bundled DMA contiguity matrix
# (basedata/markets/), restricting to Florida + Georgia.
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def dma():
    adj = pd.read_csv(_MARKETS / "dma_adjacency.csv", index_col=0)
    meta = pd.read_csv(_MARKETS / "dma_metadata.csv")
    return adj, meta


def _flga(meta):
    return meta[meta["state"].isin(["FL", "GA"])]["dma_name"].tolist()


def _geo_panel(units, T=16, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (len(units), 2))
    lvl = rng.uniform(8.0, 12.0, len(units))
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, len(units)))
    rows = [{"market": u, "t": t, "Y": float(Y[t, j]),
             "post": int(t >= T - n_post)}
            for j, u in enumerate(units) for t in range(T)]
    return pd.DataFrame(rows)


class TestSYNDESRealDMAGeography:
    _BASE = dict(outcome="Y", unitid="market", time="t", post_col="post",
                 mode="two_way_global", run_inference=False, solver="SCIP",
                 gap_limit=0.2, time_limit=12.0)

    def test_feasible_design_respects_real_borders(self, dma):
        adj, meta = dma
        units = _flga(meta)
        res = SYNDES({"df": _geo_panel(units), **self._BASE, "K": 4,
                      "adjacency": adj, "spillover_threshold": 0.5}).fit()
        treated = [str(x) for x in np.asarray(res.selected_unit_labels).tolist()]
        assert len(treated) == 4
        sub = adj.reindex(index=treated, columns=treated).to_numpy()
        assert int(np.triu(sub, 1).sum()) == 0          # no two share a border

    def test_full_matrix_auto_subsets_to_panel(self, dma):
        # Passing the full 206x206 matrix for a 16-unit panel must subset, not error.
        adj, meta = dma
        res = SYNDES({"df": _geo_panel(_flga(meta)), **self._BASE, "K": 3,
                      "adjacency": adj, "spillover_threshold": 0.5}).fit()
        assert len(np.asarray(res.selected_unit_labels)) == 3

    def test_restricting_treatment_to_flga_too_few_for_K(self, dma):
        adj, meta = dma
        flga = _flga(meta)
        donors = meta[meta["state"].isin(["AL", "SC"])]["dma_name"].tolist()[:6]
        forbid = donors                                  # treat only FL+GA (16)
        with pytest.raises(MlsynthConfigError):
            SYNDES({"df": _geo_panel(flga + donors), **self._BASE, "K": 20,
                    "not_to_be_treated": forbid}).fit()

    def test_adjacency_missing_a_panel_unit(self, dma):
        adj, meta = dma
        units = _flga(meta)[:8] + ["Atlantis, XX"]       # fake DMA not in matrix
        with pytest.raises(MlsynthDataError):
            SYNDES({"df": _geo_panel(units), **self._BASE, "K": 3,
                    "adjacency": adj, "spillover_threshold": 0.5}).fit()

    def test_over_constrained_borders_infeasible(self, dma):
        adj, meta = dma
        units = _flga(meta)                              # 16 bordering DMAs
        with pytest.raises(MlsynthEstimationError):      # K=14 > max independent set
            SYNDES({"df": _geo_panel(units), **self._BASE, "K": 14,
                    "adjacency": adj, "spillover_threshold": 0.5}).fit()
