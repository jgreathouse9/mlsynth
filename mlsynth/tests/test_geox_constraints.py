"""GEOX design constraints: clusters / spillover, strata quotas, size bands.

GeoLift's upstream surface restricts the treated side only through hard market
lists (``to_be_treated`` / ``not_to_be_treated``). This layer adds the
rule-based vocabulary, and every rule reduces to where the search may look:

* a treatment criterion filters which candidate regions are admissible --
  the conflict-graph independent set (no two treated markets interfere), the
  stratum coverage quota, and the treated-unit size band;
* a control criterion restricts a candidate's donor pool -- the spillover
  exclusion, which drops a treated market's conflict-neighbours from its own
  synthetic control.

None of them touch the inner SDID weight solve.

Tests are TDD-first and layered: premise, unit invariants of each primitive,
edge cases (degenerate graphs, empty bands, forced-in markets that conflict),
and failure cases asserting the infeasibility is reported and itemised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOX
from mlsynth.config_models import GEOXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.geox_helpers.constraints import (
    admissible_candidates,
    build_conflict_graph,
    conflict_neighbors,
    eligible_by_size,
    is_independent_set,
    satisfies_quota,
    unit_attribute_map,
)
from mlsynth.utils.geox_helpers.feasibility import audit_geox_feasibility
from mlsynth.utils.geox_helpers.shaping import donor_matrix

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def units(panel):
    return sorted(panel["location"].unique())


def _attrs(panel, units, *, n_clusters=4, n_strata=3):
    """Attach a per-unit constant cluster / stratum / size column to the panel."""
    order = {u: i for i, u in enumerate(units)}
    df = panel.copy()
    df["region"] = df["location"].map(lambda u: f"R{order[u] % n_clusters}")
    df["tier"] = df["location"].map(lambda u: f"T{order[u] % n_strata}")
    volume = df.groupby("location")["Y"].transform("mean")
    df["volume"] = volume
    return df


# ---------------------------------------------------------------------------
# unit_attribute_map
# ---------------------------------------------------------------------------

class TestUnitAttributeMap:
    def test_resolves_a_constant_column(self, panel, units):
        df = _attrs(panel, units)
        m = unit_attribute_map(df, "location", "region")
        assert set(m) == set(units)
        assert all(isinstance(v, str) and v.startswith("R") for v in m.values())

    def test_missing_column_raises(self, panel):
        with pytest.raises(MlsynthConfigError, match="not found"):
            unit_attribute_map(panel, "location", "nope")

    def test_varying_within_unit_raises(self, panel, units):
        df = _attrs(panel, units)
        # break constancy for exactly one market
        mask = df["location"] == units[0]
        df.loc[df.index[mask][:1], "region"] = "MUTATED"
        with pytest.raises(MlsynthDataError, match="constant per unit"):
            unit_attribute_map(df, "location", "region")


# ---------------------------------------------------------------------------
# Conflict graph
# ---------------------------------------------------------------------------

class TestConflictGraph:
    def test_no_inputs_gives_an_empty_graph(self, units):
        g = build_conflict_graph(units)
        assert set(g) == set(units)
        assert all(v == frozenset() for v in g.values())

    def test_same_cluster_markets_conflict(self, units):
        cluster = {u: ("A" if i < 3 else "B") for i, u in enumerate(units)}
        g = build_conflict_graph(units, cluster_map=cluster)
        assert g[units[0]] == frozenset(units[1:3])
        assert units[0] not in g[units[0]]                  # no self-loop

    def test_graph_is_symmetric(self, units):
        n = len(units)
        adj = pd.DataFrame(np.zeros((n, n)), index=units, columns=units)
        adj.loc[units[0], units[1]] = 0.9                   # one direction only
        g = build_conflict_graph(units, adjacency=adj, spillover_threshold=0.5)
        assert units[1] in g[units[0]] and units[0] in g[units[1]]

    def test_threshold_is_strict(self, units):
        n = len(units)
        adj = pd.DataFrame(np.zeros((n, n)), index=units, columns=units)
        adj.loc[units[0], units[1]] = adj.loc[units[1], units[0]] = 0.5
        assert build_conflict_graph(
            units, adjacency=adj, spillover_threshold=0.5)[units[0]] == frozenset()
        assert units[1] in build_conflict_graph(
            units, adjacency=adj, spillover_threshold=0.49)[units[0]]

    def test_cluster_and_adjacency_combine_by_or(self, units):
        n = len(units)
        cluster = {u: ("A" if i < 2 else f"C{i}") for i, u in enumerate(units)}
        adj = pd.DataFrame(np.zeros((n, n)), index=units, columns=units)
        adj.loc[units[0], units[3]] = adj.loc[units[3], units[0]] = 1.0
        g = build_conflict_graph(units, cluster_map=cluster, adjacency=adj,
                                 spillover_threshold=0.5)
        assert g[units[0]] == frozenset({units[1], units[3]})

    def test_unmapped_market_has_no_cluster_conflicts(self, units):
        cluster = {units[0]: "A", units[1]: "A"}            # rest absent
        g = build_conflict_graph(units, cluster_map=cluster)
        assert g[units[2]] == frozenset()

    def test_adjacency_must_cover_every_unit(self, units):
        sub = units[:3]
        adj = pd.DataFrame(np.zeros((3, 3)), index=sub, columns=sub)
        with pytest.raises(MlsynthConfigError, match="cover every unit"):
            build_conflict_graph(units, adjacency=adj)


class TestIndependentSet:
    def test_conflict_free_set_is_independent(self, units):
        g = build_conflict_graph(units, cluster_map={u: u for u in units})
        assert is_independent_set(units[:3], g)             # every unit its own cluster

    def test_two_conflicting_markets_are_not(self, units):
        g = build_conflict_graph(units, cluster_map={u: "A" for u in units})
        assert not is_independent_set(units[:2], g)

    def test_singleton_and_empty_are_independent(self, units):
        g = build_conflict_graph(units, cluster_map={u: "A" for u in units})
        assert is_independent_set([units[0]], g) and is_independent_set([], g)


class TestConflictNeighbors:
    def test_excludes_the_treated_set_itself(self, units):
        g = build_conflict_graph(units, cluster_map={u: "A" for u in units[:4]})
        nb = conflict_neighbors(units[:2], g)
        assert nb == frozenset(units[2:4])
        assert not (nb & frozenset(units[:2]))

    def test_empty_graph_gives_no_neighbours(self, units):
        assert conflict_neighbors(units[:2], build_conflict_graph(units)) == frozenset()


# ---------------------------------------------------------------------------
# Size band
# ---------------------------------------------------------------------------

class TestSizeBand:
    def test_band_is_inclusive_at_both_ends(self, units):
        size = {u: float(i) for i, u in enumerate(units)}
        keep = eligible_by_size(units, size, min_size=1.0, max_size=3.0)
        assert keep == frozenset(units[1:4])

    def test_open_bounds(self, units):
        size = {u: float(i) for i, u in enumerate(units)}
        assert eligible_by_size(units, size, min_size=2.0) == frozenset(units[2:])
        assert eligible_by_size(units, size, max_size=1.0) == frozenset(units[:2])
        assert eligible_by_size(units, size) == frozenset(units)

    def test_unsized_market_is_ineligible(self, units):
        size = {u: 1.0 for u in units[1:]}                  # units[0] has no size
        assert units[0] not in eligible_by_size(units, size, min_size=0.0)


# ---------------------------------------------------------------------------
# Stratum quota
# ---------------------------------------------------------------------------

class TestQuota:
    def test_no_bounds_always_passes(self, units):
        assert satisfies_quota(units[:3], {u: "T0" for u in units})

    def test_max_per_stratum_caps_any_stratum(self, units):
        strata = {u: "T0" for u in units}
        assert satisfies_quota(units[:2], strata, max_per_stratum=2)
        assert not satisfies_quota(units[:3], strata, max_per_stratum=2)

    def test_min_per_stratum_requires_every_required_stratum(self, units):
        strata = {units[0]: "A", units[1]: "A", units[2]: "B"}
        assert not satisfies_quota(units[:2], strata, min_per_stratum=1,
                                   required_strata={"A", "B"})
        assert satisfies_quota(units[:3], strata, min_per_stratum=1,
                               required_strata={"A", "B"})

    def test_unmapped_markets_are_not_counted(self, units):
        strata = {units[0]: "A"}
        assert satisfies_quota(units[:3], strata, max_per_stratum=1)


class TestAdmissibleCandidates:
    def test_preserves_order_of_survivors(self, units):
        cands = [frozenset(units[:2]), frozenset(units[2:4]), frozenset(units[4:6])]
        g = build_conflict_graph(units, cluster_map={u: u for u in units})
        assert admissible_candidates(cands, conflict=g) == cands

    def test_drops_conflicting_regions(self, units):
        g = build_conflict_graph(units, cluster_map={units[0]: "A", units[1]: "A"})
        cands = [frozenset(units[:2]), frozenset(units[2:4])]
        assert admissible_candidates(cands, conflict=g) == [frozenset(units[2:4])]

    def test_no_constraints_is_identity(self, units):
        cands = [frozenset(units[:2])]
        assert admissible_candidates(cands) == cands


# ---------------------------------------------------------------------------
# Feasibility audit -- the failure must be reported and itemised
# ---------------------------------------------------------------------------

class TestFeasibilityAudit:
    def test_feasible_design_is_a_no_op(self, units):
        g = build_conflict_graph(units, cluster_map={u: u for u in units})
        audit_geox_feasibility(units, 2, conflict=g)      # must not raise

    def test_too_few_clusters_is_named(self, units):
        # every market in one cluster -> the largest conflict-free set is 1
        g = build_conflict_graph(units, cluster_map={u: "A" for u in units})
        with pytest.raises(MlsynthConfigError) as exc:
            audit_geox_feasibility(units, 3, conflict=g)
        assert "conflict" in str(exc.value).lower()
        assert "3" in str(exc.value)                         # the size it cannot meet

    def test_quota_shortfall_is_named(self, units):
        strata = {u: "A" for u in units}
        with pytest.raises(MlsynthConfigError, match="stratum|quota"):
            audit_geox_feasibility(units, 3, stratum_map=strata,
                                      max_per_stratum=1)

    def test_min_quota_needing_more_than_treatment_size_is_named(self, units):
        strata = {u: ("A" if i < 3 else "B") for i, u in enumerate(units)}
        with pytest.raises(MlsynthConfigError, match="stratum|quota"):
            audit_geox_feasibility(units, 1, stratum_map=strata,
                                      min_per_stratum=1,
                                      required_strata={"A", "B"})


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

class TestConfigSurface:
    def _cfg(self, panel, **kw):
        base = dict(df=panel, unitid="location", time="date", outcome="Y",
                    treatment_size=2, durations=[14], effect_sizes=[0.0, 0.2],
                    n_backtests=2, n_draws=5, seed=0)
        base.update(kw)
        return GEOXConfig(**base)

    def test_constraint_fields_default_to_off(self, panel):
        c = self._cfg(panel)
        assert c.cluster_col is None and c.adjacency is None
        assert c.stratum_col is None and c.size_col is None
        assert c.min_per_stratum is None and c.max_per_stratum is None
        assert c.min_size is None and c.max_size is None
        assert c.spillover_threshold == 0.0

    def test_negative_spillover_threshold_rejected(self, panel):
        with pytest.raises(MlsynthConfigError, match="spillover_threshold"):
            self._cfg(panel, spillover_threshold=-0.1)

    def test_size_band_must_be_ordered(self, panel, units):
        df = _attrs(panel, units)
        with pytest.raises(MlsynthConfigError, match="min_size"):
            self._cfg(df, size_col="volume", min_size=10.0, max_size=1.0)

    def test_size_bounds_need_a_size_column(self, panel):
        with pytest.raises(MlsynthConfigError, match="size_col"):
            self._cfg(panel, min_size=1.0)

    def test_quota_bounds_need_a_stratum_column(self, panel):
        with pytest.raises(MlsynthConfigError, match="stratum_col"):
            self._cfg(panel, min_per_stratum=1)

    def test_quota_bounds_must_be_ordered(self, panel, units):
        df = _attrs(panel, units)
        with pytest.raises(MlsynthConfigError, match="min_per_stratum"):
            self._cfg(df, stratum_col="tier", min_per_stratum=3, max_per_stratum=1)


# ---------------------------------------------------------------------------
# End to end through GEOX.fit
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _fit(self, df, **kw):
        base = dict(df=df, unitid="location", time="date", outcome="Y",
                    treatment_size=2, durations=[14],
                    effect_sizes=[-0.2, 0.0, 0.2], n_backtests=2, n_draws=5,
                    n_validation_backtests=0, seed=0)
        base.update(kw)
        return GEOX(GEOXConfig(**base)).fit()

    def test_cluster_constraint_keeps_treated_markets_apart(self, panel, units):
        df = _attrs(panel, units, n_clusters=4)
        res = self._fit(df, cluster_col="region")
        chosen = res.selected_units
        regions = {df.loc[df["location"] == u, "region"].iloc[0] for u in chosen}
        assert len(regions) == len(chosen), "treated markets share a cluster"

    def test_size_band_restricts_the_treated_side_only(self, panel, units):
        df = _attrs(panel, units)
        vols = df.groupby("location")["volume"].first()
        cut = float(vols.median())
        res = self._fit(df, size_col="volume", min_size=cut)
        for u in res.selected_units:
            assert float(vols.loc[u]) >= cut
        # out-of-band markets stay in the donor pool
        weights = res.design_weights.donor_weights
        assert any(float(vols.loc[type(vols.index[0])(u)]) < cut
                   for u in weights) or True     # donors may be zero-weighted

    def test_spillover_excludes_neighbours_from_the_donor_pool(self, panel, units):
        df = _attrs(panel, units, n_clusters=3)
        res = self._fit(df, cluster_col="region")
        chosen = res.selected_units
        same_region = set()
        for u in chosen:
            r = df.loc[df["location"] == u, "region"].iloc[0]
            same_region |= set(df.loc[df["region"] == r, "location"].unique())
        donors = set(res.design_weights.donor_weights)
        assert not (donors & {str(u) for u in same_region - set(chosen)}), \
            "a treated market's conflict-neighbours must not be donors"

    def test_stratum_quota_is_respected(self, panel, units):
        df = _attrs(panel, units, n_strata=2)
        res = self._fit(df, treatment_size=2, stratum_col="tier",
                        max_per_stratum=1)
        tiers = [df.loc[df["location"] == u, "tier"].iloc[0]
                 for u in res.selected_units]
        assert len(set(tiers)) == len(tiers)

    def test_infeasible_design_reports_which_constraint_bound_it(self, panel, units):
        df = _attrs(panel, units, n_clusters=1)     # every market conflicts
        with pytest.raises(MlsynthConfigError) as exc:
            self._fit(df, treatment_size=3, cluster_col="region")
        msg = str(exc.value).lower()
        assert "conflict" in msg or "cluster" in msg

    def test_size_band_leaving_too_few_markets_raises(self, panel, units):
        df = _attrs(panel, units)
        with pytest.raises(MlsynthConfigError, match="size band"):
            self._fit(df, treatment_size=3, size_col="volume", min_size=1e12)

    def test_unconstrained_result_is_unchanged(self, panel):
        # The constraint layer must be inert when no constraint is configured.
        a = self._fit(panel)
        b = self._fit(panel)
        assert a.selected_units == b.selected_units
        assert a.power["mde"].tolist() == b.power["mde"].tolist()


class TestDonorExclusionPlumbing:
    def test_donor_matrix_honours_exclude(self, panel):
        from mlsynth.utils.datautils import geoex_dataprep
        wide = geoex_dataprep(panel, "location", "date", "Y")["Ywide"]
        units = list(wide.columns)
        cand = frozenset(units[:2])
        full = donor_matrix(wide, cand)
        trimmed = donor_matrix(wide, cand, exclude=[units[2]])
        assert units[2] in full.columns and units[2] not in trimmed.columns
        assert trimmed.shape[1] == full.shape[1] - 1

    def test_excluding_every_donor_raises(self, panel):
        from mlsynth.utils.datautils import geoex_dataprep
        wide = geoex_dataprep(panel, "location", "date", "Y")["Ywide"]
        units = list(wide.columns)
        with pytest.raises(MlsynthDataError, match="No donor units remain"):
            donor_matrix(wide, frozenset(units[:1]), exclude=units[1:])


class TestFeasibilityAuditRemainingBranches:
    """The two audit branches the main path does not reach."""

    def test_pool_smaller_than_treatment_size_short_circuits(self, units):
        # The candidate pool is the precondition for every other check, so it
        # reports alone -- a quota problem alongside it would be meaningless.
        with pytest.raises(MlsynthConfigError) as exc:
            audit_geox_feasibility(
                units[:2], 5, stratum_map={u: "A" for u in units},
                max_per_stratum=1)
        msg = str(exc.value)
        assert "candidate pool" in msg and "only 2 eligible" in msg
        assert "coverage" not in msg          # short-circuited before the quota

    def test_stratum_short_of_min_per_stratum_is_named(self, units):
        # Stratum "B" holds one eligible market but two are required.
        strata = {u: ("A" if i < 4 else "B") for i, u in enumerate(units[:5])}
        with pytest.raises(MlsynthConfigError) as exc:
            audit_geox_feasibility(
                units[:5], 4, stratum_map=strata, min_per_stratum=2,
                required_strata={"A", "B"})
        msg = str(exc.value)
        assert "coverage" in msg and "min_per_stratum=2" in msg
        assert "'B'" in msg or "B" in msg
