"""TDD for the MAREX IndexSet refactor (Phase 1) + geographic restrictions.

Phase 1 puts ``IndexSet`` in charge of unit/time identity: ``prepare_marex_panel``
ingests through the canonical :func:`geoex_dataprep` (which enforces a strongly
balanced panel) and carries ``unit_index`` / ``time_index`` IndexSets that are the
single source of truth downstream. Behaviour on a balanced panel is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import MAREX
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.marex_helpers.setup import prepare_marex_panel
from mlsynth.utils.marex_helpers.restrictions import (
    DesignRestrictions,
    apply_restrictions_marex,
)


def _panel(N=8, T=14, seed=0, balanced=True):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    region = ["A" if j < N // 2 else "B" for j in range(N)]
    # Deterministic sizes with the same spread in each region, so a mid-band
    # always leaves >=1 treatable unit per region (MAREX needs >=1 treated/cluster).
    size = np.tile([2.0, 4.0, 6.0, 8.0], (N + 3) // 4)[:N]
    rows = []
    for j in range(N):
        for t in range(T):
            rows.append({"unit": f"u{j}", "time": t, "Y": float(Y[t, j]),
                         "region": region[j], "size": float(size[j])})
    df = pd.DataFrame(rows)
    if not balanced:
        df = df.iloc[3:]            # drop a few rows -> unbalanced panel
    return df


class TestMAREXIndexSet:
    def _prep(self, df, **kw):
        return prepare_marex_panel(
            df, outcome="Y", unitid="unit", time="time", cluster="region",
            T0=10, inference=False, blank_periods=0, T_post=None, **kw)

    def test_panel_carries_indexsets(self):
        panel = self._prep(_panel())
        assert isinstance(panel.unit_index, IndexSet)
        assert isinstance(panel.time_index, IndexSet)

    def test_unit_index_is_source_of_truth(self):
        # Y_full rows align to unit_index labels; clusters align to that order.
        panel = self._prep(_panel())
        assert list(panel.Y_full.index) == list(panel.unit_index.labels)
        assert len(panel.clusters) == len(panel.unit_index)

    def test_time_index_matches_columns(self):
        panel = self._prep(_panel())
        assert list(panel.Y_full.columns) == list(panel.time_index.labels)

    def test_balance_enforced_via_geoex(self):
        # geoex_dataprep rejects an unbalanced panel -> translated MlsynthDataError.
        with pytest.raises(MlsynthDataError):
            self._prep(_panel(balanced=False))

    def test_behaviour_preserved_on_balanced_panel(self):
        # The refactor preserves the unit order and the (units x time) Y_full.
        panel = self._prep(_panel())
        assert panel.Y_full.shape == (8, 14)
        assert set(panel.unit_index.labels) == {f"u{j}" for j in range(8)}


# ----------------------------------------------------------------------
# Layer 1 -- apply_restrictions_marex (constraint emission on z / v)
# ----------------------------------------------------------------------

class TestApplyRestrictionsMarex:
    def _vars(self, N=6, K=2):
        import cvxpy as cp
        z = cp.Variable((N, K), boolean=True)
        v = cp.Variable((N, K), nonneg=True)
        M = np.zeros((N, K), dtype=bool)
        for j in range(N):
            M[j, 0 if j < N // 2 else 1] = True      # first half cluster 0, rest 1
        return z, v, M

    def test_force_forbid_conflict_counts(self):
        z, v, M = self._vars()
        r = DesignRestrictions(forced_in=[0], forbidden=[1, 2],
                               conflict_pairs=[(0, 3)])
        cons = apply_restrictions_marex(z, v, M, r)
        assert len(cons) == 4                         # 1 forced + 2 forbidden + 1 conflict

    def test_donor_exclusion_only_within_cluster(self):
        z, v, M = self._vars()                        # u0,u1,u2 in cluster 0
        # (0,1) same cluster -> 1 constraint; (0,4) cross cluster -> skipped.
        r = DesignRestrictions(donor_exclusion=[(0, 1), (0, 4)])
        cons = apply_restrictions_marex(z, v, M, r)
        assert len(cons) == 1

    def test_empty_restrictions_no_constraints(self):
        z, v, M = self._vars()
        assert apply_restrictions_marex(z, v, M, DesignRestrictions()) == []


# ----------------------------------------------------------------------
# Layer 3 -- the solved MAREX design honours the restrictions
# ----------------------------------------------------------------------

class TestMAREXRestrictionsEnforced:
    _BASE = dict(outcome="Y", unitid="unit", time="time", cluster="region",
                 T0=10, m_max=1, solver="SCIP", verbose=False)

    def _fit(self, **over):
        return MAREX({"df": _panel(), **self._BASE, **over}).fit()

    def _treated(self, res):
        return set(map(str, res.assignment["treated"]))

    def test_not_to_be_treated_excluded(self):
        res = self._fit(not_to_be_treated=["u0", "u1", "u4"])
        assert self._treated(res).isdisjoint({"u0", "u1", "u4"})

    def test_to_be_treated_forced_in(self):
        res = self._fit(to_be_treated=["u0", "u4"])
        assert {"u0", "u4"} <= self._treated(res)

    def test_adjacency_no_two_treated_border(self):
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        # make the two natural treated picks (one per region) border-conflict
        for a, b in [("u0", "u1"), ("u4", "u5")]:
            A.loc[a, b] = A.loc[b, a] = 1.0
        res = self._fit(adjacency=A, spillover_threshold=0.5)
        treated = self._treated(res)
        Ab = A.to_numpy()
        idx = {labels.index(u) for u in treated}
        assert not any(Ab[i, j] > 0 for i in idx for j in idx if i != j)

    def test_size_band_excludes_out_of_band(self):
        df = _panel()
        size = df.drop_duplicates("unit").set_index("unit")["size"]
        lo, hi = 3.0, 7.0                            # keeps sizes {4, 6} per region
        res = MAREX({"df": df, **self._BASE, "size_col": "size",
                     "min_size": lo, "max_size": hi}).fit()
        treated = self._treated(res)
        in_band = set(size[(size >= lo) & (size <= hi)].index)
        assert treated <= in_band

    def test_exclude_bordering_donors(self):
        labels = [f"u{j}" for j in range(8)]
        A = pd.DataFrame(0.0, index=labels, columns=labels)
        for a, b in [("u0", "u2"), ("u4", "u6")]:    # within-region borders
            A.loc[a, b] = A.loc[b, a] = 1.0
        res = MAREX({"df": _panel(), **self._BASE, "m_max": 1,
                     "adjacency": A, "spillover_threshold": 0.5,
                     "exclude_bordering_donors": True}).fit()
        treated = set(map(str, res.assignment["treated"]))
        control = set(map(str, res.assignment["control"]))
        Ab = A.to_numpy()
        ti = {labels.index(u) for u in treated}
        ci = {labels.index(u) for u in control}
        assert not any(Ab[i, j] > 0 for i in ti for j in ci)   # no treated-donor border


# ----------------------------------------------------------------------
# Config / failure semantics
# ----------------------------------------------------------------------

class TestMAREXRestrictionsConfig:
    _BASE = dict(outcome="Y", unitid="unit", time="time", cluster="region",
                 T0=10, m_max=1)

    def _make(self, **over):
        return MAREX({"df": _panel(), **self._BASE, **over})

    def test_unknown_forced_unit(self):
        with pytest.raises(MlsynthConfigError):
            self._make(to_be_treated=["ghost"]).fit()

    def test_forced_forbidden_overlap(self):
        with pytest.raises(MlsynthConfigError):
            self._make(to_be_treated=["u0"], not_to_be_treated=["u0"])

    def test_size_band_requires_col(self):
        with pytest.raises(MlsynthConfigError):
            self._make(min_size=2.0)

    def test_exclude_bordering_needs_adjacency(self):
        with pytest.raises(MlsynthConfigError):
            self._make(exclude_bordering_donors=True)

    def test_restrictions_reject_relaxed(self):
        with pytest.raises(MlsynthConfigError):
            self._make(relaxed=True, not_to_be_treated=["u0"])

    def test_infeasible_restrictions_reported(self):
        # Forbidding every unit of region B leaves its treated synthetic with no
        # member (MAREX needs >=1 treated per cluster) -> translated estimation
        # error naming the restrictions, not a bare solver status.
        from mlsynth.exceptions import MlsynthEstimationError
        with pytest.raises(MlsynthEstimationError) as ei:
            MAREX({"df": _panel(), **self._BASE, "solver": "SCIP",
                   "not_to_be_treated": ["u4", "u5", "u6", "u7"]}).fit()
        assert "restriction" in str(ei.value).lower()
