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

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.fast_scm_helpers.structure import IndexSet
from mlsynth.utils.marex_helpers.setup import prepare_marex_panel


def _panel(N=8, T=14, seed=0, balanced=True):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T, 2))
    L = rng.uniform(0.3, 1.0, (N, 2))
    lvl = rng.uniform(8.0, 12.0, N)
    Y = lvl + F @ L.T + rng.normal(scale=0.3, size=(T, N))
    region = ["A" if j < N // 2 else "B" for j in range(N)]
    rows = []
    for j in range(N):
        for t in range(T):
            rows.append({"unit": f"u{j}", "time": t, "Y": float(Y[t, j]),
                         "region": region[j]})
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
