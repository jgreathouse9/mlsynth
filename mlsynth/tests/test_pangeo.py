"""Tests for the PANGEO experimental-design estimator.

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): parallelism scoring + admissible-pair
  enumeration + best split.
* Layer 2 (data utilities): historical-panel ingestion + arm pools.
* Layer 3 (integration): the design exact-covers each arm, respects Q,
  and beats random assignment on pre-period parallelism (the whole point).
* Layer 4 (public API contracts): import, frozen results, config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import PANGEO
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.pangeo_helpers import (
    PangeoResults,
    enumerate_candidate_pairs,
    gap_variance,
    make_seasonal_sales_panel,
    prepare_pangeo_inputs,
    run_pangeo,
)
from mlsynth.utils.pangeo_helpers.parallelism import best_split


@pytest.fixture
def panel():
    return make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                     T=120, seed=0)


# ----------------------------------------------------------------------
# Layer 1: parallelism helpers
# ----------------------------------------------------------------------

class TestParallelism:
    def test_gap_variance_zero_for_parallel(self):
        # Two trajectories differing by a constant level -> perfectly parallel.
        a = np.array([1.0, 2.0, 3.0, 2.5])
        b = a + 7.0
        assert gap_variance(a, b) == pytest.approx(0.0, abs=1e-12)

    def test_gap_variance_positive_for_nonparallel(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])   # anti-parallel
        assert gap_variance(a, b) > 1.0

    def test_best_split_picks_parallel_halves(self):
        # 4 units: two pairs of parallel (level-shifted) trajectories.
        base1 = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        base2 = np.array([0.0, 1.0, 4.0, 1.0, 2.0])
        Ypre = np.vstack([base1, base1 + 10, base2, base2 + 10])
        members = np.array([0, 1, 2, 3])
        score, side_a, side_b = best_split(members, Ypre, max_size=2)
        # The optimal split groups {0 with 2} vs {1 with 3} OR {0,1} vs {2,3}?
        # The level-shifted duplicates are perfectly parallel, so any split that
        # pairs base1-types against base1-types gives a near-zero gap variance.
        assert score < 1e-6

    def test_enumerate_pairs_nonempty(self, panel):
        inp = prepare_pangeo_inputs(panel, "sales", "arm", "unit", "time")
        idx = inp.arm_units["A"]
        pairs = enumerate_candidate_pairs(idx, inp.Y, max_size=3)
        assert len(pairs) > 0
        for p in pairs:
            assert len(p["side_a"]) <= 3 and len(p["side_b"]) <= 3
            assert set(p["side_a"]).isdisjoint(p["side_b"])


# ----------------------------------------------------------------------
# Layer 2: data utilities
# ----------------------------------------------------------------------

class TestSetup:
    def test_prepare_inputs(self, panel):
        inp = prepare_pangeo_inputs(panel, "sales", "arm", "unit", "time")
        assert inp.Y.shape == (18, 120)
        assert set(inp.arm_units) == {"A", "B", "C"}
        assert all(idx.size == 6 for idx in inp.arm_units.values())

    def test_small_arm_rejected(self):
        # Two arms with one geo each: enough units overall, but no arm can
        # form a supergeo pair.
        df = make_seasonal_sales_panel(units_per_arm=1, arms=("A", "B"),
                                       T=20, seed=0)
        with pytest.raises(MlsynthConfigError):
            prepare_pangeo_inputs(df, "sales", "arm", "unit", "time")

    def test_missing_column_rejected(self, panel):
        with pytest.raises(MlsynthDataError):
            prepare_pangeo_inputs(panel, "nope", "arm", "unit", "time")


# ----------------------------------------------------------------------
# Layer 3: integration
# ----------------------------------------------------------------------

class TestDesign:
    def test_exact_cover_and_respects_Q(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": 3, "display_graphs": False}).fit()
        for arm, d in res.arm_designs.items():
            covered = []
            for p in d.pairs:
                assert len(p.treatment) <= 3 and len(p.control) <= 3   # Q
                covered.extend(p.treatment + p.control)
            # Exact cover: every arm unit assigned exactly once.
            assert sorted(covered) == sorted(
                [u for u in res.assignment if u.startswith(arm)]
            )
            assert len(covered) == len(set(covered)) == d.n_units

    def test_beats_random_parallelism(self, panel):
        """The designed split is far more parallel than random assignment."""
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": 3, "display_graphs": False}).fit()
        states = sorted(panel.unit.unique())
        times = sorted(panel.time.unique())
        Y = panel.pivot(index="unit", columns="time",
                        values="sales").loc[states, times].to_numpy()
        pos = {u: i for i, u in enumerate(states)}
        rng = np.random.default_rng(1)
        for arm, d in res.arm_designs.items():
            dgv = sum(gap_variance(Y[[pos[u] for u in p.treatment]].mean(0),
                                   Y[[pos[u] for u in p.control]].mean(0))
                      for p in d.pairs)
            arm_idx = [pos[u] for u in states
                       if panel[panel.unit == u].arm.iloc[0] == arm]
            rand = []
            for _ in range(200):
                perm = rng.permutation(arm_idx)
                h = len(perm) // 2
                rand.append(gap_variance(Y[perm[:h]].mean(0),
                                         Y[perm[h:2 * h]].mean(0)))
            assert dgv < np.median(rand)        # design beats typical random
            assert d.mean_parallelism_r2 > 0.5  # genuinely parallel

    def test_Q_one_recovers_matched_pairs(self, panel):
        """Q=1 forces singleton supergeos (classic matched pairs)."""
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time",
                      "max_supergeo_size": 1, "display_graphs": False}).fit()
        for d in res.arm_designs.values():
            for p in d.pairs:
                assert len(p.treatment) == 1 and len(p.control) == 1


# ----------------------------------------------------------------------
# Layer 4: public API contracts
# ----------------------------------------------------------------------

class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import PANGEO as _P
        assert _P is PANGEO

    def test_results_frozen(self, panel):
        res = PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                      "unitid": "unit", "time": "time", "display_graphs": False}).fit()
        assert isinstance(res, PangeoResults)
        with pytest.raises(Exception):
            res.max_supergeo_size = 99

    def test_invalid_Q_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            PANGEO({"df": panel, "outcome": "sales", "arm": "arm",
                    "unitid": "unit", "time": "time", "max_supergeo_size": 0, "display_graphs": False})
