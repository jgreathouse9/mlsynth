"""Coverage tests for mlsynth.utils.mlsc_helpers.simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth.utils.mlsc_helpers.simulation import MLSCSample, simulate_mlsc_sample


def test_simulate_mlsc_sample_shapes_and_types():
    s = simulate_mlsc_sample(N_states=4, counties_per_state=3, T=6,
                             rng=np.random.default_rng(0))
    assert isinstance(s, MLSCSample)
    assert s.data_s.shape == (4, 6)
    assert s.data_c.shape == (12, 6)
    np.testing.assert_array_equal(s.n_c, np.full(4, 3))
    assert len(s.w_c) == 4
    for w in s.w_c:
        np.testing.assert_allclose(w.sum(), 1.0)
        np.testing.assert_allclose(w, np.full(3, 1.0 / 3))
    # Default treated_t is T - 1.
    assert s.idx == 0
    assert s.t == 5
    assert isinstance(s.df_agg, pd.DataFrame)
    assert isinstance(s.df_disagg, pd.DataFrame)
    assert set(s.df_agg.columns) == {"state", "time", "y", "treated"}
    assert set(s.df_disagg.columns) == {"county", "state", "time", "y", "treated"}
    assert len(s.df_agg) == 4 * 6
    assert len(s.df_disagg) == 12 * 6


def test_simulate_mlsc_aggregation_matches_weighted_mean():
    s = simulate_mlsc_sample(N_states=3, counties_per_state=4, T=5,
                             rng=np.random.default_rng(2))
    # data_s is the within-state equal-weight average of data_c.
    expected = s.data_c.reshape(3, 4, 5).mean(axis=1)
    np.testing.assert_allclose(s.data_s, expected)


def test_simulate_mlsc_treatment_indicators():
    s = simulate_mlsc_sample(N_states=3, counties_per_state=2, T=4,
                             treated_idx=1, treated_t=2,
                             rng=np.random.default_rng(0))
    assert s.idx == 1 and s.t == 2
    agg = s.df_agg
    treated_rows = agg[(agg["state"] == "s1") & (agg["time"] >= 2)]
    assert treated_rows["treated"].eq(1).all()
    assert agg[agg["state"] != "s1"]["treated"].eq(0).all()
    assert agg[(agg["state"] == "s1") & (agg["time"] < 2)]["treated"].eq(0).all()
    dis = s.df_disagg
    # Every county inside treated state mirrors the indicator.
    assert dis[(dis["state"] == "s1") & (dis["time"] >= 2)]["treated"].eq(1).all()
    assert dis[dis["state"] != "s1"]["treated"].eq(0).all()


def test_simulate_mlsc_determinism():
    a = simulate_mlsc_sample(N_states=3, counties_per_state=2, T=4,
                             rng=np.random.default_rng(11))
    b = simulate_mlsc_sample(N_states=3, counties_per_state=2, T=4,
                             rng=np.random.default_rng(11))
    np.testing.assert_array_equal(a.data_s, b.data_s)
    np.testing.assert_array_equal(a.data_c, b.data_c)
    pd.testing.assert_frame_equal(a.df_agg, b.df_agg)
    pd.testing.assert_frame_equal(a.df_disagg, b.df_disagg)


def test_simulate_mlsc_default_rng_runs():
    # Exercises the `rng or np.random.default_rng()` and treated_t=None branches.
    s = simulate_mlsc_sample(N_states=2, counties_per_state=2, T=3)
    assert s.t == 2  # T - 1
    assert s.data_s.shape == (2, 3)
