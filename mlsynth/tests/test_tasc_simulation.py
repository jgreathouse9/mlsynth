"""Coverage tests for mlsynth.utils.tasc_helpers.simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth.utils.tasc_helpers.simulation import (
    TASCSample,
    _stable_transition,
    simulate_tasc_sample,
)


def test_stable_transition_spectral_radius():
    rng = np.random.default_rng(0)
    A = _stable_transition(5, 0.95, rng)
    assert A.shape == (5, 5)
    radius = np.abs(np.linalg.eigvals(A)).max()
    assert radius <= 0.95 + 1e-8


def test_simulate_tasc_sample_shapes_and_types():
    s = simulate_tasc_sample(N=6, T=12, T0=6, d=3, rng=np.random.default_rng(0))
    assert isinstance(s, TASCSample)
    assert s.Y.shape == (6, 12)
    assert s.A.shape == (3, 3)
    assert s.H.shape == (6, 3)
    assert s.Q.shape == (3, 3)
    assert s.R.shape == (6, 6)
    assert s.N == 6 and s.T == 12 and s.T0 == 6 and s.d == 3
    assert isinstance(s.df, pd.DataFrame)
    assert set(s.df.columns) == {"unit", "time", "y", "treat"}
    # 1 treated + (N-1) donors, each T rows.
    assert len(s.df) == 6 * 12
    assert s.df["unit"].nunique() == 6
    # treated unit gets treat=1 in post-period; donors always 0.
    treated = s.df[s.df["unit"] == "treated"]
    assert treated[treated["time"] >= 6]["treat"].eq(1).all()
    assert treated[treated["time"] < 6]["treat"].eq(0).all()
    donors = s.df[s.df["unit"] != "treated"]
    assert donors["treat"].eq(0).all()
    # Q and R are scaled identities.
    np.testing.assert_array_equal(s.Q, 0.01 * np.eye(3))
    np.testing.assert_array_equal(s.R, 0.1 * np.eye(6))


def test_simulate_tasc_sample_determinism():
    a = simulate_tasc_sample(N=5, T=8, T0=4, d=2, rng=np.random.default_rng(7))
    b = simulate_tasc_sample(N=5, T=8, T0=4, d=2, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.A, b.A)
    np.testing.assert_array_equal(a.H, b.H)
    pd.testing.assert_frame_equal(a.df, b.df)


def test_simulate_tasc_sample_default_rng_runs():
    # Exercises the `rng or np.random.default_rng()` default branch.
    s = simulate_tasc_sample(N=4, T=6, T0=3, d=2)
    assert s.Y.shape == (4, 6)


def test_simulate_tasc_sample_regime_scales():
    s = simulate_tasc_sample(N=4, T=5, T0=2, d=2, q_scale=0.1, r_scale=1.0,
                             rng=np.random.default_rng(1))
    np.testing.assert_array_equal(s.Q, 0.1 * np.eye(2))
    np.testing.assert_array_equal(s.R, 1.0 * np.eye(4))
