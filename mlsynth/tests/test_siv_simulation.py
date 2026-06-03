"""Coverage tests for mlsynth.utils.siv_helpers.simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.siv_helpers.simulation import (
    SIVSample,
    simulate_siv_sample,
)


def test_default_shapes_and_types():
    s = simulate_siv_sample(rng=np.random.default_rng(0))
    assert isinstance(s, SIVSample)
    assert s.Y.shape == (26, 16)
    assert s.R.shape == (26, 16)
    assert s.Z.shape == (26, 16)
    assert s.J == 26 and s.T == 16 and s.T0 == 10
    assert isinstance(s.df, pd.DataFrame)
    assert set(s.df.columns) == {"unit", "time", "y", "r", "z"}
    assert len(s.df) == 26 * 16


def test_pre_period_treatment_and_instrument_zero():
    # R and Z are gated by post = (t >= T0); pre-period must be exactly zero.
    s = simulate_siv_sample(J=4, T=8, T0=5, rng=np.random.default_rng(1))
    assert np.all(s.R[:, :5] == 0.0)
    assert np.all(s.Z[:, :5] == 0.0)
    # post-period generally non-zero
    assert np.any(s.R[:, 5:] != 0.0)


def test_default_rng_branch():
    s = simulate_siv_sample(J=4, T=6, T0=3)
    assert s.Y.shape == (4, 6)


@pytest.mark.parametrize("r", [0.5, 0.7, 0.9])
def test_correlation_sweep(r):
    s = simulate_siv_sample(J=5, T=8, T0=4, r=r, rng=np.random.default_rng(2))
    assert s.Y.shape == (5, 8)


def test_determinism():
    a = simulate_siv_sample(J=6, T=10, T0=5, rng=np.random.default_rng(11))
    b = simulate_siv_sample(J=6, T=10, T0=5, rng=np.random.default_rng(11))
    np.testing.assert_array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.R, b.R)
    np.testing.assert_array_equal(a.Z, b.Z)
    pd.testing.assert_frame_equal(a.df, b.df)


def test_df_values_match_matrices():
    s = simulate_siv_sample(J=3, T=5, T0=2, rng=np.random.default_rng(5))
    row = s.df[(s.df["unit"] == "u01") & (s.df["time"] == 3)].iloc[0]
    assert row["y"] == pytest.approx(s.Y[1, 3])
    assert row["r"] == pytest.approx(s.R[1, 3])
    assert row["z"] == pytest.approx(s.Z[1, 3])
