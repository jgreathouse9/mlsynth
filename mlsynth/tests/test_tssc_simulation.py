"""Coverage tests for mlsynth.utils.tssc_helpers.simulation (pure DGP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.tssc_helpers.simulation import (
    TSSCSample,
    _factors,
    simulate_tssc_sample,
)


def test_factors_shape_and_recursions():
    rng = np.random.default_rng(0)
    f = _factors(20, rng)
    assert f.shape == (20, 3)
    assert np.all(np.isfinite(f))
    # initial values are zero by construction
    assert f[0, 0] == 0.0
    assert f[0, 1] == 0.0


def test_factors_determinism():
    a = _factors(15, np.random.default_rng(3))
    b = _factors(15, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_simulate_structure():
    rng = np.random.default_rng(0)
    s = simulate_tssc_sample(T1=8, T2=4, N_co=5, alpha=1.0, rng=rng)
    assert isinstance(s, TSSCSample)
    T = s.T1 + s.T2
    assert s.y_treated.shape == (T,)
    assert s.donors.shape == (T, 5)
    assert s.factors.shape == (T, 3)
    assert s.N_co == 5
    assert list(s.df.columns) == ["unit", "time", "y", "treat"]
    # one treated + N_co donors over T periods
    assert len(s.df) == (5 + 1) * T
    assert s.df["unit"].nunique() == 6
    treated_rows = s.df[s.df["unit"] == "treated"]
    assert treated_rows.loc[treated_rows["time"] >= s.T1, "treat"].eq(1).all()
    assert treated_rows.loc[treated_rows["time"] < s.T1, "treat"].eq(0).all()
    assert s.df[s.df["unit"] != "treated"]["treat"].eq(0).all()


def test_alpha_shift():
    rng0 = np.random.default_rng(11)
    rng1 = np.random.default_rng(11)
    s_a = simulate_tssc_sample(T1=6, T2=3, N_co=3, alpha=0.0, rng=rng0)
    s_b = simulate_tssc_sample(T1=6, T2=3, N_co=3, alpha=5.0, rng=rng1)
    # alpha is an additive constant on every outcome
    np.testing.assert_allclose(s_b.y_treated - s_a.y_treated, 5.0)


def test_determinism_same_seed():
    s1 = simulate_tssc_sample(T1=6, T2=3, N_co=4, rng=np.random.default_rng(9))
    s2 = simulate_tssc_sample(T1=6, T2=3, N_co=4, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(s1.y_treated, s2.y_treated)
    np.testing.assert_array_equal(s1.donors, s2.donors)
    np.testing.assert_array_equal(s1.factors, s2.factors)
    pd.testing.assert_frame_equal(s1.df, s2.df)


def test_default_rng_path_runs():
    s = simulate_tssc_sample(T1=5, T2=2, N_co=3)
    assert s.y_treated.shape == (7,)


def test_default_dimensions():
    s = simulate_tssc_sample(rng=np.random.default_rng(0))
    assert s.T1 == 76 and s.T2 == 34 and s.N_co == 10
    assert s.y_treated.shape == (110,)
    assert s.donors.shape == (110, 10)


def test_frozen_dataclass_immutable():
    s = simulate_tssc_sample(T1=4, T2=2, N_co=2, rng=np.random.default_rng(0))
    with pytest.raises(Exception):
        s.N_co = 99  # type: ignore[misc]
