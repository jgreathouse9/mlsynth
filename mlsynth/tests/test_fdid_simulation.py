"""Coverage tests for mlsynth.utils.fdid_helpers.simulation (pure DGP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.fdid_helpers.simulation import (
    FDIDSample,
    _DGP_PARAMS,
    _factors,
    simulate_fdid_sample,
)


def test_factors_shape_and_burn():
    rng = np.random.default_rng(0)
    f = _factors(30, rng)
    assert f.shape == (30, 3)
    assert np.all(np.isfinite(f))


def test_factors_determinism_and_burn_branch():
    # default burn exercises t>=2 path; burn=1 exercises lag2=0 fallback at t==1
    a = _factors(10, np.random.default_rng(1))
    b = _factors(10, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    small = _factors(5, np.random.default_rng(1), burn=1)
    assert small.shape == (5, 3)


@pytest.mark.parametrize("dgp", [1, 2, 3, 4])
def test_simulate_each_dgp(dgp):
    rng = np.random.default_rng(0)
    s = simulate_fdid_sample(dgp, N=8, T1=6, T2=4, rng=rng)
    assert isinstance(s, FDIDSample)
    assert s.dgp == dgp
    T = s.T1 + s.T2
    assert s.Y_treated.shape == (T,)
    assert s.Y_controls.shape == (8, T)
    assert list(s.df.columns) == ["unit", "time", "y", "treat"]
    # one treated + N controls, each over T periods
    assert len(s.df) == (8 + 1) * T
    assert s.df["unit"].nunique() == 9
    # treat flag is 1 only for treated unit at t>=T1
    treated_rows = s.df[s.df["unit"] == "treated"]
    assert treated_rows.loc[treated_rows["time"] >= s.T1, "treat"].eq(1).all()
    assert treated_rows.loc[treated_rows["time"] < s.T1, "treat"].eq(0).all()
    assert s.df[s.df["unit"] != "treated"]["treat"].eq(0).all()


def test_loading_groups_differ_when_c2_differs():
    # dgp 2 has c1=1, c2=2 -> two control groups have different factor loadings
    s = simulate_fdid_sample(2, N=8, T1=6, T2=4, rng=np.random.default_rng(0))
    a0, c0, c1, c2 = _DGP_PARAMS[2]
    assert c1 != c2
    half = 8 // 2
    # the two halves should not be identical
    assert not np.allclose(s.Y_controls[:half], s.Y_controls[half:])


def test_determinism_same_seed():
    s1 = simulate_fdid_sample(1, N=6, T1=5, T2=3, rng=np.random.default_rng(42))
    s2 = simulate_fdid_sample(1, N=6, T1=5, T2=3, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(s1.Y_treated, s2.Y_treated)
    np.testing.assert_array_equal(s1.Y_controls, s2.Y_controls)
    pd.testing.assert_frame_equal(s1.df, s2.df)


def test_default_rng_path_runs():
    # rng=None branch
    s = simulate_fdid_sample(1, N=4, T1=4, T2=2)
    assert s.Y_treated.shape == (6,)


def test_invalid_dgp_raises():
    with pytest.raises(ValueError):
        simulate_fdid_sample(5)


def test_frozen_dataclass_immutable():
    s = simulate_fdid_sample(1, N=4, T1=4, T2=2, rng=np.random.default_rng(0))
    with pytest.raises(Exception):
        s.dgp = 2  # type: ignore[misc]
