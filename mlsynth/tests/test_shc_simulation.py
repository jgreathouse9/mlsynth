"""Coverage tests for mlsynth.utils.shc_helpers.simulation (pure DGP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.shc_helpers.simulation import (
    _cosine_deriv,
    _cosine_shape,
    _hermite_cubic,
    simulate_shc_latent,
    simulate_shc_panel,
)


def test_cosine_shape_and_deriv():
    u = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(_cosine_shape(u, 2.0, 4.0), 2.0 + np.cos(u / 4.0))
    np.testing.assert_allclose(_cosine_deriv(u, 4.0), -np.sin(u / 4.0) / 4.0)


def test_hermite_cubic_matches_endpoints():
    width = 5.0
    v0, d0, v1, d1 = 1.0, 0.5, -2.0, -0.3
    # value at s=0 must equal v0
    val0 = _hermite_cubic(np.array([0.0]), width, v0, d0, v1, d1)[0]
    assert val0 == pytest.approx(v0)
    # value at s=width must equal v1
    val1 = _hermite_cubic(np.array([width]), width, v0, d0, v1, d1)[0]
    assert val1 == pytest.approx(v1)


def test_latent_shapes_and_dims_regular():
    m, h, n = 25, 4, 25
    ell, T_o, N = simulate_shc_latent(m=m, h=h, n=n, regular=True, seed=0)
    assert T_o == m * (4 * h + 1)  # 425
    assert N == T_o - n - (m - 1)
    assert ell.shape == (T_o + n,)
    assert np.all(np.isfinite(ell))


def test_latent_determinism_regular():
    a = simulate_shc_latent(seed=3, regular=True)
    b = simulate_shc_latent(seed=3, regular=True)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_latent_irregular_branch_and_determinism():
    a, To_a, N_a = simulate_shc_latent(seed=7, regular=False)
    b, To_b, N_b = simulate_shc_latent(seed=7, regular=False)
    np.testing.assert_array_equal(a, b)
    assert (To_a, N_a) == (To_b, N_b)
    # irregular differs from regular for the same seed
    reg, _, _ = simulate_shc_latent(seed=7, regular=True)
    assert not np.array_equal(a, reg)


def test_latent_bad_w_f_length_raises():
    with pytest.raises(ValueError):
        simulate_shc_latent(h=4, w_f=(1.0, 0.0))


def test_latent_alternate_dims():
    # m=50 dimension from the docstring
    ell, T_o, N = simulate_shc_latent(m=50, h=4, n=25, seed=1)
    assert T_o == 850
    assert ell.shape == (T_o + 25,)


def test_panel_structure_and_info():
    df, info = simulate_shc_panel(m=25, h=4, n=25, sigma=0.1, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["unit", "time", "y", "treated"]
    T_o = info["T_o"]
    n = info["n"]
    assert len(df) == T_o + n
    # treated flag is 1 exactly for t > T_o
    assert df.loc[df["time"] <= T_o, "treated"].eq(0).all()
    assert df.loc[df["time"] > T_o, "treated"].eq(1).all()
    assert info["latent"].shape == (T_o + n,)
    assert info["latent_post"].shape == (n,)
    assert info["latent_pre_block"].shape == (info["m"],)
    np.testing.assert_array_equal(info["time"], np.arange(1, T_o + n + 1))


def test_panel_determinism():
    d1, i1 = simulate_shc_panel(seed=5)
    d2, i2 = simulate_shc_panel(seed=5)
    pd.testing.assert_frame_equal(d1, d2)
    np.testing.assert_array_equal(i1["latent"], i2["latent"])


def test_panel_irregular():
    df, info = simulate_shc_panel(seed=2, regular=False)
    assert np.all(np.isfinite(df["y"].to_numpy()))
    assert info["N"] == info["T_o"] - info["n"] - (info["m"] - 1)
