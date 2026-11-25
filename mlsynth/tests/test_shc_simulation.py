"""Coverage tests for mlsynth.utils.shc_helpers.simulation (pure DGP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

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


# --------------------------------------------------------------------------- #
# The post window: n <= m
#
# T_o = m(4h + 1), and the last macro-segment (the treated shape) has only its
# cosine half, which ends at T_o + m - 1. Chen, Yang & Yang (2024, Sec. 3.1,
# Eqs. 35-37) define the latent component only through that period, with
# n = m in every design (25/25, 50/50). A post window longer than m used to
# leave cells T_o + m, ..., T_o + n - 1 as whatever np.empty held.
# --------------------------------------------------------------------------- #
def _w_f(h):
    return (1.0,) + (0.0,) * (h - 1)


def test_a_post_window_longer_than_m_is_refused():
    # m = 5, h = 2, n = 18 is the panel that left 13 cells unwritten.
    with pytest.raises(ValueError, match=r"n=18.*m=5.*T_o \+ m - 1"):
        simulate_shc_latent(m=5, h=2, n=18, w_f=_w_f(2))


def test_a_post_window_one_past_m_is_refused():
    with pytest.raises(ValueError, match=r"n=8.*m=7"):
        simulate_shc_latent(m=7, h=3, n=8, w_f=_w_f(3))


def test_the_panel_inherits_the_refusal():
    with pytest.raises(ValueError, match=r"n=7.*m=5"):
        simulate_shc_panel(m=5, h=3, n=7, w_f=_w_f(3))


@pytest.mark.parametrize("m, h, n", [(5, 3, 5), (10, 4, 10), (1, 1, 1)])
def test_a_post_window_equal_to_m_is_fully_finite(m, h, n):
    ell, T_o, _N = simulate_shc_latent(m=m, h=h, n=n, w_f=_w_f(h))
    assert ell.shape == (T_o + n,)
    assert np.all(np.isfinite(ell))


@pytest.mark.parametrize("m, h, n", [(5, 3, 1), (10, 4, 3), (25, 4, 4)])
def test_a_post_window_shorter_than_m_is_fully_finite(m, h, n):
    ell, T_o, _N = simulate_shc_latent(m=m, h=h, n=n, w_f=_w_f(h))
    assert ell.shape == (T_o + n,)
    assert np.all(np.isfinite(ell))


def test_the_builder_raises_on_a_cell_it_did_not_write():
    """The construction itself reports unwritten cells.

    The public refusal keeps n <= m, so the builder is called here directly
    with n > m: its tail is never written and must be reported, not returned.
    Built on ``np.empty`` the tail would hold leftover memory and pass.
    """
    from mlsynth.utils.shc_helpers.simulation import _latent_series

    m, h, n = 5, 2, 18
    with pytest.raises(RuntimeError, match=r"unwritten.*index 50"):
        _latent_series(m=m, h=h, n=n, alpha=np.zeros(h),
                       Pis=np.full(h, 10.0), w_f=np.asarray(_w_f(h)))


def test_the_builder_writes_every_cell_when_n_is_at_most_m():
    from mlsynth.utils.shc_helpers.simulation import _latent_series

    m, h = 5, 2
    ell = _latent_series(m=m, h=h, n=m, alpha=np.zeros(h),
                         Pis=np.full(h, 10.0), w_f=np.asarray(_w_f(h)))
    assert ell.shape == (m * (4 * h + 1) + m,)
    assert np.all(np.isfinite(ell))


@settings(deadline=None)
@given(m=st.integers(1, 12), h=st.integers(1, 5), data=st.data(),
       seed=st.integers(0, 2**31 - 1), regular=st.booleans())
def test_every_cell_is_written_and_a_longer_window_only_appends(
        m, h, data, seed, regular):
    n = data.draw(st.integers(1, m), label="n")
    w_f = _w_f(h)
    ell, T_o, N = simulate_shc_latent(m=m, h=h, n=n, w_f=w_f,
                                      regular=regular, seed=seed)
    assert T_o == m * (4 * h + 1)
    assert len(ell) == T_o + n
    assert np.all(np.isfinite(ell))
    # The latent path does not depend on n: the n-period window is a prefix
    # of the m-period one (T_o + min(n, m) = T_o + n cells shared).
    longest, _, _ = simulate_shc_latent(m=m, h=h, n=m, w_f=w_f,
                                        regular=regular, seed=seed)
    np.testing.assert_array_equal(ell, longest[:T_o + n])


@settings(deadline=None)
@given(m=st.integers(1, 12), h=st.integers(1, 5), data=st.data(),
       seed=st.integers(0, 2**31 - 1), regular=st.booleans())
def test_every_post_window_longer_than_m_is_refused(m, h, data, seed, regular):
    n = data.draw(st.integers(m + 1, 3 * m), label="n")
    with pytest.raises(ValueError, match=rf"n={n}\b.*m={m}\b"):
        simulate_shc_latent(m=m, h=h, n=n, w_f=_w_f(h),
                            regular=regular, seed=seed)
