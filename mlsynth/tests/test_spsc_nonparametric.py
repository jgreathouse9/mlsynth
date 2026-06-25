"""Tests for the nonparametric (sieve-basis) Single Proxy Synthetic Control.

The ``basis_degree`` knob generalises the reference's ``Y.basis`` from the linear
single proxy (``degree=1``) to a polynomial sieve (``degree>=2``). These tests
pin three things: (1) ``degree=1`` is bit-identical to the legacy linear fit;
(2) the sieve runs end-to-end, over-identifies the bridge, and still recovers a
known effect; and (3) the config / variant plumbing exposes it.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.proximal_helpers.spsc.estimation import (
    _poly_basis,
    estimate_spsc,
)


def _toy_panel(seed: int = 0, T0: int = 40, T1: int = 20, N: int = 6,
               effect: float = 2.0):
    """Small linear two-factor panel with a constant post-period effect.

    Treated loads on the mean of the donor loadings, so a convex-ish donor
    combination reproduces its factor -- the regime SPSC identifies.
    """
    rng = np.random.default_rng(seed)
    T = T0 + T1
    f = np.cumsum(rng.standard_normal((T, 2)), axis=0) / 5.0      # smooth factors
    load = rng.uniform(0.2, 0.8, size=(N, 2))
    donors = f @ load.T + 0.05 * rng.standard_normal((T, N))
    y = f @ load.mean(0) + 0.05 * rng.standard_normal(T)
    y[T0:] += effect
    return y, donors, T0


class TestPolyBasis:
    def test_degree_one_is_identity_column(self):
        u = np.array([1.0, -2.0, 3.0])
        np.testing.assert_array_equal(_poly_basis(u, 1), u.reshape(-1, 1))

    def test_degree_three_stacks_powers(self):
        u = np.array([2.0, 3.0])
        b = _poly_basis(u, 3)
        assert b.shape == (2, 3)
        np.testing.assert_allclose(b[:, 0], u)
        np.testing.assert_allclose(b[:, 1], u ** 2)
        np.testing.assert_allclose(b[:, 2], u ** 3)

    def test_flattens_input(self):
        assert _poly_basis(np.ones((5, 1)), 2).shape == (5, 2)


class TestBasisDegreeInEstimation:
    @pytest.mark.parametrize("detrend", [False, True])
    def test_degree_one_matches_legacy_default(self, detrend):
        """basis_degree=1 reproduces the default call bit-for-bit."""
        y, donors, T0 = _toy_panel(seed=1)
        legacy = estimate_spsc(y, donors, T0, detrend=detrend)
        deg1 = estimate_spsc(y, donors, T0, detrend=detrend, basis_degree=1)
        for a, b in zip(legacy, deg1):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    @pytest.mark.parametrize("degree", [2, 3])
    @pytest.mark.parametrize("detrend", [False, True])
    def test_sieve_runs_and_recovers_effect(self, degree, detrend):
        """The nonparametric sieve returns a finite ATT/SE near the true effect."""
        y, donors, T0 = _toy_panel(seed=2, effect=2.0)
        cf, gamma, att, se, trend, lam, _path, _pse = estimate_spsc(
            y, donors, T0, detrend=detrend, basis_degree=degree)
        assert np.isfinite(att) and np.isfinite(se)
        assert cf.shape == y.shape and gamma.shape == (donors.shape[1],)
        assert abs(att - 2.0) < 0.5

    def test_higher_degree_widens_the_moment_matrix(self):
        """A larger sieve adds GYb moment columns (over-identification)."""
        from mlsynth.utils.proximal_helpers.spsc.estimation import (
            _build_detrend_matrix, _effect, _psi, _LAMBDA_GRID,
        )
        y, donors, T0 = _toy_panel(seed=3)
        T = y.shape[0]
        A = (np.arange(T) >= T0).astype(float)
        B_post = A.reshape(-1, 1)
        D = _build_detrend_matrix(T0, T, 5)
        widths = []
        for degree in (1, 3):
            theta = _effect(y, donors, A, D, None, _LAMBDA_GRID, B_post, degree)
            widths.append(_psi(theta, y, donors, A, D, B_post, degree).shape[1])
        assert widths[1] == widths[0] + 2     # two extra GYb columns at degree 3

    def test_invalid_degree_raises(self):
        y, donors, T0 = _toy_panel(seed=4)
        with pytest.raises(ValueError):
            estimate_spsc(y, donors, T0, basis_degree=0)
