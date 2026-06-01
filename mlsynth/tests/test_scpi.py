"""Tests for the CFPT/scpi uncertainty quantification module.

Cattaneo, Feng, Palomba & Titiunik (2025), arXiv:2210.05026.

Layered per agents/agents_tests.md:

* Layer 1 (numerical helpers): conditional-moment estimation; sub-Gaussian
  half-width; the in-sample closed-form ellipsoid bound.
* Layer 2 (predictand bands): the four predictands aggregate correctly and
  the interval algebra matches the paper.
* Layer 4 (API contracts): frozen results, significance flag.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mlsynth.utils.scpi_helpers import (
    SCPIBand,
    SCPIResults,
    conditional_moments,
    in_sample_band_gaussian,
    out_of_sample_intervals,
    unit_moments,
)


# ----------------------------------------------------------------------
# Layer 1: numerical helpers
# ----------------------------------------------------------------------

class TestMoments:
    def test_zero_mean_option(self):
        r = np.array([1.0, -1.0, 2.0, -2.0])
        mu0, s0 = conditional_moments(r, assume_zero_mean=True)
        mu1, s1 = conditional_moments(r, assume_zero_mean=False)
        assert mu0 == 0.0
        assert abs(mu1 - 0.0) < 1e-12          # symmetric data
        assert s0 == pytest.approx(np.mean(r ** 2))

    def test_unit_moments_shapes(self):
        R = np.random.default_rng(0).normal(size=(30, 4))
        mu, sigma = unit_moments(R, ["a", "b", "c", "d"])
        assert set(mu) == {"a", "b", "c", "d"}
        assert all(s >= 0 for s in sigma.values())


class TestInSample:
    def test_closed_form_brackets_zero_symmetric(self):
        # With a symmetric Gaussian draw the in-sample band straddles 0.
        rng = np.random.default_rng(1)
        J0 = 5
        Yn = rng.normal(size=(40, J0))
        Q = Yn.T @ Yn
        Sigma = np.eye(J0) * 0.5
        a = rng.normal(size=J0)
        m_in, mbar_in = in_sample_band_gaussian(
            a, Q, Sigma, alpha_in=0.1, n_sim=4000, random_state=2)
        assert m_in < 0 < mbar_in
        assert mbar_in > m_in


# ----------------------------------------------------------------------
# Layer 2: predictand bands
# ----------------------------------------------------------------------

def _toy(L=5, m=3, T0=20, att=2.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.normal(scale=sigma, size=(T0, m))
    effects = att + rng.normal(scale=sigma, size=(L, m))
    units = [f"u{j}" for j in range(m)]
    periods = list(range(L))
    return effects, pre, units, periods


class TestPredictands:
    def test_family_sizes_and_points(self):
        effects, pre, units, periods = _toy()
        res = out_of_sample_intervals(effects, pre, units, periods, alpha=0.1)
        assert isinstance(res, SCPIResults)
        assert len(res.tsus) == len(units) * len(periods)
        assert len(res.taus) == len(units)
        assert len(res.tsua) == len(periods)
        # TAUA point == grand mean of effects
        assert res.taua.point == pytest.approx(effects.mean())
        # TSUA point at period k == cross-unit mean at k
        assert res.tsua[0].point == pytest.approx(effects[0, :].mean())
        # TAUS point for a unit == its time-mean
        assert res.taus["u0"].point == pytest.approx(effects[:, 0].mean())

    def test_interval_algebra(self):
        # band = [point - (mu + h), point - (mu - h)]
        effects, pre, units, periods = _toy(seed=3)
        res = out_of_sample_intervals(effects, pre, units, periods, alpha=0.1)
        b = res.tsus[("u0", 0)]
        m_out, mbar_out = b.out_sample
        assert b.lower == pytest.approx(b.point - mbar_out)
        assert b.upper == pytest.approx(b.point - m_out)

    def test_iid_tighter_than_general(self):
        effects, pre, units, periods = _toy(L=8, seed=4)
        iid = out_of_sample_intervals(effects, pre, units, periods,
                                      alpha=0.1, time_dependence="iid")
        gen = out_of_sample_intervals(effects, pre, units, periods,
                                      alpha=0.1, time_dependence="general")
        w_iid = iid.taua.upper - iid.taua.lower
        w_gen = gen.taua.upper - gen.taua.lower
        assert w_gen > w_iid                    # general makes no shrink

    def test_simultaneous_wider_than_pointwise(self):
        effects, pre, units, periods = _toy(L=6, seed=5)
        res = out_of_sample_intervals(effects, pre, units, periods, alpha=0.1)
        pt = res.tsus[("u0", 0)]
        sim = res.simultaneous["u0"][0]
        assert (sim.upper - sim.lower) > (pt.upper - pt.lower)


# ----------------------------------------------------------------------
# Layer 4: API contracts
# ----------------------------------------------------------------------

class TestAPI:
    def test_frozen_and_significance(self):
        effects, pre, units, periods = _toy(att=10.0, sigma=0.1, seed=6)
        res = out_of_sample_intervals(effects, pre, units, periods, alpha=0.1)
        assert res.taua.significant is True     # large effect, small noise
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.taua.point = 0.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.method = "x"
