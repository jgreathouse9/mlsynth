"""Coverage + correctness tests for the proximal data-generating processes."""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.proximal_helpers.simulation import (
    DRSimSample,
    PIOIDSimSample,
    ProxSurrogateSimSample,
    SPSCSimSample,
    _ar1_noise,
    _ifem_factor,
    simulate_dr_proximal_normal,
    simulate_pioid_linear,
    simulate_proximal_surrogates,
    simulate_spsc_ifem,
)


class TestSimulateSpscIfem:
    def test_default_shapes_and_fields(self):
        s = simulate_spsc_ifem(seed=0)
        assert isinstance(s, SPSCSimSample)
        T = s.T0 + 50
        assert s.y.shape == (T,)
        assert s.donors.shape == (T, 16)          # 8 valid + 8 invalid
        assert s.T0 == 50
        assert s.n_valid == 8 and s.n_invalid == 8
        assert s.true_effect.shape == (50,)
        # true_att is the mean of the realised per-period effects
        np.testing.assert_allclose(s.true_att, s.true_effect.mean())
        # effect sits near the nominal 3.0 (small mean-zero perturbation)
        assert abs(s.true_att - 3.0) < 0.1

    def test_custom_dimensions(self):
        s = simulate_spsc_ifem(T0=30, T1=20, n_valid=5, n_invalid=3, seed=1)
        assert s.y.shape == (50,)
        assert s.donors.shape == (50, 8)
        assert s.T0 == 30 and s.n_valid == 5 and s.n_invalid == 3
        assert s.true_effect.shape == (20,)

    def test_zero_invalid_donors(self):
        s = simulate_spsc_ifem(n_invalid=0, seed=2)
        assert s.donors.shape[1] == 8

    def test_true_att_override_shifts_post(self):
        s0 = simulate_spsc_ifem(true_att=0.0, seed=3)
        s5 = simulate_spsc_ifem(true_att=5.0, seed=3)
        # pre-period identical; post-period shifted by exactly 5.0
        np.testing.assert_allclose(s5.y[: s5.T0], s0.y[: s0.T0])
        np.testing.assert_allclose(s5.y[s5.T0:], s0.y[s0.T0:] + 5.0)

    def test_seed_is_deterministic(self):
        a = simulate_spsc_ifem(seed=7).y
        b = simulate_spsc_ifem(seed=7).y
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(3)
        s = simulate_spsc_ifem(rng=rng, seed=999, n_valid=4, n_invalid=2)
        assert s.donors.shape[1] == 6

    def test_corr_one_couples_treated_and_donor_errors(self):
        # with corr=1 the idiosyncratic errors share the common shock entirely;
        # the call must still produce a finite, correctly shaped draw
        s = simulate_spsc_ifem(corr=1.0, seed=5)
        assert np.all(np.isfinite(s.y)) and np.all(np.isfinite(s.donors))

    @pytest.mark.parametrize("kwargs", [
        {"n_valid": 0}, {"n_valid": 9},
        {"n_invalid": -1},
        {"T0": 0}, {"T1": 0},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            simulate_spsc_ifem(seed=0, **kwargs)


class TestIfemFactor:
    def test_shape_includes_warmup_rows(self):
        baseline = np.ones(52)
        f = _ifem_factor(50, np.random.default_rng(0), rho=0.5, sd=0.1,
                         baseline=baseline)
        assert f.shape == (52, 2)

    def test_baseline_enters_additively(self):
        # zero innovation scale -> factor equals the baseline (broadcast to 2 cols)
        baseline = np.arange(12, dtype=float)
        f = _ifem_factor(10, np.random.default_rng(0), rho=0.5, sd=0.0,
                         baseline=baseline)
        np.testing.assert_allclose(f, np.column_stack([baseline, baseline]))


class TestSimulateDrProximalNormal:
    def test_default_shapes_and_fields(self):
        s = simulate_dr_proximal_normal(T=200, seed=0)
        assert isinstance(s, DRSimSample)
        assert s.y.shape == (200,)
        assert s.donor_outcomes.shape == (200, 2)
        assert s.donor_proxies.shape == (200, 2)
        assert s.T0 == 100
        assert s.true_att == 2.0
        assert s.n_confounders == 2
        assert s.misspecified is False

    def test_custom_dimensions(self):
        s = simulate_dr_proximal_normal(T=300, n_confounders=3, true_att=1.5, seed=1)
        assert s.T0 == 150
        assert s.donor_outcomes.shape == (300, 3)
        assert s.donor_proxies.shape == (300, 3)
        assert s.true_att == 1.5 and s.n_confounders == 3

    def test_treatment_shift_in_post_mean(self):
        # the post-period mean exceeds the pre-period mean by ~true_att (the only
        # systematic pre/post difference; confounding is stationary)
        s = simulate_dr_proximal_normal(T=4000, true_att=2.0, seed=2)
        diff = s.y[s.T0:].mean() - s.y[: s.T0].mean()
        assert abs(diff - 2.0) < 0.5

    def test_misspecify_injects_nonlinear_signal(self):
        lin = simulate_dr_proximal_normal(T=2000, misspecify=False, seed=3)
        mis = simulate_dr_proximal_normal(T=2000, misspecify=True, seed=3)
        assert mis.misspecified is True
        # the proxies are drawn identically; only Y differs (the U^2 term)
        np.testing.assert_array_equal(lin.donor_outcomes, mis.donor_outcomes)
        assert not np.allclose(lin.y, mis.y)

    def test_seed_is_deterministic(self):
        a = simulate_dr_proximal_normal(T=300, seed=7).y
        b = simulate_dr_proximal_normal(T=300, seed=7).y
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(5)
        s = simulate_dr_proximal_normal(T=120, n_confounders=4, rng=rng, seed=999)
        assert s.donor_outcomes.shape == (120, 4)

    @pytest.mark.parametrize("kwargs", [{"T": 1}, {"n_confounders": 0}])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            simulate_dr_proximal_normal(seed=0, **kwargs)


class TestSimulateProximalSurrogates:
    def test_default_shapes_and_fields(self):
        s = simulate_proximal_surrogates(T=200, seed=0)
        assert isinstance(s, ProxSurrogateSimSample)
        assert s.y.shape == (200,)
        assert s.donor_outcomes.shape == (200, 1)
        assert s.donor_proxies.shape == (200, 1)
        assert s.surrogate_outcomes.shape == (200, 1)
        assert s.surrogate_proxies.shape == (200, 1)
        assert s.T0 == 100
        assert s.n_donor_factors == 1 and s.n_surrogate_factors == 1
        # the effect factor rho has mean ~1, so the realised ATT sits near 1
        assert abs(s.true_att - 1.0) < 0.6

    def test_custom_dimensions(self):
        s = simulate_proximal_surrogates(
            T=120, T0=80, n_donor_factors=2, n_surrogate_factors=3, seed=1)
        assert s.T0 == 80
        assert s.donor_outcomes.shape == (120, 2)
        assert s.donor_proxies.shape == (120, 2)
        assert s.surrogate_outcomes.shape == (120, 3)
        assert s.surrogate_proxies.shape == (120, 3)

    def test_donor_and_proxy_share_trend_not_noise(self):
        # W and Z0 are two error-prone copies of the same trending factor:
        # both trend up, but they are not identical (independent noise)
        s = simulate_proximal_surrogates(T=400, seed=2)
        assert s.donor_outcomes[-50:].mean() > s.donor_outcomes[:50].mean()
        assert not np.allclose(s.donor_outcomes, s.donor_proxies)

    def test_iid_errors_differ_from_ar(self):
        kw = dict(T=200, seed=3)
        ar = simulate_proximal_surrogates(ar_errors=True, **kw).y
        iid = simulate_proximal_surrogates(ar_errors=False, **kw).y
        assert not np.allclose(ar, iid)

    def test_seed_is_deterministic(self):
        a = simulate_proximal_surrogates(T=150, seed=7).y
        b = simulate_proximal_surrogates(T=150, seed=7).y
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(5)
        s = simulate_proximal_surrogates(T=120, rng=rng, seed=999)
        assert s.y.shape == (120,)

    @pytest.mark.parametrize("kwargs", [
        {"T": 1}, {"n_donor_factors": 0}, {"n_surrogate_factors": 0},
        {"T": 100, "T0": 0}, {"T": 100, "T0": 100},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            simulate_proximal_surrogates(seed=0, **kwargs)


class TestSimulatePioidLinear:
    def test_default_shapes_and_fields(self):
        s = simulate_pioid_linear(seed=0)
        assert isinstance(s, PIOIDSimSample)
        T = 2 * 80
        assert s.y.shape == (T,)
        # n_units=7 -> 6 controls: n_Z=3 proxies, n_W=3 donors
        assert s.donor_outcomes.shape == (T, 3)
        assert s.donor_proxies.shape == (T, 3)
        assert s.T0 == 80
        assert s.true_att == 2.0
        assert s.n_units == 7

    def test_custom_dimensions(self):
        s = simulate_pioid_linear(n_units=11, t0=30, seed=1)
        # 10 controls: n_Z=5 proxies, n_W=5 donors; T = 2 * t0
        assert s.y.shape == (60,)
        assert s.donor_proxies.shape == (60, 5)
        assert s.donor_outcomes.shape == (60, 5)
        assert s.T0 == 30 and s.n_units == 11

    def test_smallest_grid(self):
        # n_units=5 -> 4 controls: n_Z=2, n_W=2 (the reference grid's floor)
        s = simulate_pioid_linear(n_units=5, t0=30, seed=2)
        assert s.donor_proxies.shape == (60, 2)
        assert s.donor_outcomes.shape == (60, 2)

    def test_true_att_override_shifts_post(self):
        s0 = simulate_pioid_linear(true_att=0.0, seed=3)
        s5 = simulate_pioid_linear(true_att=5.0, seed=3)
        # the effect is added only to the treated series' post-period; the pre
        # period and the donor / proxy blocks are untouched
        np.testing.assert_allclose(s5.y[: s5.T0], s0.y[: s0.T0])
        np.testing.assert_allclose(s5.y[s5.T0:], s0.y[s0.T0:] + 5.0)
        np.testing.assert_array_equal(s5.donor_outcomes, s0.donor_outcomes)
        np.testing.assert_array_equal(s5.donor_proxies, s0.donor_proxies)

    def test_nonstationary_factor_trends(self):
        # the 0.5*log(t) trend makes the untreated level drift up over time
        s = simulate_pioid_linear(dist_lambda="nonstationary", t0=200, seed=4)
        pre = s.y[: s.T0]
        assert pre[-50:].mean() > pre[:50].mean()

    def test_ar_errors_differ_from_iid(self):
        kw = dict(n_units=7, t0=80, seed=5)
        ar = simulate_pioid_linear(dist_epsilon="AR", **kw).y
        iid = simulate_pioid_linear(dist_epsilon="iid", **kw).y
        assert not np.allclose(ar, iid)

    def test_constrained_vs_unconstrained_loading(self):
        # the treated loading level differs (1.0 vs 1.5), so the treated series
        # differs while the donor / proxy blocks (which never use U_0) match
        c = simulate_pioid_linear(u_setting="constrained", seed=6)
        u = simulate_pioid_linear(u_setting="unconstrained", seed=6)
        np.testing.assert_array_equal(c.donor_outcomes, u.donor_outcomes)
        assert not np.allclose(c.y, u.y)

    def test_seed_is_deterministic(self):
        a = simulate_pioid_linear(seed=7).y
        b = simulate_pioid_linear(seed=7).y
        np.testing.assert_array_equal(a, b)

    def test_rng_takes_precedence_over_seed(self):
        rng = np.random.default_rng(5)
        s = simulate_pioid_linear(n_units=11, rng=rng, seed=999)
        assert s.donor_outcomes.shape[1] == 5

    @pytest.mark.parametrize("kwargs", [
        {"n_units": 4},           # below the minimum
        {"n_units": 6},           # even -> non-conforming loading
        {"t0": 0},                # empty pre-period
        {"dist_lambda": "bogus"},
        {"dist_epsilon": "bogus"},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            simulate_pioid_linear(seed=0, **kwargs)


class TestAr1Noise:
    def test_iid_shape_and_randomness(self):
        e = _ar1_noise(50, 3, np.random.default_rng(0), phi=0.1, ar=False)
        assert e.shape == (50, 3)

    def test_ar_starts_at_zero_state(self):
        # first row is the raw innovation (no prior state); AR recursion builds up
        rng = np.random.default_rng(0)
        e = _ar1_noise(10, None, rng, phi=0.5, ar=True)
        assert e.shape == (10,)
        # a pure-AR series with phi differs from the same innovations i.i.d.
        rng2 = np.random.default_rng(0)
        iid = _ar1_noise(10, None, rng2, phi=0.5, ar=False)
        assert not np.allclose(e, iid)
