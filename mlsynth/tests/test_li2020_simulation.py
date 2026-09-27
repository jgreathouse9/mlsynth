"""The Li (2020) JASA data-generating processes.

Reference: Kathleen T. Li, "Statistical Inference for Average Treatment
Effects Estimated by Synthetic Control Methods", *JASA* 115(532):2068-2083,
Section 5.1 and Section 5.4. Eight loading configurations over one factor
process, in two variants: DGP1-DGP2 and DGP5-DGP8 use a stationary first
factor, DGP3-DGP4 replace it by a unit root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.tssc_helpers.simulation import (
    LI2020_DGPS,
    Li2020Sample,
    li2020_loadings,
    simulate_li2020_sample,
)


# ----------------------------------------------------------------------
# Loadings: the eight configurations, as the paper writes them
# ----------------------------------------------------------------------

class TestLoadings:
    def test_dgp1_is_homogeneous_over_six_controls(self):
        """b_1 = b_2 = ... = b_7 = 1, b_8 = ... = b_11 = 0."""
        B = li2020_loadings("dgp1", N=11)
        assert B.shape == (11, 3)
        np.testing.assert_allclose(B[:7], 1.0)
        np.testing.assert_allclose(B[7:], 0.0)

    def test_dgp2_doubles_the_treated_loading(self):
        """The treated and control units come from different distributions."""
        B = li2020_loadings("dgp2", N=11)
        np.testing.assert_allclose(B[0], 2.0)
        np.testing.assert_allclose(B[1:7], 1.0)
        np.testing.assert_allclose(B[7:], 0.0)

    def test_the_unit_root_pair_reuse_the_stationary_loadings(self):
        """DGP3 and DGP4 differ from DGP1 and DGP2 only in the factor."""
        np.testing.assert_array_equal(li2020_loadings("dgp3", N=11),
                                      li2020_loadings("dgp1", N=11))
        np.testing.assert_array_equal(li2020_loadings("dgp4", N=11),
                                      li2020_loadings("dgp2", N=11))

    @pytest.mark.parametrize("dgp,treated,first,second", [
        ("dgp5", 1.0, 2.0, -0.5),
        ("dgp6", 1.0, -2.0, 0.5),
        ("dgp7", 1.0, 0.5, 0.2),
        ("dgp8", 0.2, 1.0, 0.5),
    ])
    @pytest.mark.parametrize("N", [11, 21, 31, 51, 81])
    def test_the_large_n_configurations_split_the_controls_in_half(
        self, dgp, treated, first, second, N
    ):
        """j = 2..(N+1)/2 take one value, j = (N+3)/2..N the other."""
        B = li2020_loadings(dgp, N=N)
        assert B.shape == (N, 3)
        np.testing.assert_allclose(B[0], treated)
        half = (N + 1) // 2
        np.testing.assert_allclose(B[1:half], first)
        np.testing.assert_allclose(B[half:], second)
        assert B[1:half].shape[0] == B[half:].shape[0] == (N - 1) // 2

    def test_an_unknown_dgp_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="dgp"):
            li2020_loadings("dgp9", N=11)

    def test_a_panel_too_narrow_for_the_configuration_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="N"):
            li2020_loadings("dgp1", N=5)


# ----------------------------------------------------------------------
# The factor process
# ----------------------------------------------------------------------

class TestFactors:
    def test_the_first_factor_is_an_ar1_with_coefficient_point_eight(self):
        s = simulate_li2020_sample(dgp="dgp1", T1=4000, T2=20,
                                   rng=np.random.default_rng(0))
        f1 = s.factors[:, 0]
        rho = float(np.corrcoef(f1[:-1], f1[1:])[0, 1])
        assert rho == pytest.approx(0.8, abs=0.05)

    def test_the_unit_root_variant_is_integrated(self):
        """f_1 is I(1) under DGP3: its first difference is the stationary one."""
        s = simulate_li2020_sample(dgp="dgp3", T1=4000, T2=20,
                                   rng=np.random.default_rng(0))
        f1 = s.factors[:, 0]
        rho_level = float(np.corrcoef(f1[:-1], f1[1:])[0, 1])
        d = np.diff(f1)
        rho_diff = float(np.corrcoef(d[:-1], d[1:])[0, 1])
        assert rho_level > 0.97
        assert abs(rho_diff) < 0.1

    def test_the_stationary_and_unit_root_runs_differ_only_in_the_first_factor(self):
        a = simulate_li2020_sample(dgp="dgp1", rng=np.random.default_rng(3))
        b = simulate_li2020_sample(dgp="dgp3", rng=np.random.default_rng(3))
        np.testing.assert_allclose(a.factors[:, 1:], b.factors[:, 1:])
        assert not np.allclose(a.factors[:, 0], b.factors[:, 0])

    def test_the_third_factor_is_an_ma2(self):
        """f_3 has no autocorrelation past lag 2."""
        s = simulate_li2020_sample(dgp="dgp1", T1=6000, T2=20,
                                   rng=np.random.default_rng(1))
        f3 = s.factors[:, 2]
        r3 = float(np.corrcoef(f3[:-3], f3[3:])[0, 1])
        assert abs(r3) < 0.05


# ----------------------------------------------------------------------
# Errors, treatment effect, panel shape
# ----------------------------------------------------------------------

class TestSample:
    def test_the_uniform_errors_have_unit_variance(self):
        """uniform[-sqrt(3), sqrt(3)] is the paper's choice, variance one."""
        s = simulate_li2020_sample(dgp="dgp1", T1=8000, T2=20, error="uniform",
                                   rng=np.random.default_rng(0))
        assert float(np.var(s.errors)) == pytest.approx(1.0, abs=0.05)
        assert float(np.max(np.abs(s.errors))) <= np.sqrt(3.0) + 1e-12

    def test_the_normal_errors_are_also_unit_variance(self):
        s = simulate_li2020_sample(dgp="dgp1", T1=8000, T2=20, error="normal",
                                   rng=np.random.default_rng(0))
        assert float(np.var(s.errors)) == pytest.approx(1.0, abs=0.05)

    def test_no_effect_when_alpha0_is_zero(self):
        s = simulate_li2020_sample(dgp="dgp1", alpha0=0.0,
                                   rng=np.random.default_rng(0))
        np.testing.assert_allclose(s.effect, 0.0)
        assert s.true_att == 0.0

    def test_the_effect_touches_only_the_post_period(self):
        s = simulate_li2020_sample(dgp="dgp1", alpha0=1.0, T1=90, T2=20,
                                   rng=np.random.default_rng(0))
        assert s.effect.shape == (110,)
        np.testing.assert_allclose(s.effect[:90], 0.0)
        assert np.all(s.effect[90:] > 0.0)

    def test_the_population_effect_is_one_and_a_half_alpha0(self):
        """E[e^z/(1+e^z)] = 1/2 for z symmetric about zero, so E(D_1t) = 1.5 a0."""
        s = simulate_li2020_sample(dgp="dgp1", alpha0=2.0, T1=90, T2=40000,
                                   rng=np.random.default_rng(0))
        assert float(np.mean(s.effect[90:])) == pytest.approx(3.0, abs=0.05)
        assert s.true_att == pytest.approx(3.0, rel=1e-12)

    def test_the_treated_series_carries_the_effect(self):
        s = simulate_li2020_sample(dgp="dgp1", alpha0=1.0,
                                   rng=np.random.default_rng(0))
        np.testing.assert_allclose(s.Y_treated - s.Y_treated_untreated, s.effect,
                                   atol=1e-12)

    def test_the_long_panel_is_well_formed(self):
        s = simulate_li2020_sample(dgp="dgp2", N=11, T1=90, T2=20,
                                   rng=np.random.default_rng(0))
        assert isinstance(s, Li2020Sample)
        assert isinstance(s.df, pd.DataFrame)
        assert set(s.df.columns) == {"unit", "time", "y", "D"}
        assert len(s.df) == 11 * 110
        treated = s.df[s.df["unit"] == "treated"].sort_values("time")
        assert (treated["D"].to_numpy() == (np.arange(110) >= 90)).all()
        assert (s.df[s.df["unit"] != "treated"]["D"] == 0).all()
        assert s.df[s.df["unit"] != "treated"]["unit"].nunique() == 10
        np.testing.assert_allclose(treated["y"].to_numpy(), s.Y_treated)

    def test_determinism(self):
        kw = dict(dgp="dgp5", N=21, T1=50, T2=10, alpha0=1.0)
        a = simulate_li2020_sample(rng=np.random.default_rng(9), **kw)
        b = simulate_li2020_sample(rng=np.random.default_rng(9), **kw)
        np.testing.assert_array_equal(a.Y_treated, b.Y_treated)
        np.testing.assert_array_equal(a.Y_controls, b.Y_controls)
        pd.testing.assert_frame_equal(a.df, b.df)

    def test_the_intercept_is_one_for_every_unit(self):
        """a = (1, 1, ..., 1) in Equation 26."""
        s = simulate_li2020_sample(dgp="dgp1", N=11, T1=5000, T2=20,
                                   rng=np.random.default_rng(2))
        # Units 8..11 have zero loadings, so their mean is the intercept alone.
        zero_loaded = s.Y_controls[:, -4:]
        np.testing.assert_allclose(zero_loaded.mean(axis=0), 1.0, atol=0.05)

    def test_every_registered_dgp_runs(self):
        for dgp in LI2020_DGPS:
            s = simulate_li2020_sample(dgp=dgp, N=11, T1=40, T2=10,
                                       rng=np.random.default_rng(0))
            assert np.all(np.isfinite(s.Y_treated))
            assert np.all(np.isfinite(s.Y_controls))

    def test_a_bad_error_distribution_is_refused(self):
        with pytest.raises(MlsynthConfigError, match="error"):
            simulate_li2020_sample(dgp="dgp1", error="cauchy",
                                   rng=np.random.default_rng(0))
