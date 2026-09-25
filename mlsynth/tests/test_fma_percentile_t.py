"""Tests for the Wang, Racine & Wang (2025) percentile-t bootstrap in FMA.

Reference: Lixiong Wang, Jeffrey S. Racine & Qiying Wang, "Bootstrap
inference on a factor model based average treatment effects estimator",
*Econometric Reviews* 45(1):78-95. Section 3.1 gives the three bootstrap
steps, Appendix A.1 defines ``Omega`` and its estimator, and Appendix A.2
defines the bootstrap-world ``Omega*``.

The procedure studentizes the average ATT: each draw produces
``S* = sqrt(T2) (ATT* - ATT) / sqrt(Omega*)``, and the interval inverts the
order statistics of ``S*``. It differs from the Web Appendix F bootstrap
already in FMA on both estimand (average ATT, not per-period ``ATT_t``) and
mechanism (Gaussian draws calibrated to the treated unit's own residual
variance, not resampled control residuals).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.fma_helpers.inference import (
    asymptotic_inference,
    percentile_t_inference,
    robust_omega,
)


# ----------------------------------------------------------------------
# Fixtures: a factor design with a known loading
# ----------------------------------------------------------------------

def _design(T0=30, T2=10, r=3, seed=0, hetero=False, sigma=1.0):
    """Build (treated_outcome, counterfactual, F_aug, T0) for a factor DGP."""
    rng = np.random.default_rng(seed)
    T = T0 + T2
    F = rng.standard_normal((T, r))
    F_aug = np.concatenate([np.ones((T, 1)), F], axis=1)
    lam = np.array([1.0] + [0.5 * (j + 1) for j in range(r)])
    scale = sigma * (1.0 + 2.0 * (np.arange(T) / T)) if hetero else sigma
    y = F_aug @ lam + rng.standard_normal(T) * scale
    # Counterfactual from the pre-period OLS fit, as FMA builds it.
    beta = np.linalg.solve(F_aug[:T0].T @ F_aug[:T0], F_aug[:T0].T @ y[:T0])
    cf = F_aug @ beta
    return y, cf, F_aug, T0


# ----------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------

class TestSmoke:
    def test_returns_finite_interval_containing_the_att(self):
        y, cf, F_aug, T0 = _design()
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, alpha=0.05, n_replicates=400, seed=0,
        )
        att = float(np.mean((y - cf)[T0:]))
        for key in ("se_att", "lower", "upper", "p_value", "omega",
                    "omega1", "omega2"):
            assert np.isfinite(out[key]), key
        assert out["lower"] < att < out["upper"]
        assert out["se_att"] > 0.0
        assert 0.0 < out["p_value"] <= 1.0

    def test_statistics_shape_and_finiteness(self):
        y, cf, F_aug, T0 = _design()
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=250, seed=1,
        )
        S = out["statistics"]
        assert S.shape == (250,)
        assert np.all(np.isfinite(S))
        assert out["n_replicates"] == 250


# ----------------------------------------------------------------------
# Appendix A.1: the Omega estimator
# ----------------------------------------------------------------------

class TestRobustOmega:
    def test_reduces_to_the_homoskedastic_plug_in(self):
        """Equal-magnitude residuals collapse the sandwich to sigma^2 A^-1.

        Appendix A.1 has Psi = A^-1 V A^-1 with V = E(u^2 f f'). When u_t^2
        is constant at c^2, V = c^2 A and Psi = c^2 A^-1, so
        Omega1 = phi c^2 eta' A^-1 eta -- the form the Theorem 3.1 path
        already computes. The robust estimator must nest it exactly.
        """
        _, _, F_aug, T0 = _design(T0=30, T2=10)
        T2 = 10
        c = 1.7
        resid = np.full(T0, c)
        omega, omega1, omega2 = robust_omega(
            factors_with_const=F_aug, T0=T0, T2=T2, residuals_pre=resid,
        )
        F_pre, F_post = F_aug[:T0], F_aug[T0:]
        eta = F_post.mean(axis=0)
        A = (F_pre.T @ F_pre) / T0
        expected1 = (T2 / T0) * c ** 2 * float(eta @ np.linalg.inv(A) @ eta)
        assert omega1 == pytest.approx(expected1, rel=1e-10)
        assert omega2 == pytest.approx(c ** 2, rel=1e-12)
        assert omega == pytest.approx(omega1 + omega2, rel=1e-12)

    def test_alternating_sign_residuals_also_collapse(self):
        """Only u_t^2 enters, so the signs of the residuals cannot matter."""
        _, _, F_aug, T0 = _design(T0=24, T2=8)
        resid = np.full(T0, 0.9)
        flipped = resid * np.where(np.arange(T0) % 2 == 0, 1.0, -1.0)
        a = robust_omega(factors_with_const=F_aug, T0=T0, T2=8,
                         residuals_pre=resid)
        b = robust_omega(factors_with_const=F_aug, T0=T0, T2=8,
                         residuals_pre=flipped)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_heteroskedasticity_moves_omega_away_from_the_plug_in(self):
        """The test above has no power unless heteroskedasticity separates them.

        Residuals whose squares are correlated with the projection g_t^2 give
        a different Omega1 from the homoskedastic plug-in at the same mean
        square, which is the whole point of the sandwich.
        """
        _, _, F_aug, T0 = _design(T0=30, T2=10)
        T2 = 10
        F_pre, F_post = F_aug[:T0], F_aug[T0:]
        eta = F_post.mean(axis=0)
        A = (F_pre.T @ F_pre) / T0
        g = F_pre @ np.linalg.solve(A, eta)
        # Put the variance where g^2 is largest, holding the mean square at 1.
        order = np.argsort(g ** 2)
        w = np.zeros(T0)
        w[order[T0 // 2:]] = 2.0
        w[order[:T0 // 2]] = 0.0
        resid = np.sqrt(w)
        _, omega1, omega2 = robust_omega(
            factors_with_const=F_aug, T0=T0, T2=T2, residuals_pre=resid,
        )
        plug_in = (T2 / T0) * omega2 * float(eta @ np.linalg.inv(A) @ eta)
        assert omega2 == pytest.approx(1.0, rel=1e-12)
        assert omega1 > plug_in * 1.05

    def test_omega2_uses_the_T0_normalisation_not_the_dof_correction(self):
        """Appendix A.1 fixes Omega2 = T1^-1 sum u^2, with no dof correction.

        FMA's ``estimate_loading_and_counterfactual`` divides by T0 - (r + 1)
        for the Theorem 3.1 path. The bootstrap's sigma^2_tr (Step 1) is the
        uncorrected mean square, and the two must not be conflated.
        """
        from mlsynth.utils.fma_helpers.fit import (
            estimate_loading_and_counterfactual,
        )
        y, cf, F_aug, T0 = _design(T0=8, T2=5, r=3)
        _, _, _, dof_corrected = estimate_loading_and_counterfactual(
            y, F_aug[:, 1:], T0,
        )
        resid = (y - cf)[:T0]
        _, _, omega2 = robust_omega(
            factors_with_const=F_aug, T0=T0, T2=5, residuals_pre=resid,
        )
        assert omega2 == pytest.approx(float(np.mean(resid ** 2)), rel=1e-12)
        # T0 = 8 against r + 1 = 4 columns: the two differ by a factor of 2.
        assert dof_corrected == pytest.approx(omega2 * T0 / (T0 - 4), rel=1e-10)
        assert omega2 < dof_corrected

    def test_singular_design_falls_back_to_pinv(self):
        _, _, F_aug, T0 = _design(T0=20, T2=6, r=3)
        F_aug[:, 3] = F_aug[:, 2]          # exact collinearity
        out = robust_omega(factors_with_const=F_aug, T0=T0, T2=6,
                           residuals_pre=np.full(T0, 0.5))
        assert all(np.isfinite(v) for v in out)


# ----------------------------------------------------------------------
# Section 3.1 Step 2: the bootstrap statistic
# ----------------------------------------------------------------------

class TestStatistic:
    def test_statistics_are_approximately_pivotal(self):
        """S* mimics sqrt(T2)(ATT - ATT_true)/sqrt(Omega): mean 0, sd near 1."""
        y, cf, F_aug, T0 = _design(T0=60, T2=20, seed=5)
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=4000, seed=11,
        )
        S = out["statistics"]
        assert abs(float(S.mean())) < 0.10
        assert 0.75 < float(S.std()) < 1.35

    def test_statistics_are_invariant_to_the_outcome_scale(self):
        """S* is studentized, so rescaling the treated series cannot move it.

        Both numerator and denominator are homogeneous of degree one in the
        residual scale, and the Gaussian draws scale with sigma_tr, so with a
        shared seed the statistics agree to floating point.
        """
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=3)
        base = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=300, seed=7,
        )
        c = 4.25
        scaled = percentile_t_inference(
            treated_outcome=c * y, counterfactual=c * cf,
            factors_with_const=F_aug, T0=T0, n_replicates=300, seed=7,
        )
        np.testing.assert_allclose(scaled["statistics"], base["statistics"],
                                   rtol=1e-10, atol=1e-12)
        assert scaled["se_att"] == pytest.approx(c * base["se_att"], rel=1e-10)
        assert scaled["lower"] == pytest.approx(c * base["lower"], rel=1e-10)
        assert scaled["upper"] == pytest.approx(c * base["upper"], rel=1e-10)
        assert scaled["p_value"] == pytest.approx(base["p_value"], rel=1e-12)

    def test_shifting_the_gap_shifts_the_interval_but_not_the_statistics(self):
        """ATT enters only as a location, per the paper's AT T = 0 argument."""
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=4)
        shift = 2.5
        y_shifted = y.copy()
        y_shifted[T0:] += shift
        base = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=300, seed=2,
        )
        moved = percentile_t_inference(
            treated_outcome=y_shifted, counterfactual=cf,
            factors_with_const=F_aug, T0=T0, n_replicates=300, seed=2,
        )
        np.testing.assert_allclose(moved["statistics"], base["statistics"],
                                   rtol=1e-12)
        assert moved["lower"] == pytest.approx(base["lower"] + shift, rel=1e-10)
        assert moved["upper"] == pytest.approx(base["upper"] + shift, rel=1e-10)

    def test_determinism_and_seed_sensitivity(self):
        y, cf, F_aug, T0 = _design()
        kw = dict(treated_outcome=y, counterfactual=cf,
                  factors_with_const=F_aug, T0=T0, n_replicates=200)
        a = percentile_t_inference(seed=0, **kw)
        b = percentile_t_inference(seed=0, **kw)
        c = percentile_t_inference(seed=1, **kw)
        np.testing.assert_array_equal(a["statistics"], b["statistics"])
        assert not np.allclose(a["statistics"], c["statistics"])


# ----------------------------------------------------------------------
# Section 3.1 Step 3: inverting the order statistics
# ----------------------------------------------------------------------

class TestInversion:
    def test_bounds_invert_the_sorted_statistics_with_the_tails_reversed(self):
        """Equation (10): the upper bound subtracts the LOWER order statistic.

        ``[ATT - S*_((1-a/2)M) sqrt(Om/T2), ATT - S*_(aM/2) sqrt(Om/T2)]``.
        Reversing these two -- the classic percentile-t slip -- leaves a
        superficially plausible interval, so it is pinned exactly here.
        """
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=9)
        M, alpha = 1000, 0.05
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, alpha=alpha, n_replicates=M, seed=13,
        )
        att = float(np.mean((y - cf)[T0:]))
        S = np.sort(out["statistics"])
        root = np.sqrt(out["omega"] / 10)
        k_lo = int(np.ceil(alpha / 2 * M)) - 1
        k_hi = int(np.ceil((1 - alpha / 2) * M)) - 1
        assert (k_lo, k_hi) == (24, 974)
        assert out["lower"] == pytest.approx(att - S[k_hi] * root, rel=1e-12)
        assert out["upper"] == pytest.approx(att - S[k_lo] * root, rel=1e-12)

    def test_se_att_is_the_robust_root(self):
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=6)
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=200, seed=4,
        )
        assert out["se_att"] == pytest.approx(
            float(np.sqrt(out["omega"] / 10)), rel=1e-12
        )

    def test_interval_widens_as_alpha_falls(self):
        y, cf, F_aug, T0 = _design(T0=40, T2=10, seed=8)
        kw = dict(treated_outcome=y, counterfactual=cf,
                  factors_with_const=F_aug, T0=T0, n_replicates=1000, seed=5)
        wide = percentile_t_inference(alpha=0.01, **kw)
        narrow = percentile_t_inference(alpha=0.10, **kw)
        assert (wide["upper"] - wide["lower"]) > (narrow["upper"]
                                                 - narrow["lower"])

    def test_p_value_is_large_when_the_gap_is_noise(self):
        y, cf, F_aug, T0 = _design(T0=40, T2=10, seed=12)
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=1000, seed=3,
        )
        assert out["p_value"] > 0.15

    def test_p_value_is_small_for_a_large_planted_effect(self):
        y, cf, F_aug, T0 = _design(T0=40, T2=10, seed=12)
        y = y.copy()
        y[T0:] += 12.0
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=1000, seed=3,
        )
        assert out["p_value"] < 0.01
        assert out["lower"] > 0.0

    def test_p_value_never_collapses_to_exactly_zero(self):
        """The (1 + #exceedances)/(M + 1) convention keeps it strictly positive."""
        y, cf, F_aug, T0 = _design(T0=40, T2=10, seed=12)
        y = y.copy()
        y[T0:] += 500.0
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=200, seed=3,
        )
        assert out["p_value"] > 0.0
        assert out["p_value"] == pytest.approx(2.0 / 201.0, rel=1e-12)


# ----------------------------------------------------------------------
# Edge cases and failures
# ----------------------------------------------------------------------

class TestEdgeCases:
    def test_no_post_periods_returns_nan(self):
        y, cf, F_aug, _ = _design(T0=30, T2=10)
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=40, n_replicates=100, seed=0,
        )
        assert np.isnan(out["se_att"])
        assert np.isnan(out["lower"]) and np.isnan(out["upper"])
        assert out["n_replicates"] == 0
        assert out["statistics"].size == 0

    def test_single_post_period_works(self):
        y, cf, F_aug, T0 = _design(T0=25, T2=1, seed=2)
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=200, seed=0,
        )
        assert np.isfinite(out["se_att"]) and out["se_att"] > 0
        assert out["lower"] < out["upper"]

    def test_saturated_pre_period_warns_and_returns_nan(self):
        """T0 = r + 1 leaves no residual dof, so Omega* vanishes on every draw.

        The residuals come out at 1e-32 and not at 0, so the check has to be
        on the design (T0 against the column count) and not on the measured
        residual variance.
        """
        rng = np.random.default_rng(0)
        T0, T2, r = 4, 5, 3
        F = rng.standard_normal((T0 + T2, r))
        F_aug = np.concatenate([np.ones((T0 + T2, 1)), F], axis=1)
        lam = np.array([1.0, 0.3, -0.2, 0.7])
        y = F_aug @ lam
        y[T0:] += 1.0
        cf = F_aug @ np.linalg.solve(F_aug[:T0].T @ F_aug[:T0],
                                     F_aug[:T0].T @ y[:T0])
        with pytest.warns(UserWarning, match="residual degrees of freedom"):
            out = percentile_t_inference(
                treated_outcome=y, counterfactual=cf,
                factors_with_const=F_aug, T0=T0, n_replicates=100, seed=0,
            )
        assert np.isnan(out["se_att"])
        assert out["n_replicates"] == 0

    def test_heteroskedastic_design_runs(self):
        y, cf, F_aug, T0 = _design(T0=40, T2=10, hetero=True, seed=1)
        out = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=300, seed=0,
        )
        assert np.isfinite(out["omega"]) and out["omega"] > 0


# ----------------------------------------------------------------------
# Relationship to the Theorem 3.1 interval
# ----------------------------------------------------------------------

class TestVersusAsymptotic:
    def test_robust_se_matches_theorem_31_se_when_residuals_are_flat(self):
        """The two SEs coincide exactly under equal-magnitude residuals.

        The only remaining difference is the dof correction, so the Theorem
        3.1 call is fed the same uncorrected mean square.
        """
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=0)
        T2 = 10
        c = 1.3
        flat = np.full(T0, c)
        y_flat = cf.copy()
        y_flat[:T0] = cf[:T0] + flat
        omega, _, _ = robust_omega(factors_with_const=F_aug, T0=T0, T2=T2,
                                   residuals_pre=flat)
        se_robust = float(np.sqrt(omega / T2))
        se_asym, _, _, _ = asymptotic_inference(
            treated_outcome=y_flat, counterfactual=cf,
            factors_with_const=F_aug, residual_variance=c ** 2, T0=T0,
        )
        assert se_robust == pytest.approx(se_asym, rel=1e-10)


# ----------------------------------------------------------------------
# Estimator wiring
# ----------------------------------------------------------------------

def _panel(N_co=12, T1=20, T2=10, seed=0):
    from mlsynth.utils.fma_helpers.simulation import simulate_fma_sample
    return simulate_fma_sample(
        dgp="dgp1", N_co=N_co, T1=T1, T2=T2, variance_case="equal",
        rng=np.random.default_rng(seed),
    ).df


def _fit(methods, **extra):
    import warnings as _warnings
    from mlsynth import FMA
    cfg = {"df": _panel(), "outcome": "y", "treat": "D", "unitid": "unit",
           "time": "time", "display_graphs": False,
           "inference_methods": methods}
    cfg.update(extra)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return FMA(cfg).fit()


class TestEstimatorWiring:
    def test_config_accepts_percentile_t(self):
        from mlsynth.config_models import FMAConfig
        cfg = FMAConfig(df=_panel(), outcome="y", treat="D", unitid="unit",
                        time="time", inference_methods=["percentile_t"])
        assert cfg.inference_methods == ["percentile_t"]

    def test_config_still_rejects_an_unknown_method(self):
        from mlsynth.config_models import FMAConfig
        from mlsynth.exceptions import MlsynthConfigError
        with pytest.raises(MlsynthConfigError, match="percentile_t"):
            FMAConfig(df=_panel(), outcome="y", treat="D", unitid="unit",
                      time="time", inference_methods=["percentile-t"])

    def test_fit_populates_the_percentile_t_fields(self):
        res = _fit(["percentile_t"], n_bootstrap=300)
        d = res.inference_detail
        assert d.method == "percentile_t"
        assert np.isfinite(d.percentile_t_att_se) and d.percentile_t_att_se > 0
        assert d.percentile_t_att_lower < d.att < d.percentile_t_att_upper
        assert 0.0 < d.percentile_t_att_p_value <= 1.0
        assert np.isfinite(d.percentile_t_omega)
        assert d.percentile_t_statistics.shape == (300,)
        assert d.percentile_t_n_replicates == 300

    def test_percentile_t_alone_still_fills_the_standardized_slot(self):
        """The base contract's att_ci must resolve without the normal CI.

        Both procedures target the post-period average, so either can carry
        the standardized interval; the per-period and placebo bands cannot.
        """
        res = _fit(["percentile_t"], n_bootstrap=300)
        assert res.inference.ci_lower is not None
        assert res.inference.ci_upper is not None
        assert res.inference.standard_error is not None
        assert res.att_ci == (res.inference.ci_lower, res.inference.ci_upper)
        d = res.inference_detail
        assert res.inference.ci_lower == pytest.approx(d.percentile_t_att_lower)
        assert res.inference.ci_upper == pytest.approx(d.percentile_t_att_upper)

    def test_asymptotic_keeps_the_standardized_slot_when_both_run(self):
        res = _fit(["asymptotic", "percentile_t"], n_bootstrap=300)
        d = res.inference_detail
        assert np.isfinite(d.asymptotic_att_se)
        assert np.isfinite(d.percentile_t_att_se)
        assert res.inference.ci_lower == pytest.approx(d.asymptotic_att_lower)
        assert res.inference.ci_upper == pytest.approx(d.asymptotic_att_upper)

    def test_the_two_bootstraps_are_independent_options(self):
        """Web Appendix F fills the per-period fields and nothing else."""
        res = _fit(["bootstrap"], n_bootstrap=200)
        d = res.inference_detail
        assert d.bootstrap_att_t_lower.size > 0
        assert np.isnan(d.percentile_t_att_se)
        assert d.percentile_t_statistics.size == 0

    def test_percentile_t_leaves_the_per_period_fields_empty(self):
        res = _fit(["percentile_t"], n_bootstrap=200)
        d = res.inference_detail
        assert d.bootstrap_att_t_lower.size == 0
        assert d.bootstrap_replicates.size == 0

    def test_default_inference_methods_unchanged(self):
        from mlsynth.config_models import FMAConfig
        cfg = FMAConfig(df=_panel(), outcome="y", treat="D", unitid="unit",
                        time="time")
        assert cfg.inference_methods == ["asymptotic"]

    def test_all_four_procedures_run_together(self):
        res = _fit(["asymptotic", "bootstrap", "percentile_t", "placebo"],
                   n_bootstrap=200)
        d = res.inference_detail
        assert np.isfinite(d.asymptotic_att_se)
        assert np.isfinite(d.percentile_t_att_se)
        assert d.bootstrap_att_t_lower.size > 0
        assert d.placebo_att_curves.size > 0
        assert d.method == "asymptotic;bootstrap;percentile_t;placebo"


# ----------------------------------------------------------------------
# Path B: the paper's own Monte Carlo, at a size the suite can carry
# ----------------------------------------------------------------------

def _wrw_draw(T1, T2, N_co, sigma_co, rng):
    """One draw from Wang, Racine & Wang (2025) Equation 12 and Section 4.

    The factor processes are Hsiao, Ching & Wan's, which
    :func:`~mlsynth.utils.fma_helpers.simulation._factors_dgp1` already
    implements; what differs from ``simulate_fma_sample`` is the variance
    grid, which fixes ``sigma_tr = 1`` and varies ``sigma_co``.
    """
    from mlsynth.utils.fma_helpers.simulation import _factors_dgp1

    T, N = T1 + T2, N_co + 1
    F = _factors_dgp1(T, rng)
    lam = rng.normal(1.0, 1.0, size=(N, 3))
    u = np.empty((N, T))
    u[0] = rng.normal(0.0, 1.0, T)               # sigma_tr = 1
    u[1:] = rng.normal(0.0, sigma_co, (N_co, T))
    Y = 1.0 + lam @ F.T + u                      # alpha = 1, true ATT = 0
    return Y[0], Y[1:].T


class TestPaperCoverage:
    """The headline of the paper, at 300 draws instead of 2,000.

    At ``(N, T1, T2) = (15, 10, 10)`` the normal interval of Li & Sonnier
    (2023) covers a nominal 95% about 80% of the time (their Table 1), and
    the studentized bootstrap restores it to about 93%. The thresholds
    below sit well inside that separation, and hold across the three
    independent seed blocks they were checked on (0 / 5000 / 9000), so they
    are not fitted to the one seed the test runs.
    """

    def test_bootstrap_holds_its_size_where_the_normal_interval_does_not(self):
        from scipy.stats import norm
        from mlsynth.utils.fma_helpers.factors import extract_factors
        from mlsynth.utils.fma_helpers.fit import (
            estimate_loading_and_counterfactual,
        )

        n_sims, T1, T2, N_co = 300, 10, 10, 15
        z = float(norm.ppf(0.975))
        hits_normal = hits_boot = seen = 0
        for j in range(n_sims):
            rng = np.random.default_rng(j)
            y, Yco = _wrw_draw(T1, T2, N_co, 1.0, rng)
            _, _, F, _ = extract_factors(
                Yco, stationarity="stationary", preprocessing="demean",
                n_factors=None, max_factors=10,
            )
            _, cf, F_aug, _ = estimate_loading_and_counterfactual(y, F, T1)
            if T1 <= F_aug.shape[1]:
                continue
            gap = y - cf
            att = float(gap[T1:].mean())
            # Appendix A.1's Omega with normal critical values -- the
            # interval the paper's "Asymptotic" column reports.
            omega, _, _ = robust_omega(
                factors_with_const=F_aug, T0=T1, T2=T2,
                residuals_pre=gap[:T1],
            )
            se = float(np.sqrt(omega / T2))
            pt = percentile_t_inference(
                treated_outcome=y, counterfactual=cf,
                factors_with_const=F_aug, T0=T1, n_replicates=400,
                seed=10_000 + j,
            )
            if not np.isfinite(pt["lower"]):
                continue
            seen += 1
            hits_normal += abs(att) <= z * se
            hits_boot += pt["lower"] <= 0.0 <= pt["upper"]

        assert seen >= 0.95 * n_sims
        cov_normal = hits_normal / seen
        cov_boot = hits_boot / seen
        assert cov_boot >= 0.90, cov_boot
        assert cov_normal <= 0.88, cov_normal
        assert cov_boot - cov_normal >= 0.08


class TestExactFit:
    def test_a_counterfactual_that_fits_the_pre_period_exactly_warns(self):
        """sigma^2_tr = 0 leaves the Gaussian draws with no scale.

        Reached through the function's own contract: the caller supplies the
        counterfactual, and one that reproduces the treated pre-period
        exactly gives Omega_hat = 0 even though the design has residual
        degrees of freedom to spare.
        """
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=0)
        cf_exact = cf.copy()
        cf_exact[:T0] = y[:T0]
        with pytest.warns(UserWarning, match="reproduces the treated"):
            out = percentile_t_inference(
                treated_outcome=y, counterfactual=cf_exact,
                factors_with_const=F_aug, T0=T0, n_replicates=100, seed=0,
            )
        assert np.isnan(out["se_att"])
        assert np.isnan(out["omega"])
        assert out["n_replicates"] == 0


class TestDrawScaleCancels:
    """sigma^2_tr sets the scale of the draws and then cancels.

    ``ATT* - ATT`` is linear in the draws and ``Omega*`` is quadratic in
    them, so their ratio is free of the scale: the reported interval does
    not depend on the variance the bootstrap errors are drawn from. What
    the variance choice buys is the guard that it is non-zero. This is
    also why the procedure is immune to the treated/control variance ratio
    that breaks Xu's (2017) interval -- studentizing removes the scale,
    and Xu's percentile interval keeps it.
    """

    def test_statistics_do_not_depend_on_the_draw_variance(self):
        y, cf, F_aug, T0 = _design(T0=30, T2=10, seed=0)
        T2 = 10
        shipped = percentile_t_inference(
            treated_outcome=y, counterfactual=cf, factors_with_const=F_aug,
            T0=T0, n_replicates=250, seed=7,
        )

        def by_hand(sigma):
            F_pre, F_post = F_aug[:T0], F_aug[T0:]
            eta = F_post.mean(axis=0)
            P = np.linalg.solve(F_pre.T @ F_pre, F_pre.T)
            u = np.random.default_rng(7).standard_normal((250, T0 + T2))
            u_pre, u_post = u[:, :T0] * sigma, u[:, T0:] * sigma
            shift = u_pre @ P.T
            num = -(shift @ eta) + u_post.mean(axis=1)
            om, _, _ = robust_omega(
                factors_with_const=F_aug, T0=T0, T2=T2,
                residuals_pre=u_pre - shift @ F_pre.T,
            )
            return np.sqrt(T2) * num / np.sqrt(om)

        sigma_tr = float(np.sqrt(np.mean((y - cf)[:T0] ** 2)))
        for multiplier in (1.0, 5.0, 0.01):
            np.testing.assert_allclose(
                by_hand(sigma_tr * multiplier), shipped["statistics"],
                rtol=0, atol=1e-12,
            )
