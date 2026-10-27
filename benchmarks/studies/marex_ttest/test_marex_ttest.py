"""Tests for the instruments of the MAREX t-test study: the debiased t-test with
its blocks placed in a chosen span, its p-value, and the arms on a MAREX design."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats

from benchmarks.studies.marex_global_test_dependence import panels
from benchmarks.studies.marex_ttest import ttest as tt
from mlsynth.utils.inferutils import debiased_sc_ttest

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _sc_panel(T0=60, T1=12, J=6, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(T0 + T1, 2)).cumsum(axis=0) * 0.2
    Y0 = F @ rng.uniform(0.2, 1.0, (2, J)) + 0.3 * rng.standard_normal((T0 + T1, J))
    y = Y0[:, :3].mean(axis=1) + 0.3 * rng.standard_normal(T0 + T1)
    return y, Y0


# --- the blocks ------------------------------------------------------------------------


@SETTINGS
@given(T0=st.integers(6, 200), T1=st.integers(1, 24), K=st.integers(2, 6), data=st.data())
def test_the_blocks_are_k_equal_runs_ending_at_t0_inside_the_span(T0, T1, K, data):
    span = data.draw(st.integers(K, T0))
    B = tt.blocks(T0, T1, K, span)
    r = min(span // K, T1)
    assert B.shape == (K, r)
    assert B[-1, -1] == T0 - 1                         # the last block ends the pre-period
    assert B.min() >= T0 - span                        # every block inside the span
    np.testing.assert_array_equal(B.ravel(), np.arange(T0 - K * r, T0))   # consecutive, disjoint


def test_the_whole_pre_period_as_span_is_scinference_placement():
    # r = min(T0 // K, T1), the last K r pre-periods (ttest.R::sc.cf).
    np.testing.assert_array_equal(tt.blocks(104, 12, 3, 104).ravel(), np.arange(68, 104))
    np.testing.assert_array_equal(tt.blocks(104, 12, 3, 31).ravel(), np.arange(74, 104))


def test_a_span_too_short_for_k_blocks_is_refused():
    with pytest.raises(ValueError, match="span"):
        tt.blocks(50, 12, 4, 3)


# --- the test against the library ------------------------------------------------------


@pytest.mark.parametrize("K", [2, 3, 4])
def test_over_the_whole_pre_period_it_is_the_library_test(K):
    y, Y0 = _sc_panel()
    got = tt.debiased_ttest(y, Y0, 60, 12, K, span=60)
    want = debiased_sc_ttest(y, Y0, 60, 12, K=K)
    assert got.att == pytest.approx(want["att"], abs=1e-12)
    assert got.se == pytest.approx(want["se"], abs=1e-12)
    np.testing.assert_allclose(got.tau_k, want["tau_k"], atol=1e-12)
    assert got.r == want["r"]


def test_a_shorter_span_moves_the_blocks_and_the_answer():
    y, Y0 = _sc_panel()
    whole = tt.debiased_ttest(y, Y0, 60, 12, 3, span=60)
    blank = tt.debiased_ttest(y, Y0, 60, 12, 3, span=18)
    assert blank.r == 6 and whole.r == 12
    assert blank.att != pytest.approx(whole.att)


def test_fixed_weights_are_used_as_given():
    y, Y0 = _sc_panel()
    w = np.full(6, 1 / 6)
    got = tt.debiased_ttest(y, Y0, 60, 12, 3, span=60, weights=w)
    gap = y - Y0 @ w
    B = tt.blocks(60, 12, 3, 60)
    want = gap[60:].mean() - gap[B].mean(axis=1)
    np.testing.assert_allclose(got.tau_k, want, atol=1e-12)


def test_fixed_weights_of_the_wrong_length_are_refused():
    y, Y0 = _sc_panel()
    with pytest.raises(ValueError, match="weights"):
        tt.debiased_ttest(y, Y0, 60, 12, 3, span=60, weights=np.ones(5) / 5)


def test_a_series_of_the_wrong_length_is_refused():
    y, Y0 = _sc_panel()
    with pytest.raises(ValueError, match="T0 \\+ T1"):
        tt.debiased_ttest(y[:-1], Y0, 60, 12, 3, span=60)


# --- the statistic's law ---------------------------------------------------------------


@pytest.mark.parametrize("K", [3, 4])
def test_with_fixed_weights_and_gaussian_noise_the_statistic_is_exactly_t(K):
    # A fixed control and iid N(0, 1) gaps: each tau_k is the post mean less a
    # block mean, sd(tau_k) estimates 1/sqrt(r) independently of the means, and
    # sqrt(1 + K r / T1) sd(tau_k) / sqrt(K) is the standard error of their
    # average. The t-statistic is then t with K - 1 degrees of freedom exactly.
    rng = np.random.default_rng(K)
    T0, T1, n = 40, 8, 6000
    Y0 = np.zeros((T0 + T1, 1))
    t = np.array([tt.debiased_ttest(rng.standard_normal(T0 + T1), Y0, T0, T1, K, span=T0,
                                    weights=np.ones(1)).tstat for _ in range(n)])
    assert stats.kstest(t, stats.t(K - 1).cdf).pvalue > 0.01
    p = np.array([tt.pvalue(x, K) for x in t])
    assert abs(np.mean(p <= 0.05) - 0.05) < 0.01


# --- the p-value -----------------------------------------------------------------------


def test_the_p_value_by_hand():
    assert tt.pvalue(stats.t(2).ppf(0.975), 3) == pytest.approx(0.05)
    assert tt.pvalue(0.0, 3) == 1.0


@SETTINGS
@given(t=st.floats(-50, 50, allow_nan=False), K=st.integers(2, 8))
def test_the_p_value_is_two_sided_and_falls_with_the_statistic(t, K):
    assert tt.pvalue(t, K) == pytest.approx(tt.pvalue(-t, K), abs=1e-15)
    assert 0.0 <= tt.pvalue(t, K) <= 1.0
    assert tt.pvalue(abs(t) + 1.0, K) <= tt.pvalue(t, K)


def test_folds_that_agree_exactly_give_zero_or_full_evidence():
    # No spread across the folds: a gap of zero everywhere is no evidence, and a
    # gap that steps up by one at T0 in every fold is all the evidence there is.
    Y0, w = np.zeros((48, 1)), np.ones(1)
    flat = tt.debiased_ttest(np.zeros(48), Y0, 40, 8, 3, span=40, weights=w)
    step = tt.debiased_ttest(np.r_[np.zeros(40), np.ones(8)], Y0, 40, 8, 3, span=40, weights=w)
    assert (flat.se, flat.tstat, tt.pvalue(flat.tstat, 3)) == (0.0, 0.0, 1.0)
    assert (step.se, step.tstat, tt.pvalue(step.tstat, 3)) == (0.0, np.inf, 0.0)


def test_an_effect_shifts_every_fold_estimate_by_itself():
    y, Y0 = _sc_panel()
    base = tt.debiased_ttest(y, Y0, 60, 12, 3, span=60)
    z = y.copy()
    z[60:] += 2.5
    moved = tt.debiased_ttest(z, Y0, 60, 12, 3, span=60)
    np.testing.assert_allclose(moved.tau_k, base.tau_k + 2.5, atol=1e-10)
    assert moved.se == pytest.approx(base.se, abs=1e-12)
    assert tt.shifted_pvalue(base, 2.5) == pytest.approx(tt.pvalue(moved.tstat, 3), abs=1e-12)


# --- the arms on a MAREX design --------------------------------------------------------


def test_the_arms_on_a_design():
    rng = np.random.default_rng([0, 104, 0])
    Y = panels.panel(104 + 12, "iid", rng)
    d = panels.design(Y, 104, 12)
    out = tt.arms(Y, d, 12)
    assert set(out) == set(tt.ARMS)
    untreated = d.w <= 1e-8
    lib = debiased_sc_ttest(Y @ d.w, Y[:, untreated], 104, 12, K=3)
    assert out["refit_pre_3"].att == pytest.approx(lib["att"], abs=1e-12)
    assert out["refit_blank_3"].r == min(d.blank // 3, 12)
    # The fixed arm uses MAREX's own control, so its gap is MAREX's contrast.
    fixed = out["fixed_blank_3"]
    u = Y @ (d.w - d.v)
    B = tt.blocks(104, 12, 3, d.blank)
    np.testing.assert_allclose(fixed.tau_k, u[104:116].mean() - u[B].mean(axis=1), atol=1e-10)


# --- size-adjusted power ---------------------------------------------------------------


def test_size_adjusted_power_by_hand():
    from benchmarks.studies.marex_ttest.run import size_adjusted_power
    null = np.r_[np.full(5, 0.01), np.linspace(0.2, 1.0, 95)]   # 5% at 0.01
    assert size_adjusted_power(null, [0.005, 0.01, 0.02, 0.5]) == 0.5


def test_a_p_value_too_coarse_for_the_level_has_no_size_adjusted_power():
    from benchmarks.studies.marex_ttest.run import size_adjusted_power
    null = np.r_[np.full(10, 0.02), np.full(90, 0.6)]           # its floor rejects 10%
    assert np.isnan(size_adjusted_power(null, [0.02, 0.02]))
