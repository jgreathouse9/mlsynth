"""Tests for the instruments of the #914 study: the global test's reference
schemes, the factor regimes and the arm that reproduces MAREX's own p-value."""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from benchmarks.studies.marex_global_test_dependence import globaltest as gt
from benchmarks.studies.marex_global_test_dependence import panels

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
series = st.lists(st.floats(-50, 50, allow_nan=False), min_size=6, max_size=40)


# --- the p-value -----------------------------------------------------------------------


@SETTINGS
@given(v=series, H=st.integers(1, 5), alpha=st.sampled_from([0.05, 0.1, 0.2]),
       stat=st.sampled_from(["abs", "signed"]))
def test_cyclic_shifts_reject_at_most_alpha_of_the_rotations(v, H, alpha, stat):
    # The group argument behind CWZ's moving-block permutations: across the N
    # rotations of any series, at most floor(alpha N) p-values fall at or below
    # alpha, whatever the statistic.
    v = np.asarray(v)
    N = v.size
    H = min(H, N - 1)
    ps = [gt.global_pvalue(np.roll(v, j)[:N - H], np.roll(v, j)[N - H:], "cyclic", stat=stat)
          for j in range(N)]
    assert sum(p <= alpha for p in ps) <= math.floor(alpha * N)


@SETTINGS
@given(v=series, H=st.integers(1, 5))
def test_a_p_value_is_a_share_of_the_reference_and_counts_the_observed(v, H):
    v = np.asarray(v)
    H = min(H, v.size - 1)
    p = gt.global_pvalue(v[:-H], v[-H:], "cyclic")
    N = v.size
    assert 1.0 / N <= p <= 1.0
    assert abs(p * N - round(p * N)) < 1e-9


def test_the_p_value_by_hand():
    blank, post = np.array([1.0, -2.0, 3.0, 0.5]), np.array([4.0, -1.0])
    # cyclic windows of |v| = [1, 2, 3, .5, 4, 1] of length 2, starting at the post block:
    # (4,1) (1,1) (1,2) (2,3) (3,.5) (.5,4): means 2.5 1 1.5 2.5 1.75 2.25 -> 2 of 6 >= 2.5
    assert gt.global_pvalue(blank, post, "cyclic") == pytest.approx(2 / 6)


def test_a_larger_post_contrast_gives_a_smaller_p_value():
    rng = np.random.default_rng(0)
    blank, post = rng.normal(size=40), rng.normal(size=12)
    ps = [gt.global_pvalue(blank, post + shift, "cyclic") for shift in (0.0, 1.0, 3.0)]
    assert ps[0] >= ps[1] >= ps[2] and ps[2] == pytest.approx(1 / 52)


@SETTINGS
@given(v=series, H=st.integers(1, 5))
def test_flat_recency_weights_are_the_unweighted_test(v, H):
    v = np.asarray(v)
    H = min(H, v.size - 1)
    assert gt.global_pvalue(v[:-H], v[-H:], "cyclic", rho=1.0) == pytest.approx(
        gt.global_pvalue(v[:-H], v[-H:], "cyclic"))


def test_recency_weights_count_near_blocks_more():
    # one large block far from the post window, one near it: the weighted test
    # counts the near one more, so the p-value moves towards it
    blank = np.zeros(30)
    far, near = blank.copy(), blank.copy()
    far[:3], near[-3:] = 10.0, 10.0
    post = np.full(3, 1.0)
    assert gt.global_pvalue(near, post, "cyclic", rho=0.9) > gt.global_pvalue(far, post, "cyclic", rho=0.9)


def test_all_subsets_match_the_library_when_enumerated():
    from mlsynth.utils.marex_helpers.inference import _global_p_value
    rng = np.random.default_rng(3)
    for _ in range(20):
        blank, post = rng.normal(size=7), rng.normal(size=3)
        s_obs = np.abs(post).mean()
        lib = _global_p_value(np.concatenate([blank, post]), 3, s_obs, 1000, rng)
        assert gt.global_pvalue(blank, post, "subsets") == pytest.approx(lib)


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="scheme"):
        gt.global_pvalue(np.zeros(10), np.zeros(3), "bootstrap")


# --- the factor regimes ------------------------------------------------------------------


@pytest.mark.parametrize("regime", panels.REGIMES)
def test_every_regime_draws_a_panel(regime):
    Y = panels.panel(120, regime, np.random.default_rng(1))
    assert Y.shape == (120, panels.J) and np.isfinite(Y).all()


def test_the_regimes_differ_in_their_memory():
    def lag1(regime):
        rng = np.random.default_rng(5)
        return np.mean([np.corrcoef(f[:-1], f[1:])[0, 1]
                        for f in (panels.factor(400, 1, 10.0, regime, rng)[:, 0] for _ in range(40))])
    r = {k: lag1(k) for k in ("iid", "ar5", "ar8", "ar95", "rw")}
    assert abs(r["iid"]) < 0.1 and r["iid"] < r["ar5"] < r["ar8"] < r["ar95"] < r["rw"]
    assert r["ar5"] == pytest.approx(0.5, abs=0.1) and r["ar8"] == pytest.approx(0.8, abs=0.08)


def test_the_season_repeats_every_year():
    # half the variance is the annual cycle, so the same weeks two years apart
    # correlate near 0.5; without the cycle they do not correlate at all
    def across_years(regime):
        rng = np.random.default_rng(2)
        f = panels.factor(52 * 4, 20, 10.0, regime, rng).reshape(4, 52, 20)
        return np.mean([np.corrcoef(f[0, :, k], f[2, :, k])[0, 1] for k in range(20)])
    assert across_years("season") == pytest.approx(0.5, abs=0.15)
    assert abs(across_years("ar5")) < 0.15


def test_an_unknown_regime_is_refused():
    with pytest.raises(ValueError, match="regime"):
        panels.factor(10, 1, 1.0, "garch", np.random.default_rng(0))


# --- the shipped arm ---------------------------------------------------------------------


def test_the_shipped_arm_is_marex_own_p_value():
    rng = np.random.default_rng(4)
    T0, H = 60, 6
    Y = panels.panel(T0 + H, "ar5", rng)
    d = panels.design(Y, T0, H)
    p = panels.shipped_pvalue(Y, d, H)
    assert p == pytest.approx(d.marex_p)
    # MAREX's default: 30% of the pre-period, raised to the fewest blank periods
    # at which its conformal quantile exists
    from mlsynth.utils.marex_helpers.windows import min_blank_periods
    assert d.blank == max(math.floor(0.3 * T0), min_blank_periods(0.05))


# --- blank windows -----------------------------------------------------------------------


def test_blank_windows_by_hand():
    blank, post = np.array([1.0, -2.0, 3.0, 0.5]), np.array([4.0, -1.0])
    # observed mean |post| 2.5; blank windows of length 2: (1,2) (2,3) (3,.5): 1.5 2.5 1.75
    # -> the observed and one window reach 2.5: 2 of 4
    assert gt.global_pvalue(blank, post, "windows") == pytest.approx(2 / 4)


@SETTINGS
@given(v=series, H=st.integers(1, 5))
def test_blank_windows_never_read_the_post_periods_as_reference(v, H):
    # adding an effect to the post periods moves only the observed statistic, so
    # the p-value cannot rise
    v = np.asarray(v)
    H = min(H, v.size // 2)
    blank, post = v[:-H], v[-H:]
    p0 = gt.global_pvalue(blank, post, "windows")
    p1 = gt.global_pvalue(blank, np.abs(post) + 1.0, "windows")
    assert p1 <= p0 + 1e-12
    assert 1.0 / (blank.size - H + 2) <= p1 <= 1.0


def test_blank_windows_need_a_blank_at_least_as_long_as_the_post():
    with pytest.raises(ValueError, match="blank"):
        gt.global_pvalue(np.zeros(3), np.zeros(5), "windows")


def test_the_signed_statistic_by_hand():
    blank, post = np.array([1.0, -2.0, 3.0, 0.5]), np.array([4.0, -1.0])
    # v = [1, -2, 3, .5, 4, -1]; |mean| of the cyclic windows from the post block:
    # (4,-1) 1.5 (-1,1) 0 (1,-2) .5 (-2,3) .5 (3,.5) 1.75 (.5,4) 2.25 -> 3 of 6 >= 1.5
    assert gt.global_pvalue(blank, post, "cyclic", stat="signed") == pytest.approx(3 / 6)


def test_an_unknown_statistic_is_refused():
    with pytest.raises(ValueError, match="stat"):
        gt.global_pvalue(np.zeros(10), np.zeros(3), "cyclic", stat="max")


def test_an_effect_on_the_treated_moves_the_post_contrast_by_itself():
    rng = np.random.default_rng(6)
    T0, H = 60, 6
    Y = panels.panel(T0 + H, "iid", rng)
    d = panels.design(Y, T0, H)
    blank, post = panels.contrast(Y, d, H)
    assert blank.size == d.blank and post.size == H
    np.testing.assert_allclose(blank, (Y @ (d.w - d.v))[d.TcE:d.T0])
    Yt = Y.copy()
    Yt[T0:, d.w > 0] += 1.5
    blank2, post2 = panels.contrast(Yt, d, H)
    np.testing.assert_array_equal(blank2, blank)
    np.testing.assert_allclose(post2 - post, 1.5, rtol=1e-12)
