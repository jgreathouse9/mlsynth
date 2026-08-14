"""Metamorphic invariants of the cumulative conformal band.

The unit suite pins the band on fixed panels. These properties assert what must
hold for *every* panel, and they are the assertions that would catch a defect the
examples happen not to exercise:

* equivariance -- rescaling the outcome and donors rescales the point estimate and
  the half-width by the same factor; shifting only the post-window moves the point
  by ``horizon * shift`` and leaves the calibration untouched;
* invariance -- relabelling donors changes nothing, and neither does adding a
  constant to every conformity score (the scores are centred before calibration,
  so a systematic offset is not mistaken for spread);
* monotonicity -- a smaller ``alpha`` never yields a narrower band;
* structure -- the band is symmetric, and its half-width is exactly the shared
  split-conformal order statistic applied to the centred rolling-origin scores.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.conformal import (
    MIN_TRAIN_PERIODS,
    cumulative_conformal_interval,
    cumulative_conformal_from_refit,
    rolling_origin_block_sums,
    split_conformal_quantile,
)

_SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def _ols_weight_fn(y, Y0):
    """Stand-in for an estimator's refit: unconstrained donor regression.

    Takes period indices, the contract ``debiased_sc_ttest`` already uses, so a
    covariate-aware estimator can subset its covariates by the same periods.
    """
    def _fn(keep_idx):
        idx = np.asarray(keep_idx)
        beta, *_ = np.linalg.lstsq(Y0[idx], y[idx], rcond=None)
        return np.asarray(beta, float)
    return _fn


@st.composite
def panels(draw, min_pre=48, max_pre=96):
    """A donor panel with enough pre-period to admit several calibration blocks."""
    n_donors = draw(st.integers(2, 6))
    horizon = draw(st.integers(1, 6))
    pre_periods = draw(st.integers(min_pre, max_pre))
    post_extra = draw(st.integers(0, 6))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    total = pre_periods + horizon + post_extra
    Y0 = rng.normal(size=(total, n_donors))
    y = Y0 @ rng.dirichlet(np.ones(n_donors)) + rng.normal(scale=0.1, size=total)
    # A rank-deficient donor block makes the refit non-unique; the band is then
    # not a function of the panel alone, so equivariance claims do not apply.
    assume(np.linalg.matrix_rank(Y0[:pre_periods]) == n_donors)
    return y, Y0, pre_periods, horizon


def _band(y, Y0, pre, horizon, alpha=0.1, min_train_frac=0.2):
    return cumulative_conformal_from_refit(
        y, Y0, pre, horizon, _ols_weight_fn(y, Y0), alpha=alpha, min_train_frac=min_train_frac
    )


class TestStructure:
    @given(panels())
    @_SETTINGS
    def test_band_is_symmetric_about_the_point(self, panel):
        y, Y0, pre, horizon = panel
        b = _band(y, Y0, pre, horizon)
        assert b.point - b.lower == pytest.approx(b.half_width)
        assert b.upper - b.point == pytest.approx(b.half_width)
        assert b.horizon == horizon

    @given(panels())
    @_SETTINGS
    def test_half_width_is_the_shared_order_statistic_of_centred_scores(self, panel):
        y, Y0, pre, horizon = panel
        b = _band(y, Y0, pre, horizon)
        scores = rolling_origin_block_sums(
            y, Y0, pre, horizon, _ols_weight_fn(y, Y0), min_train_frac=0.2
        )
        assert b.n_scores == scores.size
        if scores.size:
            expected = split_conformal_quantile(scores - scores.mean(), alpha=0.1)
            assert b.half_width == pytest.approx(expected)

    @given(panels())
    @_SETTINGS
    def test_point_accumulates_the_post_window(self, panel):
        y, Y0, pre, horizon = panel
        b = _band(y, Y0, pre, horizon)
        w = _ols_weight_fn(y, Y0)(np.arange(pre))
        gap = y[pre:pre + horizon] - Y0[pre:pre + horizon] @ w
        assert b.point == pytest.approx(float(gap.sum()), abs=1e-8)


class TestEquivariance:
    @given(panels(), st.floats(0.25, 4.0))
    @_SETTINGS
    def test_rescaling_the_panel_rescales_point_and_half_width(self, panel, c):
        y, Y0, pre, horizon = panel
        base = _band(y, Y0, pre, horizon)
        scaled = _band(c * y, c * Y0, pre, horizon)
        assert scaled.point == pytest.approx(c * base.point, rel=1e-6, abs=1e-8)
        if np.isfinite(base.half_width):
            assert scaled.half_width == pytest.approx(c * base.half_width, rel=1e-6, abs=1e-8)

    @given(panels(), st.floats(-5.0, 5.0))
    @_SETTINGS
    def test_shifting_only_the_post_window_moves_the_point_alone(self, panel, shift):
        y, Y0, pre, horizon = panel
        base = _band(y, Y0, pre, horizon)
        shifted_y = y.copy()
        shifted_y[pre:pre + horizon] += shift
        shifted = _band(shifted_y, Y0, pre, horizon)
        assert shifted.point == pytest.approx(base.point + horizon * shift, abs=1e-6)
        assert shifted.half_width == pytest.approx(base.half_width, nan_ok=True)


class TestInvariance:
    @given(panels())
    @_SETTINGS
    def test_relabelling_donors_changes_nothing(self, panel):
        y, Y0, pre, horizon = panel
        perm = np.random.default_rng(0).permutation(Y0.shape[1])
        base = _band(y, Y0, pre, horizon)
        permuted = _band(y, Y0[:, perm], pre, horizon)
        assert permuted.point == pytest.approx(base.point, abs=1e-6)
        if np.isfinite(base.half_width):
            assert permuted.half_width == pytest.approx(base.half_width, rel=1e-6, abs=1e-6)

    @given(
        st.lists(st.floats(-50, 50, allow_nan=False), min_size=10, max_size=40),
        st.floats(-20, 20),
    )
    @_SETTINGS
    def test_offsetting_every_score_leaves_the_half_width_alone(self, scores, offset):
        s = np.asarray(scores, dtype=float)
        base = cumulative_conformal_interval(point=1.0, scores=s, alpha=0.1)
        offsetted = cumulative_conformal_interval(point=1.0, scores=s + offset, alpha=0.1)
        assert offsetted.half_width == pytest.approx(base.half_width, rel=1e-9, abs=1e-9)


class TestMonotonicity:
    @given(panels(), st.sampled_from([0.02, 0.05, 0.1]), st.sampled_from([0.2, 0.3, 0.5]))
    @_SETTINGS
    def test_smaller_alpha_never_narrows_the_band(self, panel, tight, loose):
        assume(tight < loose)
        y, Y0, pre, horizon = panel
        narrow = _band(y, Y0, pre, horizon, alpha=loose)
        wide = _band(y, Y0, pre, horizon, alpha=tight)
        assert wide.half_width >= narrow.half_width or (
            np.isinf(wide.half_width) and np.isinf(narrow.half_width)
        )

    @given(
        st.lists(st.floats(-50, 50, allow_nan=False), min_size=20, max_size=60),
        st.sampled_from([0.05, 0.1, 0.2]),
    )
    @_SETTINGS
    def test_half_width_never_exceeds_the_largest_centred_score(self, scores, alpha):
        s = np.asarray(scores, dtype=float)
        b = cumulative_conformal_interval(point=0.0, scores=s, alpha=alpha)
        if np.isfinite(b.half_width):
            assert b.half_width <= np.max(np.abs(s - s.mean())) + 1e-9


class TestCalibrationSet:
    @given(panels())
    @_SETTINGS
    def test_origins_do_not_overlap(self, panel):
        y, Y0, pre, horizon = panel
        scores = rolling_origin_block_sums(
            y, Y0, pre, horizon, _ols_weight_fn(y, Y0), min_train_frac=0.2
        )
        start = max(MIN_TRAIN_PERIODS, int(pre * 0.2))
        assert scores.size == len(range(start, pre - horizon + 1, horizon))

    @given(panels())
    @_SETTINGS
    def test_a_later_first_origin_never_adds_calibration_blocks(self, panel):
        y, Y0, pre, horizon = panel
        early = _band(y, Y0, pre, horizon, min_train_frac=0.2)
        late = _band(y, Y0, pre, horizon, min_train_frac=0.6)
        assert late.n_scores <= early.n_scores
