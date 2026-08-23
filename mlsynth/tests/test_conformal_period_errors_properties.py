"""Metamorphic invariants of the per-period rolling-origin errors.

The unit suite pins the series on fixed panels. These properties assert what must
hold for every panel, and they are the assertions that would catch a defect the
examples happen not to exercise:

* equivalence -- summing each block reproduces ``rolling_origin_block_sums``
  exactly, so the two readers of the origin schedule cannot drift apart;
* structure -- the series length is the origin count times the horizon, and the
  scored periods are contiguous;
* equivariance -- rescaling the outcome and donors rescales every error by the same
  factor;
* invariance -- relabelling donors changes nothing;
* monotonicity -- training on more of the pre-period never yields more windows.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.conformal import (
    MIN_TRAIN_PERIODS,
    rolling_origin_block_sums,
    rolling_origin_period_errors,
)

_SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def _ols_weight_fn(y, Y0):
    """Stand-in for an estimator's refit: unconstrained donor regression."""
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
    seed = draw(st.integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    total = pre_periods + horizon + post_extra
    Y0 = rng.normal(size=(total, n_donors))
    y = Y0 @ rng.dirichlet(np.ones(n_donors)) + rng.normal(scale=0.1, size=total)
    assume(np.linalg.matrix_rank(Y0[:pre_periods]) == n_donors)
    return y, Y0, pre_periods, horizon


def _origin_count(pre_periods, horizon, min_train_frac):
    start = max(MIN_TRAIN_PERIODS, int(pre_periods * float(min_train_frac)))
    return len(range(start, int(pre_periods) - int(horizon) + 1, int(horizon)))


class TestEquivalence:
    @given(panels())
    @_SETTINGS
    def test_block_sums_are_the_per_period_errors_summed(self, panel):
        y, Y0, pre, horizon = panel
        wf = _ols_weight_fn(y, Y0)
        per = rolling_origin_period_errors(y, Y0, pre, horizon, wf, min_train_frac=0.2)
        sums = rolling_origin_block_sums(y, Y0, pre, horizon, wf, min_train_frac=0.2)
        assert per.size == sums.size * horizon
        if sums.size:
            assert per.reshape(-1, horizon).sum(axis=1) == pytest.approx(sums)


class TestStructure:
    @given(panels())
    @_SETTINGS
    def test_length_is_the_origin_count_times_the_horizon(self, panel):
        y, Y0, pre, horizon = panel
        per = rolling_origin_period_errors(y, Y0, pre, horizon,
                                           _ols_weight_fn(y, Y0), min_train_frac=0.2)
        assert per.size == _origin_count(pre, horizon, 0.2) * horizon

    @given(panels())
    @_SETTINGS
    def test_every_scored_period_lies_inside_the_pre_period(self, panel):
        y, Y0, pre, horizon = panel
        start = max(MIN_TRAIN_PERIODS, int(pre * 0.2))
        n = _origin_count(pre, horizon, 0.2)
        assert start + n * horizon <= pre


class TestEquivariance:
    @given(panels(), st.floats(0.25, 4.0, allow_nan=False, allow_infinity=False))
    @_SETTINGS
    def test_rescaling_the_panel_rescales_every_error(self, panel, c):
        y, Y0, pre, horizon = panel
        base = rolling_origin_period_errors(y, Y0, pre, horizon,
                                            _ols_weight_fn(y, Y0), min_train_frac=0.2)
        scaled = rolling_origin_period_errors(
            c * y, c * Y0, pre, horizon, _ols_weight_fn(c * y, c * Y0),
            min_train_frac=0.2)
        assume(base.size)
        assert scaled == pytest.approx(c * base, rel=1e-8, abs=1e-9)


class TestInvariance:
    @given(panels(), st.integers(0, 2 ** 31 - 1))
    @_SETTINGS
    def test_relabelling_donors_changes_nothing(self, panel, seed):
        y, Y0, pre, horizon = panel
        perm = np.random.default_rng(seed).permutation(Y0.shape[1])
        base = rolling_origin_period_errors(y, Y0, pre, horizon,
                                            _ols_weight_fn(y, Y0), min_train_frac=0.2)
        swapped = rolling_origin_period_errors(
            y, Y0[:, perm], pre, horizon, _ols_weight_fn(y, Y0[:, perm]),
            min_train_frac=0.2)
        assert swapped == pytest.approx(base, rel=1e-7, abs=1e-8)


class TestMonotonicity:
    @given(panels())
    @_SETTINGS
    def test_training_on_more_of_the_pre_period_never_adds_windows(self, panel):
        y, Y0, pre, horizon = panel
        wf = _ols_weight_fn(y, Y0)
        few = rolling_origin_period_errors(y, Y0, pre, horizon, wf, min_train_frac=0.6)
        many = rolling_origin_period_errors(y, Y0, pre, horizon, wf, min_train_frac=0.2)
        assert few.size <= many.size
