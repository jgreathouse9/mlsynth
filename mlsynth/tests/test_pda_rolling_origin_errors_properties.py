"""Metamorphic invariants of the PDA rolling-origin errors.

The unit suite pins the series on fixed panels. These assert what must hold for
every panel:

* agreement -- when the estimator is a bare linear combination of the donors, this
  reader and ``conformal.rolling_origin_period_errors`` return the same numbers, so
  the two cannot disagree about the schedule or the sign of an error;
* structure -- the length is the origin count times the horizon;
* equivariance -- rescaling the outcome and the counterfactual rescales every error
  by the same factor;
* invariance -- shifting the counterfactual shifts every error by the negative of
  the shift, and nothing else;
* monotonicity -- training on more of the pre-period never yields more windows.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.conformal import (origin_schedule,
                                     rolling_origin_counterfactual_errors,
                                     rolling_origin_period_errors)

_SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def _ols_weight_fn(y, Y0):
    def _fn(keep_idx):
        idx = np.asarray(keep_idx)
        beta, *_ = np.linalg.lstsq(Y0[idx], y[idx], rcond=None)
        return np.asarray(beta, float)
    return _fn


def _ols_refit_at(y, Y0):
    """The same fit, expressed as a counterfactual over the whole sample."""
    wf = _ols_weight_fn(y, Y0)
    return lambda origin: Y0 @ wf(np.arange(origin))


@st.composite
def panels(draw, min_pre=48, max_pre=96):
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


class TestAgreement:
    @given(panels())
    @_SETTINGS
    def test_a_linear_fit_reads_the_same_either_way(self, panel):
        y, Y0, pre, horizon = panel
        via_cf = rolling_origin_counterfactual_errors(
            y, _ols_refit_at(y, Y0), pre, horizon, min_train_frac=0.2)
        via_w = rolling_origin_period_errors(
            y, Y0, pre, horizon, _ols_weight_fn(y, Y0), min_train_frac=0.2)
        assert via_cf.size == via_w.size
        if via_w.size:
            assert via_cf == pytest.approx(via_w, rel=1e-7, abs=1e-9)


class TestStructure:
    @given(panels())
    @_SETTINGS
    def test_length_is_the_origin_count_times_the_horizon(self, panel):
        y, Y0, pre, horizon = panel
        out = rolling_origin_counterfactual_errors(
            y, _ols_refit_at(y, Y0), pre, horizon, min_train_frac=0.2)
        assert out.size == len(list(origin_schedule(pre, horizon, 0.2))) * horizon


class TestEquivariance:
    @given(panels(), st.floats(0.25, 4.0, allow_nan=False, allow_infinity=False))
    @_SETTINGS
    def test_rescaling_rescales_every_error(self, panel, c):
        y, Y0, pre, horizon = panel
        base = rolling_origin_counterfactual_errors(
            y, _ols_refit_at(y, Y0), pre, horizon, min_train_frac=0.2)
        assume(base.size)
        scaled = rolling_origin_counterfactual_errors(
            c * y, _ols_refit_at(c * y, c * Y0), pre, horizon, min_train_frac=0.2)
        assert scaled == pytest.approx(c * base, rel=1e-8, abs=1e-9)


class TestInvariance:
    @given(panels(), st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
    @_SETTINGS
    def test_shifting_the_counterfactual_shifts_every_error(self, panel, shift):
        y, Y0, pre, horizon = panel
        refit_at = _ols_refit_at(y, Y0)
        base = rolling_origin_counterfactual_errors(
            y, refit_at, pre, horizon, min_train_frac=0.2)
        assume(base.size)
        moved = rolling_origin_counterfactual_errors(
            y, lambda o: refit_at(o) + shift, pre, horizon, min_train_frac=0.2)
        assert moved == pytest.approx(base - shift, rel=1e-8, abs=1e-9)


class TestMonotonicity:
    @given(panels())
    @_SETTINGS
    def test_training_on_more_of_the_pre_period_never_adds_windows(self, panel):
        y, Y0, pre, horizon = panel
        refit_at = _ols_refit_at(y, Y0)
        few = rolling_origin_counterfactual_errors(y, refit_at, pre, horizon,
                                                   min_train_frac=0.6)
        many = rolling_origin_counterfactual_errors(y, refit_at, pre, horizon,
                                                    min_train_frac=0.2)
        assert few.size <= many.size
