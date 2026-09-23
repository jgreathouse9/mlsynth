"""Rolling-origin per-period errors for a PDA fit, from its own refit.

``mlsynth.utils.conformal.rolling_origin_period_errors`` reads the errors as
``y[block] - Y0[block] @ w``, which needs the fit to be a bare linear combination
of the donors. A PDA fit is not: LASSO carries an intercept, the modified-BIC path
standardizes the donors the way ``glmnet`` does, and forward selection and HCW
return a counterfactual with no single weight vector to multiply. So the PDA reader
asks the estimator for its counterfactual instead.

``refit_at(origin) -> counterfactual`` is the seam, and ``orchestration._build_refit``
already builds one per method: it closes over the pre-period length, so pointing it
at an origin gives a fit trained strictly before that origin. One reader covers all
four PDA variants.

The origin schedule is shared with the conformal readers
(``conformal.scores.origin_schedule``), so a PDA band and a split-conformal band
calibrate on the same windows.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (origin_schedule,
                                     rolling_origin_counterfactual_errors)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def panel(T=60, J=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, J))
    y = X @ np.array([0.5, 0.3, 0.2]) + 1.5 + rng.normal(scale=0.1, size=T)
    return y, X


def ols_refit_at(y, X, *, intercept=True):
    """A stand-in estimator: OLS on the periods before the origin, with intercept."""
    def _fn(origin):
        A = np.column_stack([np.ones(len(X)), X]) if intercept else X
        beta, *_ = np.linalg.lstsq(A[:origin], y[:origin], rcond=None)
        return A @ beta
    return _fn


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_returns_one_error_per_scored_period():
    y, X = panel()
    out = rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 48, 6)
    assert out.ndim == 1 and out.dtype == float
    assert out.size == len(list(origin_schedule(48, 6, 0.3))) * 6


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_series_is_each_window_of_the_gap_concatenated():
    y, X = panel()
    refit_at = ols_refit_at(y, X)
    out = rolling_origin_counterfactual_errors(y, refit_at, 48, 6)
    wanted = np.concatenate([(y - refit_at(o))[o:o + 6]
                             for o in origin_schedule(48, 6, 0.3)])
    assert out == pytest.approx(wanted)


def test_the_estimator_is_refit_once_per_origin_in_increasing_order():
    y, X = panel()
    base, seen = ols_refit_at(y, X), []

    def spy(origin):
        seen.append(origin)
        return base(origin)

    rolling_origin_counterfactual_errors(y, spy, 48, 6)
    assert seen == list(origin_schedule(48, 6, 0.3))


def test_it_scores_the_same_windows_as_the_conformal_readers():
    """One schedule, so a resampled band and a split band calibrate alike."""
    y, X = panel(T=80, seed=4)
    out = rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 64, 8,
                                               min_train_frac=0.25)
    assert out.size == len(list(origin_schedule(64, 8, 0.25))) * 8


def test_an_estimator_that_reproduces_the_outcome_scores_zero_error():
    y, X = panel()
    out = rolling_origin_counterfactual_errors(y, lambda o: y.copy(), 48, 6)
    assert out == pytest.approx(np.zeros_like(out), abs=1e-12)


def test_a_constant_shift_in_the_counterfactual_shifts_every_error():
    y, X = panel()
    base = ols_refit_at(y, X)
    out = rolling_origin_counterfactual_errors(y, base, 48, 6)
    shifted = rolling_origin_counterfactual_errors(y, lambda o: base(o) + 3.0, 48, 6)
    assert shifted == pytest.approx(out - 3.0)


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_pre_period_too_short_for_one_window_returns_nothing():
    y, X = panel(T=20)
    out = rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 12, 6)
    assert out.size == 0 and out.dtype == float


def test_a_horizon_equal_to_the_admissible_span_gives_one_block():
    y, X = panel(T=30)
    out = rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 16, 6,
                                               min_train_frac=0.0)
    assert out.size == 6


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -4, True, 2.5, "6", None])
def test_a_horizon_that_is_not_a_positive_integer_is_refused(bad):
    y, X = panel()
    with pytest.raises(MlsynthConfigError, match="horizon"):
        rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 48, bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.3", None])
def test_a_min_train_frac_outside_the_unit_interval_is_refused(bad):
    y, X = panel()
    with pytest.raises(MlsynthConfigError, match="min_train_frac"):
        rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 48, 6,
                                             min_train_frac=bad)


def test_a_pre_period_longer_than_the_sample_is_refused():
    y, X = panel(T=40)
    with pytest.raises(MlsynthDataError, match="pre_periods"):
        rolling_origin_counterfactual_errors(y, ols_refit_at(y, X), 80, 6)


def test_a_counterfactual_that_does_not_span_the_sample_is_refused():
    """A refit returning only its training window cannot score the next one."""
    y, X = panel()
    with pytest.raises(MlsynthDataError, match="counterfactual"):
        rolling_origin_counterfactual_errors(y, lambda o: y[:o], 48, 6)


def test_a_non_finite_counterfactual_is_refused():
    y, X = panel()
    bad = y.copy(); bad[50] = np.nan
    with pytest.raises(MlsynthDataError, match="finite"):
        rolling_origin_counterfactual_errors(y, lambda o: bad, 48, 6)
