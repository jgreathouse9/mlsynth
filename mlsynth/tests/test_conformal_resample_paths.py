"""One call from a fitted control to resampled cumulative error paths.

The pieces exist separately: :func:`rolling_origin_counterfactual_errors` turns a
refit-by-origin callable into a contiguous series of out-of-sample per-period
errors, and :func:`block_error_paths` turns that series into an ``(n_sim, horizon)``
matrix of accumulable paths. Every estimator wanting a resampled band composes the
same two in the same order, so the composition is the helper.

``resample_cumulative_paths`` is that composition and nothing more. This suite pins
that it is nothing more -- the two-step form and the one-call form return the same
numbers for the same seed -- so an estimator adopting it inherits exactly the
construction the benchmark cross-validated against Wheeler, and a future change to
either piece cannot reach only one of the two paths.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (
    block_error_paths,
    origin_schedule,
    resample_cumulative_paths,
    rolling_origin_counterfactual_errors,
)


def panel(T=60, J=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, J))
    y = X @ np.array([0.5, 0.3, 0.2]) + 1.5 + rng.normal(scale=0.1, size=T)
    return y, X


def ols_refit_at(y, X):
    def _fn(origin):
        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A[:origin], y[:origin], rcond=None)
        return A @ beta
    return _fn


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_returns_one_row_per_simulation_and_one_column_per_horizon():
    y, X = panel()
    paths = resample_cumulative_paths(y, ols_refit_at(y, X), 48, 6, n_sim=500)
    assert paths.shape == (500, 6)
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", [0, 1, 3])
def test_it_is_exactly_the_two_step_composition(block):
    """The headline: no estimator gets a different construction by using it."""
    y, X = panel()
    refit_at = ols_refit_at(y, X)
    series = rolling_origin_counterfactual_errors(y, refit_at, 48, 6)
    stepwise = block_error_paths(series, horizon=6, block=block, n_sim=800,
                                 rng=np.random.default_rng(11))
    one_call = resample_cumulative_paths(y, refit_at, 48, 6, block=block,
                                         n_sim=800, seed=11)
    np.testing.assert_array_equal(one_call, stepwise)


@pytest.mark.parametrize("frac", [0.2, 0.6])
def test_the_training_fraction_reaches_the_calibration_pass(frac):
    """It decides how many windows there are, so it must not stop at the door."""
    y, X = panel()
    refit_at = ols_refit_at(y, X)
    series = rolling_origin_counterfactual_errors(y, refit_at, 48, 6,
                                                  min_train_frac=frac)
    stepwise = block_error_paths(series, horizon=6, block=0, n_sim=300,
                                 rng=np.random.default_rng(2))
    one_call = resample_cumulative_paths(y, refit_at, 48, 6, n_sim=300, seed=2,
                                         min_train_frac=frac)
    np.testing.assert_array_equal(one_call, stepwise)


def test_the_same_seed_gives_the_same_paths():
    y, X = panel()
    refit_at = ols_refit_at(y, X)
    a = resample_cumulative_paths(y, refit_at, 48, 6, n_sim=400, seed=3)
    b = resample_cumulative_paths(y, refit_at, 48, 6, n_sim=400, seed=3)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_paths():
    y, X = panel()
    refit_at = ols_refit_at(y, X)
    a = resample_cumulative_paths(y, refit_at, 48, 6, n_sim=400, seed=3)
    b = resample_cumulative_paths(y, refit_at, 48, 6, n_sim=400, seed=4)
    assert not np.array_equal(a, b)


def test_it_refits_once_per_origin_and_no_more():
    """The cost claim: the calibration pass is the only place refits happen."""
    y, X = panel()
    base, calls = ols_refit_at(y, X), []

    def spy(origin):
        calls.append(origin)
        return base(origin)

    resample_cumulative_paths(y, spy, 48, 6, n_sim=400)
    assert calls == list(origin_schedule(48, 6, 0.3))


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_block_longer_than_the_horizon_is_clamped():
    y, X = panel()
    refit_at = ols_refit_at(y, X)
    long_block = resample_cumulative_paths(y, refit_at, 48, 6, block=99,
                                           n_sim=400, seed=5)
    whole = resample_cumulative_paths(y, refit_at, 48, 6, block=6, n_sim=400, seed=5)
    np.testing.assert_array_equal(long_block, whole)


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_a_pre_period_admitting_no_origin_is_refused():
    """No calibration window means no band; that is reported, not returned empty."""
    y, X = panel(T=20)
    with pytest.raises(MlsynthDataError, match="empty"):
        resample_cumulative_paths(y, ols_refit_at(y, X), 12, 6, n_sim=100)


@pytest.mark.parametrize("bad", [0, -4, True, 2.5])
def test_a_bad_horizon_is_refused(bad):
    y, X = panel()
    with pytest.raises(MlsynthConfigError, match="horizon"):
        resample_cumulative_paths(y, ols_refit_at(y, X), 48, bad, n_sim=100)


@pytest.mark.parametrize("bad", [-1, 2.5, "3"])
def test_a_bad_block_is_refused(bad):
    y, X = panel()
    with pytest.raises(MlsynthConfigError, match="block"):
        resample_cumulative_paths(y, ols_refit_at(y, X), 48, 6, block=bad, n_sim=100)


def test_a_failed_refit_is_refused_rather_than_resampled():
    y, X = panel()
    with pytest.raises(MlsynthDataError, match="finite"):
        resample_cumulative_paths(y, lambda o: np.full_like(y, np.nan), 48, 6,
                                  n_sim=100)
