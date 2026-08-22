"""The weight-form composer: donor weights in, resampled paths out.

Two calibration readers exist because two kinds of estimator exist.
:func:`rolling_origin_counterfactual_errors` serves a fit that returns a
counterfactual -- an intercept, standardized donors, or a selection with no single
weight vector to multiply. :func:`rolling_origin_period_errors` serves a fit that
does return donor weights, which is the classical synthetic control and what
VanillaSC's refit hands back.

Each needs the same second step, so each gets a composer.
``resample_cumulative_paths_from_weights`` is the weight-form sibling of
``resample_cumulative_paths``, and this suite pins that the two agree whenever the
fit is a bare linear combination -- the case where either reader applies.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (
    block_error_paths,
    resample_cumulative_paths,
    resample_cumulative_paths_from_weights,
    rolling_origin_period_errors,
)


def panel(T=60, J=3, seed=0):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T, J))
    y = Y0 @ np.array([0.5, 0.3, 0.2]) + rng.normal(scale=0.1, size=T)
    return y, Y0


def ols_weight_fn(y, Y0):
    def _fn(keep_idx):
        idx = np.asarray(keep_idx)
        beta, *_ = np.linalg.lstsq(Y0[idx], y[idx], rcond=None)
        return np.asarray(beta, float)
    return _fn


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_returns_one_row_per_simulation_and_one_column_per_horizon():
    y, Y0 = panel()
    paths = resample_cumulative_paths_from_weights(
        y, Y0, 48, 6, ols_weight_fn(y, Y0), n_sim=500)
    assert paths.shape == (500, 6)
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block", [0, 1, 3])
def test_it_is_exactly_the_two_step_composition(block):
    y, Y0 = panel()
    wf = ols_weight_fn(y, Y0)
    series = rolling_origin_period_errors(y, Y0, 48, 6, wf)
    stepwise = block_error_paths(series, horizon=6, block=block, n_sim=800,
                                 rng=np.random.default_rng(9))
    one_call = resample_cumulative_paths_from_weights(
        y, Y0, 48, 6, wf, block=block, n_sim=800, seed=9)
    np.testing.assert_array_equal(one_call, stepwise)


def test_the_two_composers_agree_on_a_bare_linear_fit():
    """Where either reader applies, they must not disagree."""
    y, Y0 = panel()
    wf = ols_weight_fn(y, Y0)
    from_w = resample_cumulative_paths_from_weights(y, Y0, 48, 6, wf,
                                                    n_sim=600, seed=4)
    from_cf = resample_cumulative_paths(
        y, lambda o: Y0 @ wf(np.arange(o)), 48, 6, n_sim=600, seed=4)
    np.testing.assert_allclose(from_w, from_cf, rtol=1e-9, atol=1e-11)


def test_the_training_fraction_reaches_the_calibration_pass():
    y, Y0 = panel()
    wf = ols_weight_fn(y, Y0)
    series = rolling_origin_period_errors(y, Y0, 48, 6, wf, min_train_frac=0.6)
    stepwise = block_error_paths(series, horizon=6, block=0, n_sim=300,
                                 rng=np.random.default_rng(2))
    one_call = resample_cumulative_paths_from_weights(
        y, Y0, 48, 6, wf, n_sim=300, seed=2, min_train_frac=0.6)
    np.testing.assert_array_equal(one_call, stepwise)


def test_the_same_seed_gives_the_same_paths():
    y, Y0 = panel()
    wf = ols_weight_fn(y, Y0)
    a = resample_cumulative_paths_from_weights(y, Y0, 48, 6, wf, n_sim=400, seed=3)
    b = resample_cumulative_paths_from_weights(y, Y0, 48, 6, wf, n_sim=400, seed=3)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_it_needs_far_fewer_windows_than_a_split_band_does():
    """The practical gain: m windows give m*L per-period errors to draw from.

    A split band at alpha needs ceil(1/alpha) - 1 windows before its order
    statistic exists at all. The resampled band draws from the periods inside
    those windows, so two windows already supply twelve values at this horizon.
    """
    y, Y0 = panel()
    series = rolling_origin_period_errors(y, Y0, 24, 6, ols_weight_fn(y, Y0),
                                          min_train_frac=0.0)
    assert series.size // 6 < 9                      # too few windows for alpha=0.10
    paths = resample_cumulative_paths_from_weights(
        y, Y0, 24, 6, ols_weight_fn(y, Y0), n_sim=200, min_train_frac=0.0)
    assert np.isfinite(paths).all()                  # and yet a finite draw


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_a_pre_period_admitting_no_window_is_refused():
    y, Y0 = panel(T=20)
    with pytest.raises(MlsynthDataError, match="empty"):
        resample_cumulative_paths_from_weights(y, Y0, 12, 6, ols_weight_fn(y, Y0),
                                               n_sim=100)


@pytest.mark.parametrize("bad", [0, -4, True, 2.5])
def test_a_bad_horizon_is_refused(bad):
    y, Y0 = panel()
    with pytest.raises(MlsynthConfigError, match="horizon"):
        resample_cumulative_paths_from_weights(y, Y0, 48, bad,
                                               ols_weight_fn(y, Y0), n_sim=100)


def test_a_misaligned_donor_block_is_refused():
    y, Y0 = panel()
    with pytest.raises(MlsynthDataError, match="align"):
        resample_cumulative_paths_from_weights(y, Y0[:30], 24, 6,
                                               ols_weight_fn(y, Y0), n_sim=100)
