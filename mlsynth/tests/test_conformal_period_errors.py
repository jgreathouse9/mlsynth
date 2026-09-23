"""Per-period out-of-sample errors from the rolling-origin calibration pass.

``scores.rolling_origin_block_sums`` reduces each calibration window to the sum of
its ``L`` out-of-sample residuals, which is the conformity score the split-conformal
band needs. A resampled band needs what that sum destroys: the ordering of the
periods inside a window, and across consecutive windows.

``rolling_origin_period_errors`` returns the blocks concatenated in origin order.
Because origins step by a whole horizon and are visited in increasing order, the
concatenation is a contiguous stretch of out-of-sample periods -- origin ``o``
scores ``o .. o + L - 1`` and the next origin picks up at ``o + L`` -- so serial
correlation in the returned series is the panel's own and a block bootstrap over it
means something.

The two readers share one origin schedule, so they cannot drift apart in which
windows they score. That equivalence is asserted here and again as a property in
``test_conformal_period_errors_properties.py``.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import (
    MIN_TRAIN_PERIODS,
    rolling_origin_block_sums,
    rolling_origin_period_errors,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def panel(T=60, J=3, seed=0):
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(size=(T, J))
    y = Y0 @ np.array([0.5, 0.3, 0.2]) + rng.normal(scale=0.1, size=T)
    return y, Y0


def fixed_weight_fn(w):
    """A refit that ignores its training window; makes the errors predictable."""
    return lambda keep_idx: np.asarray(w, dtype=float)


def ols_weight_fn(y, Y0):
    def _fn(keep_idx):
        idx = np.asarray(keep_idx)
        beta, *_ = np.linalg.lstsq(Y0[idx], y[idx], rcond=None)
        return np.asarray(beta, float)
    return _fn


def expected_origins(pre_periods, horizon, min_train_frac):
    start = max(MIN_TRAIN_PERIODS, int(pre_periods * float(min_train_frac)))
    return list(range(start, int(pre_periods) - int(horizon) + 1, int(horizon)))


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_returns_one_error_per_scored_period():
    y, Y0 = panel()
    origins = expected_origins(48, 6, 0.3)
    out = rolling_origin_period_errors(y, Y0, 48, 6, ols_weight_fn(y, Y0))
    assert out.ndim == 1
    assert out.dtype == float
    assert out.size == len(origins) * 6


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_the_series_is_the_blocks_concatenated_in_origin_order():
    y, Y0 = panel()
    w = np.array([0.5, 0.3, 0.2])
    out = rolling_origin_period_errors(y, Y0, 48, 6, fixed_weight_fn(w))
    wanted = np.concatenate([y[o:o + 6] - Y0[o:o + 6] @ w
                             for o in expected_origins(48, 6, 0.3)])
    assert out == pytest.approx(wanted)


def test_the_scored_periods_are_contiguous():
    """Origins step by a whole horizon, so the windows abut with no gap or overlap."""
    y, Y0 = panel()
    origins = expected_origins(48, 6, 0.3)
    covered = np.concatenate([np.arange(o, o + 6) for o in origins])
    assert np.array_equal(covered, np.arange(origins[0], origins[0] + covered.size))


def test_the_refit_is_trained_only_on_periods_strictly_before_its_origin():
    y, Y0 = panel()
    seen = []

    def spy(keep_idx):
        seen.append(np.asarray(keep_idx).copy())
        return np.array([0.5, 0.3, 0.2])

    rolling_origin_period_errors(y, Y0, 48, 6, spy)
    for call, origin in zip(seen, expected_origins(48, 6, 0.3)):
        assert np.array_equal(call, np.arange(origin))


def test_summing_each_block_reproduces_the_split_conformal_scores():
    """The headline equivalence: one origin schedule, two readers of it."""
    y, Y0 = panel(T=80, seed=4)
    wf = ols_weight_fn(y, Y0)
    per_period = rolling_origin_period_errors(y, Y0, 64, 8, wf, min_train_frac=0.25)
    sums = rolling_origin_block_sums(y, Y0, 64, 8, wf, min_train_frac=0.25)
    assert per_period.size == sums.size * 8
    assert per_period.reshape(-1, 8).sum(axis=1) == pytest.approx(sums)


def test_an_exactly_reproducible_panel_scores_zero_error():
    """A donor combination that reproduces the outcome leaves nothing to calibrate."""
    rng = np.random.default_rng(2)
    Y0 = rng.normal(size=(60, 3))
    w = np.array([0.4, 0.4, 0.2])
    y = Y0 @ w
    out = rolling_origin_period_errors(y, Y0, 48, 6, fixed_weight_fn(w))
    assert out == pytest.approx(np.zeros_like(out), abs=1e-12)


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_pre_period_too_short_for_one_window_returns_nothing():
    y, Y0 = panel(T=20)
    out = rolling_origin_period_errors(y, Y0, 12, 6, ols_weight_fn(y, Y0))
    assert out.size == 0
    assert out.dtype == float


def test_the_absolute_training_floor_beats_a_tiny_fraction():
    """A fraction below the floor cannot buy windows the floor forbids."""
    y, Y0 = panel()
    out = rolling_origin_period_errors(y, Y0, 48, 6, ols_weight_fn(y, Y0),
                                       min_train_frac=0.0)
    assert out.size == len(expected_origins(48, 6, 0.0)) * 6
    assert expected_origins(48, 6, 0.0)[0] == MIN_TRAIN_PERIODS


def test_exactly_one_admissible_origin_gives_exactly_one_block():
    y, Y0 = panel(T=30)
    out = rolling_origin_period_errors(y, Y0, 16, 6, ols_weight_fn(y, Y0),
                                       min_train_frac=0.0)
    assert out.size == 6


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -4, True, 2.5, "6", None])
def test_a_horizon_that_is_not_a_positive_integer_is_refused(bad):
    y, Y0 = panel()
    with pytest.raises(MlsynthConfigError, match="horizon"):
        rolling_origin_period_errors(y, Y0, 48, bad, ols_weight_fn(y, Y0))


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.3", None])
def test_a_min_train_frac_outside_the_unit_interval_is_refused(bad):
    y, Y0 = panel()
    with pytest.raises(MlsynthConfigError, match="min_train_frac"):
        rolling_origin_period_errors(y, Y0, 48, 6, ols_weight_fn(y, Y0),
                                     min_train_frac=bad)


def test_a_pre_period_longer_than_the_panel_is_refused():
    y, Y0 = panel(T=40)
    with pytest.raises(MlsynthDataError, match="pre_periods"):
        rolling_origin_period_errors(y, Y0, 80, 6, ols_weight_fn(y, Y0))


def test_a_donor_block_misaligned_with_the_outcome_is_refused():
    y, Y0 = panel(T=40)
    with pytest.raises(MlsynthDataError, match="align"):
        rolling_origin_period_errors(y, Y0[:30], 24, 6, ols_weight_fn(y, Y0))
