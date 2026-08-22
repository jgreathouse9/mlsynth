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


# --------------------------------------------------------------------------- #
# A block as long as the series it is drawn from
# --------------------------------------------------------------------------- #
# A circular block of length b over an n-period series wraps, so at b == n every
# block is a rotation of the whole series and every path sums to the series total.
# The series is centred first, so that total is exactly zero: every path sums to
# zero, and the quantile of their magnitudes is zero. A zero-width band asserts
# perfect certainty about the accumulated effect, which is a worse answer than the
# infinite one it was reached for, so the draw refuses instead.
def test_a_block_as_long_as_the_series_is_refused():
    e = np.random.default_rng(0).normal(size=6)
    with pytest.raises(MlsynthDataError, match="block"):
        block_error_paths(e, horizon=6, block=6, n_sim=100,
                          rng=np.random.default_rng(0))


def test_a_block_longer_than_the_series_is_refused():
    """``resolve_block`` clamps the request to the horizon, so a long horizon is
    the way to reach this: the clamped block still exceeds the series."""
    e = np.random.default_rng(0).normal(size=4)
    with pytest.raises(MlsynthDataError, match="block"):
        block_error_paths(e, horizon=10, block=0, n_sim=100,
                          rng=np.random.default_rng(0))


def test_the_refusal_names_both_lengths():
    e = np.random.default_rng(0).normal(size=6)
    with pytest.raises(MlsynthDataError) as exc:
        block_error_paths(e, horizon=6, block=6, n_sim=100)
    msg = str(exc.value)
    assert "6" in msg and ("shorter block" in msg or "block" in msg)


def test_the_longest_admissible_block_still_draws():
    """Half the series is the boundary, and it must work: the guard is against a
    block whose length has stopped meaning what it says, not against long blocks."""
    e = np.random.default_rng(0).normal(size=12)
    paths = block_error_paths(e, horizon=12, block=6, n_sim=500,
                              rng=np.random.default_rng(0))
    assert paths.shape == (500, 12)
    assert np.abs(paths.sum(axis=1)).max() > 0.0


# --------------------------------------------------------------------------- #
# A block past half the series
# --------------------------------------------------------------------------- #
# Over a centred series the circular sum of a b-block is minus the circular sum of
# the complementary (n - b)-block, so the two have exactly equal spread. Block
# length therefore stops meaning "how much serial correlation is carried" at
# b = n/2 and reverses: past the midpoint a longer block draws a NARROWER total,
# mirroring a shorter one, until at b = n it draws nothing at all.
def test_the_spread_is_symmetric_about_half_the_series():
    """The identity the guard is built on, measured."""
    e = np.random.default_rng(0).normal(size=24)
    def sd(b):
        return block_error_paths(e, horizon=b, block=b, n_sim=100_000,
                                 rng=np.random.default_rng(1)).sum(axis=1).std()
    for b in (1, 2, 5):
        assert sd(b) == pytest.approx(_complement_sd(e, b), rel=0.02), b


def _complement_sd(e, b):
    """The (n - b)-block spread, computed without the guard in the way."""
    e = np.asarray(e, dtype=float)
    e = e - e.mean()
    n, k = e.size, e.size - b
    rng = np.random.default_rng(1)
    s = rng.integers(0, n, size=100_000)
    idx = (s[:, None] + np.arange(k)[None, :]) % n
    return e[idx].sum(axis=1).std()


@pytest.mark.parametrize("block", [7, 9, 11])
def test_a_block_past_half_the_series_is_refused(block):
    e = np.random.default_rng(0).normal(size=12)
    with pytest.raises(MlsynthDataError, match="half"):
        block_error_paths(e, horizon=12, block=block, n_sim=100,
                          rng=np.random.default_rng(0))


def test_the_refusal_names_the_admissible_block():
    e = np.random.default_rng(0).normal(size=12)
    with pytest.raises(MlsynthDataError) as exc:
        block_error_paths(e, horizon=12, block=9, n_sim=100)
    assert "6" in str(exc.value)


def test_a_horizon_longer_than_the_series_draws_from_short_blocks():
    """H > n is fine as long as the BLOCK fits: blocks are drawn and concatenated,
    so a twelve-period path can come from a six-period calibration series."""
    e = np.random.default_rng(0).normal(size=6)
    paths = block_error_paths(e, horizon=12, block=2, n_sim=500,
                              rng=np.random.default_rng(0))
    assert paths.shape == (500, 12)
    assert np.quantile(np.abs(paths.sum(axis=1)), 0.95) > 0.0


def test_a_single_calibration_error_says_what_is_wrong():
    """One error carries no spread: centring it leaves exactly zero, and there is
    no shorter block to fall back on. The refusal says that instead of suggesting
    a block length below one."""
    with pytest.raises(MlsynthDataError) as exc:
        block_error_paths(np.array([2.0]), horizon=3, block=0, n_sim=10)
    msg = str(exc.value)
    assert "block <= 0" not in msg
    assert "single" in msg or "one" in msg
