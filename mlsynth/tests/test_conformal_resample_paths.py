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
    # the composer derives the window count from the schedule and hands it on
    _m = len(list(origin_schedule(48, 6, 0.3)))
    stepwise = block_error_paths(
        series, horizon=6, block=block, n_sim=800,
        rng=np.random.default_rng(11), n_windows=_m if _m >= 2 else 0)
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
    _m = len(list(origin_schedule(48, 6, frac)))
    stepwise = block_error_paths(
        series, horizon=6, block=0, n_sim=300,
        rng=np.random.default_rng(2), n_windows=_m if _m >= 2 else 0)
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
def _raw_blocksum_sd(e, b):
    """Circular b-block sums of the centred series, with no correction applied."""
    e = np.asarray(e, dtype=float)
    e = e - e.mean()
    n = e.size
    S = np.array([e[(np.arange(b) + s) % n].sum() for s in range(n)])
    return float(S.std())


def test_the_spread_is_symmetric_about_half_the_series():
    """The identity the guard is built on. It is a property of the centred series
    -- a b-block and its (n - b) complement sum to zero between them -- so it is
    measured on the raw circular sums, not on the drawn output, which carries a
    b-dependent correction for exactly this shrinkage."""
    e = np.random.default_rng(0).normal(size=24)
    for b in (1, 2, 5, 9):
        assert _raw_blocksum_sd(e, b) == pytest.approx(
            _raw_blocksum_sd(e, e.size - b), rel=1e-9), b


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


# --------------------------------------------------------------------------- #
# The spread the centring removes
# --------------------------------------------------------------------------- #
# Centring forces the series to sum to zero, which makes its circular
# autocovariances sum to zero too. So for a b-block,
#     Var(S_b) = sum_{|k|<b} (b - |k|) chat(k) = b * v * (1 - (b-1)/(n-1)),
# and the drawn totals come out too narrow by sqrt((n-b)/(n-1)) -- 16 percent at
# b = 13 drawn from n = 47. The draw multiplies by the inverse of that factor,
# which is exactly 1 at b = 1 and so leaves the single-period construction, and
# the reference parity pinned on it, untouched.
def _blocksum_sd(series, block, horizon, n_sim=60_000, seed=0):
    p = block_error_paths(series, horizon=horizon, block=block, n_sim=n_sim,
                          rng=np.random.default_rng(seed))
    return float(p.sum(axis=1).std())


@pytest.mark.parametrize("n,b", [(47, 13), (47, 5), (60, 6), (100, 13)])
def test_block_sums_carry_the_spread_independent_periods_would_give(n, b):
    """White noise: a b-period total has sd sqrt(b) * sd(x). Uncorrected the draw
    lands about sqrt((n-b)/(n-1)) of that."""
    rng = np.random.default_rng(4)
    ratios = []
    for _ in range(12):
        x = rng.normal(size=n)
        target = np.sqrt(b) * (x - x.mean()).std()
        ratios.append(_blocksum_sd(x, b, b, n_sim=20_000) / target)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.06), np.mean(ratios)


def test_the_correction_is_exactly_one_at_block_one():
    """b = 1 has no centring shrinkage, so the single-period draw -- the one the
    Wheeler parity benchmark pins -- must be bit-for-bit what it always was."""
    x = np.random.default_rng(5).normal(size=40)
    paths = block_error_paths(x, horizon=6, block=1, n_sim=5000,
                              rng=np.random.default_rng(0))
    drawn = np.unique(np.round(np.abs(paths.ravel()), 12))
    expected = np.unique(np.round(np.abs(x - x.mean()), 12))
    # every drawn value is an element of the centred series, unscaled
    assert np.isin(drawn, expected).all()


def test_the_correction_holds_the_per_period_scale_across_block_lengths():
    """Uncorrected, sd(S_b)/sqrt(b) falls away as b/n grows -- that is the
    shrinkage. Corrected, it stays near sd(x) at every block length. Averaged over
    series, since one realisation's sample autocovariances are noisy enough to
    swamp the effect."""
    rng = np.random.default_rng(6)
    n, blocks = 48, (1, 4, 12, 24)
    got = {b: [] for b in blocks}
    for _ in range(10):
        x = rng.normal(size=n)
        target = (x - x.mean()).std()
        for b in blocks:
            got[b].append(_blocksum_sd(x, b, b, n_sim=20_000) / (np.sqrt(b) * target))
    means = [float(np.mean(got[b])) for b in blocks]
    assert all(abs(m - 1.0) < 0.08 for m in means), dict(zip(blocks, means))


# --------------------------------------------------------------------------- #
# Two-level draw: the window's own level, and the fluctuation about it
# --------------------------------------------------------------------------- #
# A horizon total is H times the mean error over that horizon, so it is driven by
# the calibration window's LEVEL, not by fluctuation about it. Centring removes the
# grand level, and a block of length L drawn from m concatenated windows usually
# straddles a boundary and averages two levels, diluting what is left. Passing
# ``n_windows`` splits the two apart: each drawn path gets one window's level plus
# blocks of the within-window residuals, so the level enters at full weight.
def _two_level_series(m, L, level_sd, noise_sd, seed=0):
    """m windows of length L, each with its own level."""
    rng = np.random.default_rng(seed)
    levels = rng.normal(0, level_sd, m)
    return np.concatenate([lv + rng.normal(0, noise_sd, L) for lv in levels]), levels


def test_the_level_reaches_the_total_at_full_weight():
    """With a level ten times the noise, the H-period total must be dominated by
    it: sd(total) should be near H * sd(level), not a diluted fraction."""
    m, L = 6, 10
    s, levels = _two_level_series(m, L, level_sd=1.0, noise_sd=0.1, seed=3)
    paths = block_error_paths(s, horizon=L, block=L, n_sim=40_000,
                              rng=np.random.default_rng(0), n_windows=m)
    got = paths.sum(axis=1).std()
    target = L * np.std(levels - levels.mean(), ddof=1)
    assert got == pytest.approx(target, rel=0.15), (got, target)


def test_without_n_windows_the_level_is_diluted():
    """The contrast that motivates the parameter: the same series drawn flat gives
    a materially smaller total, because most blocks straddle a boundary."""
    m, L = 6, 10
    s, _ = _two_level_series(m, L, level_sd=1.0, noise_sd=0.1, seed=3)
    flat = block_error_paths(s, horizon=L, block=L, n_sim=40_000,
                             rng=np.random.default_rng(0)).sum(axis=1).std()
    split = block_error_paths(s, horizon=L, block=L, n_sim=40_000,
                              rng=np.random.default_rng(0), n_windows=m).sum(axis=1).std()
    assert split > 1.25 * flat, (split, flat)


def test_a_flat_series_is_unchanged_by_splitting_it():
    """When the windows share a level there is nothing to separate, so the two
    draws must agree."""
    rng = np.random.default_rng(7)
    s = rng.normal(0, 1.0, 60)
    a = block_error_paths(s, horizon=10, block=10, n_sim=40_000,
                          rng=np.random.default_rng(0)).sum(axis=1).std()
    b = block_error_paths(s, horizon=10, block=10, n_sim=40_000,
                          rng=np.random.default_rng(0), n_windows=6).sum(axis=1).std()
    assert b == pytest.approx(a, rel=0.20), (a, b)


def test_one_window_cannot_identify_a_level():
    """With a single window every level deviation is zero by construction, so the
    variance the total needs is unidentified. The draw says so."""
    s = np.random.default_rng(8).normal(size=30)
    with pytest.raises(MlsynthDataError, match="two windows|n_windows"):
        block_error_paths(s, horizon=6, block=6, n_sim=100, n_windows=1)


@pytest.mark.parametrize("bad", [-1, 2.5, "3", True])
def test_a_bad_window_count_is_refused(bad):
    s = np.random.default_rng(9).normal(size=30)
    with pytest.raises(MlsynthConfigError, match="n_windows"):
        block_error_paths(s, horizon=6, block=6, n_sim=100, n_windows=bad)


def test_more_windows_than_periods_is_refused():
    s = np.random.default_rng(10).normal(size=8)
    with pytest.raises(MlsynthDataError, match="n_windows"):
        block_error_paths(s, horizon=4, block=2, n_sim=100, n_windows=9)


def test_the_level_carries_m_minus_one_degrees_of_freedom():
    """The m window means are deviations from the grand mean, so they sum to zero
    and carry m-1 degrees of freedom. Drawing from them as if they were m
    independent observations understates the level by sqrt((m-1)/m) -- eighteen
    percent at three windows, which is what a blank window of three horizons gives.
    Measured at m=3, where the two conventions are furthest apart."""
    m, L = 3, 8
    got, ddof1, ddof0 = [], [], []
    for seed in range(12):
        s, levels = _two_level_series(m, L, level_sd=1.0, noise_sd=0.05, seed=seed)
        paths = block_error_paths(s, horizon=L, block=2, n_sim=20_000,
                                  rng=np.random.default_rng(seed), n_windows=m)
        got.append(paths.sum(axis=1).std())
        c = levels - levels.mean()
        ddof1.append(L * np.std(c, ddof=1))
        ddof0.append(L * np.std(c, ddof=0))
    r1 = float(np.mean(got) / np.mean(ddof1))
    r0 = float(np.mean(got) / np.mean(ddof0))
    assert abs(r1 - 1.0) < 0.10, ("unbiased target", r1)
    assert r0 > 1.12, ("must exceed the ddof=0 target", r0)


# --------------------------------------------------------------------------- #
# Shrinking the level toward zero when the windows do not disagree
# --------------------------------------------------------------------------- #
# The m window means scatter even when every window shares a level, because each
# mean is an average of L noisy periods. Their variance estimates
#     sigma_level^2 + sigma_within^2 / L,
# so drawing from them whole charges the band for a level that is not there. The
# draw subtracts the noise floor -- the ANOVA estimator
# max(0, s2_between - s2_within / L) -- which leaves a strongly levelled series
# almost untouched and collapses the term to nothing on a flat one.
def _flat_ratio(m, width=10, horizon=10, block=5, reps=10, seed=11):
    """Split-draw total over flat-draw total, on a series with no true level."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        s = rng.normal(0, 1.0, m * width)
        flat = block_error_paths(s, horizon=horizon, block=block, n_sim=8_000,
                                 rng=np.random.default_rng(0)).sum(axis=1).std()
        split = block_error_paths(s, horizon=horizon, block=block, n_sim=8_000,
                                  rng=np.random.default_rng(0),
                                  n_windows=m).sum(axis=1).std()
        out.append(split / flat)
    return float(np.mean(out))


def test_a_flat_series_keeps_most_of_the_level_away():
    """No true level, so the split should stay near the flat draw. It does not
    reach it exactly: ``max(0, .)`` truncates the negative side of an estimator
    whose mean is zero, so the floored estimate is biased up. Unshrunk the level
    inflates the total by about 1.40 at any m -- the level variance and the
    fluctuation variance are then equal by construction -- and shrinking takes that
    to roughly 1.12 at six windows."""
    assert _flat_ratio(6) < 1.25, _flat_ratio(6)


def test_the_floor_bias_falls_as_windows_are_added():
    """The bias is a small-sample property of the estimator, so more windows must
    shrink it. This is what a longer pre-period buys a cumulative band."""
    few, many = _flat_ratio(6), _flat_ratio(30)
    assert many < few, (few, many)
    assert many < 1.12, many


def test_a_strongly_levelled_series_keeps_its_level():
    """The noise floor is small against a real level, so the term survives nearly
    whole and the total is still driven by it."""
    m, L = 6, 10
    s, levels = _two_level_series(m, L, level_sd=1.0, noise_sd=0.1, seed=3)
    paths = block_error_paths(s, horizon=L, block=L, n_sim=40_000,
                              rng=np.random.default_rng(0), n_windows=m)
    got = paths.sum(axis=1).std()
    target = L * np.std(levels - levels.mean(), ddof=1)
    assert got == pytest.approx(target, rel=0.15), (got, target)


def test_the_shrinkage_is_monotone_in_how_much_the_windows_disagree():
    m, L = 6, 10
    widths = []
    for level_sd in (0.0, 0.2, 0.5, 1.0):
        s, _ = _two_level_series(m, L, level_sd=level_sd, noise_sd=0.3, seed=5)
        widths.append(block_error_paths(s, horizon=L, block=L, n_sim=20_000,
                                        rng=np.random.default_rng(0),
                                        n_windows=m).sum(axis=1).std())
    assert all(np.diff(widths) > 0), widths


def test_the_level_variance_never_goes_negative():
    """s2_between can fall below the noise floor by chance; the estimator floors
    at zero instead of producing a negative variance."""
    rng = np.random.default_rng(13)
    for seed in range(6):
        s = rng.normal(0, 1.0, 24)
        p = block_error_paths(s, horizon=4, block=2, n_sim=4000,
                              rng=np.random.default_rng(seed), n_windows=6)
        assert np.isfinite(p).all()
