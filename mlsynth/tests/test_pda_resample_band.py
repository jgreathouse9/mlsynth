"""Block-resampled error paths for a PDA cumulative band (Wheeler's construction).

``cumulative_supt_band`` already accepts an ``(B, H)`` matrix of post-period error
paths and builds the band from it; today those paths come from
``inferutils.pda_prediction_intervals``, which spends ``n_boot`` refits (999 by
default). This suite pins a second producer of that matrix, costing one refit per
rolling origin instead.

The construction is the one Andrew Wheeler uses in LassoSynth: conformity scores
resampled and accumulated into a cumulative path. Wheeler symmetrises by mirroring
the pool, ``np.concatenate([cs, -cs])``, and draws periods independently. Drawing
whole blocks generalises that to a panel whose period errors are serially
correlated, and flipping each block's sign is the same symmetrisation applied to a
block instead of a period.

The two coincide exactly at ``block = 1``: the multiset ``{+e_i, -e_i}`` is the
multiset ``{+|e_i|, -|e_i|}``, so drawing a centred score and flipping its sign and
drawing from Wheeler's mirrored pool are the same distribution. That is the parity
test below, and it is the reason the generalisation is safe to make.

Levels: smoke, unit invariants (including reference parity), edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import block_error_paths, resolve_block

ALPHA = 0.10


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ar1(n, rho, *, sd=1.0, seed=0):
    """An AR(1) series, started from its stationary distribution."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    e[0] = rng.normal(0.0, sd / np.sqrt(1.0 - rho ** 2))
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0.0, sd)
    return e


def ar1_sum_inflation(rho, H):
    """Var(sum of H AR(1) terms) / (H * gamma_0) -- the closed form to match."""
    k = np.arange(1, H)
    return float((H + 2.0 * np.sum((H - k) * rho ** k)) / H)


def wheeler_band(scores, tau, *, alpha=ALPHA, n_sim=200_000, seed=0):
    """Wheeler's LassoSynth cumulative band, transcribed.

    ``pool = concatenate([cs, -cs])``; draw one pool element per post period,
    independently; accumulate; take equal-tailed empirical quantiles of the
    accumulated paths. The reference this suite is written against.
    """
    rng = np.random.default_rng(seed)
    cs = np.abs(np.asarray(scores, dtype=float).ravel())
    pool = np.concatenate([cs, -cs])
    L = np.asarray(tau, dtype=float).size
    sims = np.cumsum(np.asarray(tau, float).reshape(L, 1)
                     + rng.choice(pool, (L, n_sim)), axis=0)
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=1)
    return lo, hi


def band_from_paths(paths, tau, *, alpha=ALPHA):
    """The same equal-tailed band, read off an ``(n_sim, H)`` error-path matrix."""
    cum = np.cumsum(np.asarray(tau, float)[None, :] + paths, axis=1)
    lo, hi = np.quantile(cum, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_returns_one_row_per_simulation_and_one_column_per_horizon():
    rng = np.random.default_rng(0)
    paths = block_error_paths(ar1(80, 0.3), horizon=6, block=3, n_sim=500, rng=rng)
    assert paths.shape == (500, 6)
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
def test_block_one_reproduces_wheelers_band():
    """The parity claim: at block 1 this is Wheeler's construction exactly."""
    scores = ar1(60, 0.0, seed=3)
    scores = scores - scores.mean()          # the pools coincide on centred scores
    tau = np.linspace(2.0, 5.0, 8)

    lo_w, hi_w = wheeler_band(scores, tau, n_sim=200_000, seed=0)
    paths = block_error_paths(scores, horizon=8, block=1, n_sim=200_000,
                              rng=np.random.default_rng(1))
    lo_m, hi_m = band_from_paths(paths, tau)

    scale = np.abs(hi_w - lo_w)
    assert np.max(np.abs(lo_m - lo_w) / scale) < 0.02
    assert np.max(np.abs(hi_m - hi_w) / scale) < 0.02


def test_at_one_horizon_the_draws_are_the_symmetrised_score_pool():
    """With a single period there is no accumulation, so the draw is the pool."""
    scores = np.array([-3.0, -1.0, 2.0, 2.0])   # already centred
    paths = block_error_paths(scores, horizon=1, block=1, n_sim=200_000,
                              rng=np.random.default_rng(0))
    drawn = np.sort(np.unique(np.round(paths.ravel(), 12)))
    expected = np.sort(np.unique(np.round(np.concatenate([scores, -scores]), 12)))
    assert np.array_equal(drawn, expected)


def test_independent_draws_are_too_narrow_under_positive_autocorrelation():
    """Blocks keep the serial correlation an independent draw throws away."""
    rho, H = 0.7, 6
    e = ar1(4000, rho, seed=11)
    iid = block_error_paths(e, horizon=H, block=1, n_sim=40_000,
                            rng=np.random.default_rng(0))
    blocked = block_error_paths(e, horizon=H, block=H, n_sim=40_000,
                                rng=np.random.default_rng(0))
    # the inflation lives in the accumulated total, which is what the band
    # covers; the final period's marginal is the same draw either way
    ratio = blocked.sum(axis=1).std() / iid.sum(axis=1).std()
    assert ratio > 1.4
    assert abs(ratio - np.sqrt(ar1_sum_inflation(rho, H))) < 0.15 * ratio


def test_the_two_agree_when_the_errors_are_white():
    """With nothing to preserve, blocking changes nothing."""
    e = ar1(4000, 0.0, seed=5)
    iid = block_error_paths(e, horizon=6, block=1, n_sim=40_000,
                            rng=np.random.default_rng(0))
    blocked = block_error_paths(e, horizon=6, block=6, n_sim=40_000,
                                rng=np.random.default_rng(0))
    assert abs(blocked.sum(axis=1).std() / iid.sum(axis=1).std() - 1.0) < 0.08


def test_the_draws_take_their_scale_from_the_centred_series():
    """A fit sitting high shifts every error alike; the band is not charged for it.

    The assertion is on spread and not on mean: the sign flips zero the mean of
    the draws whether or not the series was centred, so a mean-zero check would
    pass on an uncentred series too. What an uncentred draw inflates is the
    scale -- a shift of 10 on a unit-variance series would carry a standard
    deviation of about 10 into every period the band accumulates.
    """
    noise = ar1(2000, 0.2, seed=7)
    paths = block_error_paths(noise + 10.0, horizon=5, block=2, n_sim=40_000,
                              rng=np.random.default_rng(0))
    assert abs(paths.std() / noise.std() - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_block_zero_means_the_whole_horizon():
    assert resolve_block(0, 8) == 8


def test_a_block_longer_than_the_horizon_is_clamped_to_it():
    assert resolve_block(20, 6) == 6


def test_a_series_shorter_than_the_block_still_draws_by_wrapping():
    """Blocks are circular, so a short series wraps instead of running out."""
    paths = block_error_paths(np.array([1.0, -1.0, 2.0]), horizon=8, block=5,
                              n_sim=100, rng=np.random.default_rng(0))
    assert paths.shape == (100, 8)
    assert np.isfinite(paths).all()


def test_non_finite_entries_are_dropped_before_drawing():
    e = np.array([1.0, np.nan, -1.0, np.inf, 2.0, -2.0])
    paths = block_error_paths(e, horizon=3, block=1, n_sim=1000,
                              rng=np.random.default_rng(0))
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_an_empty_series_is_refused():
    with pytest.raises(MlsynthDataError, match="empty"):
        block_error_paths(np.array([]), horizon=4, block=1, n_sim=10,
                          rng=np.random.default_rng(0))


def test_a_series_of_only_non_finite_values_is_refused():
    with pytest.raises(MlsynthDataError, match="empty"):
        block_error_paths(np.array([np.nan, np.inf]), horizon=4, block=1, n_sim=10,
                          rng=np.random.default_rng(0))


@pytest.mark.parametrize("bad", [-1, -8])
def test_a_negative_block_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="block"):
        resolve_block(bad, 8)


@pytest.mark.parametrize("bad", [True, 2.5, "4", None])
def test_a_non_integer_block_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="block"):
        resolve_block(bad, 8)


@pytest.mark.parametrize("bad", [0, -3])
def test_a_non_positive_horizon_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="horizon"):
        block_error_paths(np.array([1.0, -1.0]), horizon=bad, block=1, n_sim=10,
                          rng=np.random.default_rng(0))
