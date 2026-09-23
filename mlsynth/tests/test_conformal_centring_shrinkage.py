"""Restoring the block-sum spread that centring takes away.

``block_error_paths`` centres the calibration series before drawing, and it has
to: a fit sitting a little high shifts every one of its errors by the same
amount, and an uncentred draw would charge the band for that shift once per
period it accumulates.

Centring is not free. A series that sums to zero has circular autocovariances
that sum to zero too, so for a ``b``-block

    Var(S_b) = sum_{|k| < b} (b - |k|) c(k) = b v (1 - (b - 1) / (n - 1)),

and the drawn totals come out narrow by ``sqrt((n - b) / (n - 1))``. At the
thirteen-period block drawn from forty-seven calibration periods that PDA and
VanillaSC use, that is sixteen percent, and every band built on those draws is
narrow by the same factor -- which is under-coverage, in the direction nobody
wants.

The existing ``b <= n / 2`` guard catches the far end of this same curve, where
the spread collapses to nothing. The shrinkage is continuous and present at every
``b``; the guard only refuses where it becomes absurd.

The draw now multiplies by the inverse. The factor is exactly one at ``b = 1``,
so Wheeler's single-period construction is untouched and so is the parity pinned
on it.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.conformal import block_error_paths


def _totals(series, horizon, block, n_sim=4000, seed=0):
    paths = block_error_paths(series, horizon=horizon, block=block, n_sim=n_sim,
                              rng=np.random.default_rng(seed))
    return paths.sum(axis=1)


def _drawn_variance(n, b, reps=250, seed=0):
    """Variance of the drawn b-block total, over many iid series of length n."""
    rng = np.random.default_rng(seed)
    tot = [_totals(rng.standard_normal(n), b, b, n_sim=200, seed=1) for _ in range(reps)]
    return float(np.var(np.concatenate(tot)))


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_it_still_draws_finite_paths():
    e = np.random.default_rng(0).standard_normal(47)
    paths = block_error_paths(e, horizon=13, block=13, n_sim=100,
                              rng=np.random.default_rng(0))
    assert paths.shape == (100, 13)
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,b", [(47, 13), (100, 25), (200, 50), (60, 5)])
def test_the_drawn_block_total_carries_the_series_variance(n, b):
    """The defect this fixes. A b-block of an iid series has variance b*v; the
    centred draw delivered b*v*(n-b)/(n-1) instead, and the band inherited the
    shortfall."""
    assert _drawn_variance(n, b) == pytest.approx(float(b), rel=0.07)


def test_single_period_draws_are_untouched():
    """The factor is exactly one at b = 1, so Wheeler's construction and the
    parity pinned on it cannot move."""
    e = np.random.default_rng(3).standard_normal(40)
    kw = dict(horizon=8, block=1, n_sim=64)
    got = block_error_paths(e, rng=np.random.default_rng(5), **kw)
    centred = e - e.mean()
    pool = np.concatenate([centred, -centred])
    for value in np.unique(got):
        assert np.isclose(np.abs(pool - value).min(), 0.0, atol=1e-12)


def test_the_corrected_spread_is_flat_in_the_block_length():
    """``sqrt((n-1)/(n-b))`` is increasing in b, so a longer block is scaled up
    harder -- which is what makes the drawn spread per ``sqrt(b)`` flat instead of
    falling away as b grows.

    Averaged over series, not measured on one. A single fixed series carries its
    own realised autocorrelation, and a long block picks that up: on one draw of
    sixty iid normals the ratio across b runs to 1.76, which says something about
    that series and nothing about the correction. Over many series the realised
    dependence averages out and only the scaling is left.
    """
    n = 60
    rng = np.random.default_rng(0)
    blocks = (2, 5, 10, 20)
    pooled = {b: [] for b in blocks}
    for _ in range(120):
        e = rng.standard_normal(n)
        for b in blocks:
            pooled[b].append(_totals(e, b, b, n_sim=300, seed=1))
    widths = [float(np.std(np.concatenate(pooled[b])) / np.sqrt(b)) for b in blocks]
    assert max(widths) / min(widths) < 1.10
    assert all(0.9 < w < 1.1 for w in widths)


@pytest.mark.parametrize("b", [1, 2, 7, 23])
def test_shifting_the_series_still_changes_nothing(b):
    """The correction scales; it does not re-introduce the level."""
    e = np.random.default_rng(11).standard_normal(47)
    kw = dict(horizon=13, block=b, n_sim=64)
    base = block_error_paths(e, rng=np.random.default_rng(1), **kw)
    moved = block_error_paths(e + 25.0, rng=np.random.default_rng(1), **kw)
    assert np.allclose(moved, base, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("b", [1, 3, 13])
def test_rescaling_the_series_still_rescales_the_paths(b):
    e = np.random.default_rng(13).standard_normal(47)
    kw = dict(horizon=13, block=b, n_sim=64)
    base = block_error_paths(e, rng=np.random.default_rng(2), **kw)
    scaled = block_error_paths(e * 3.0, rng=np.random.default_rng(2), **kw)
    assert np.allclose(scaled, base * 3.0, rtol=1e-9, atol=1e-8)


def test_the_band_is_wider_than_it_was_before_the_correction():
    """Stated as a direction, since that is the user-visible consequence: bands
    that were narrow by sqrt((n-b)/(n-1)) are now wider by its inverse."""
    e = np.random.default_rng(17).standard_normal(47)
    corrected = np.std(_totals(e, 13, 13, n_sim=8000, seed=4))
    uncorrected = corrected * np.sqrt((47 - 13) / (47 - 1))
    assert corrected > uncorrected
    assert corrected / uncorrected == pytest.approx(np.sqrt(46 / 34), rel=1e-9)


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_the_longest_admissible_block_is_scaled_the_hardest():
    """At b = n / 2 the factor is sqrt((n-1)/(n/2)), the largest the guard allows,
    and the draw still returns finite paths."""
    e = np.random.default_rng(19).standard_normal(40)
    paths = block_error_paths(e, horizon=20, block=20, n_sim=128,
                              rng=np.random.default_rng(0))
    assert np.isfinite(paths).all()
    assert np.std(paths.sum(axis=1)) > 0.0


def test_a_two_period_series_still_draws():
    e = np.array([1.0, -3.0])
    paths = block_error_paths(e, horizon=4, block=1, n_sim=32,
                              rng=np.random.default_rng(0))
    assert np.isfinite(paths).all()


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #
def test_a_block_past_half_the_series_is_still_refused():
    """The guard and the correction describe the same curve; correcting the
    shrinkage does not license drawing past the point where it inverts."""
    e = np.random.default_rng(0).standard_normal(20)
    with pytest.raises(MlsynthDataError, match="half"):
        block_error_paths(e, horizon=11, block=11, n_sim=16,
                          rng=np.random.default_rng(0))


def test_a_single_calibration_error_is_still_refused():
    with pytest.raises(MlsynthDataError, match="single calibration error"):
        block_error_paths(np.array([2.0]), horizon=3, block=1, n_sim=16,
                          rng=np.random.default_rng(0))
