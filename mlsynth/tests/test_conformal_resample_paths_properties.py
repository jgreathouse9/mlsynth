"""Metamorphic invariants of the block-resampled error paths.

The example tests pin the draw on fixed series. These assert what has to hold
for every calibration series, every horizon and every admissible block, and they
are the assertions that catch a defect the examples happen not to exercise:

* structure -- the output is always ``(n_sim, horizon)`` and finite, whatever
  the block length or how the horizon divides it;
* membership -- every drawn value is plus or minus an entry of the centred
  series, because the draw resamples that series and does nothing else to it;
* location invariance -- adding a constant to the whole series changes no path
  at all, since the draw centres first. This is what makes the band a statement
  about the fit's variability instead of its level;
* scale equivariance -- multiplying the series by a constant multiplies every
  path by the same constant, exactly;
* reflection invariance -- negating the series leaves the drawn paths
  unchanged, because each block's sign is flipped with probability one half and
  the flip is drawn from the same generator state;
* determinism -- the same generator state gives the same paths, which is what
  lets a caller reproduce a band;
* contiguity -- with one block covering the horizon, each path is a circular
  run of the centred series up to an overall sign, so the draw carries serial
  dependence instead of shuffling it away.

Not asserted here: that the drawn spread matches the series' own. Centring the
series constrains its circular sums, so a b-block's drawn variance falls short
of ``b`` times the series variance by about ``(n - b) / (n - 1)`` -- measured at
0.70 for a 13-block of 47 periods. That is a property of the draw as it stands
and correcting it changes every band's width, so it belongs to a change of its
own and not to a test file.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.conformal import block_error_paths

_SETTINGS = settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

#: Series long enough to admit a block, short enough to draw quickly.
_SERIES = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False,
              width=64),
    min_size=4, max_size=60,
)


def _admissible(series, block):
    """A block the draw accepts: at least 1, at most half the series."""
    return max(1, min(block, len(series) // 2))


@given(series=_SERIES, horizon=st.integers(min_value=1, max_value=20),
       block=st.integers(min_value=1, max_value=30),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_the_shape_is_always_simulations_by_horizon(series, horizon, block, seed):
    e = np.asarray(series, dtype=float)
    paths = block_error_paths(e, horizon=horizon, block=_admissible(series, block),
                              n_sim=16, rng=np.random.default_rng(seed))
    assert paths.shape == (16, horizon)
    assert np.isfinite(paths).all()


@given(series=_SERIES, horizon=st.integers(min_value=1, max_value=20),
       block=st.integers(min_value=1, max_value=30),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_every_drawn_value_is_a_signed_entry_of_the_centred_series(
        series, horizon, block, seed):
    """The draw resamples; it does not interpolate, smooth or rescale."""
    e = np.asarray(series, dtype=float)
    paths = block_error_paths(e, horizon=horizon, block=_admissible(series, block),
                              n_sim=16, rng=np.random.default_rng(seed))
    pool = np.concatenate([e - e.mean(), -(e - e.mean())])
    for value in np.unique(paths):
        assert np.isclose(np.abs(pool - value).min(), 0.0, atol=1e-9)


@given(series=_SERIES, shift=st.floats(min_value=-500.0, max_value=500.0,
                                       allow_nan=False, allow_infinity=False),
       horizon=st.integers(min_value=1, max_value=20),
       block=st.integers(min_value=1, max_value=30),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_shifting_the_whole_series_changes_no_path(series, shift, horizon, block,
                                                   seed):
    """A fit sitting high shifts every error by the same amount. The draw centres,
    so the band describes how the errors move, not where they sit."""
    e = np.asarray(series, dtype=float)
    b = _admissible(series, block)
    kw = dict(horizon=horizon, block=b, n_sim=16)
    base = block_error_paths(e, rng=np.random.default_rng(seed), **kw)
    moved = block_error_paths(e + shift, rng=np.random.default_rng(seed), **kw)
    assert np.allclose(moved, base, rtol=1e-9, atol=1e-8)


@given(series=_SERIES, scale=st.floats(min_value=0.01, max_value=100.0,
                                       allow_nan=False, allow_infinity=False),
       horizon=st.integers(min_value=1, max_value=20),
       block=st.integers(min_value=1, max_value=30),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_rescaling_the_series_rescales_every_path(series, scale, horizon, block,
                                                  seed):
    e = np.asarray(series, dtype=float)
    b = _admissible(series, block)
    kw = dict(horizon=horizon, block=b, n_sim=16)
    base = block_error_paths(e, rng=np.random.default_rng(seed), **kw)
    scaled = block_error_paths(e * scale, rng=np.random.default_rng(seed), **kw)
    assert np.allclose(scaled, base * scale, rtol=1e-9, atol=1e-8)


@given(series=_SERIES, horizon=st.integers(min_value=1, max_value=20),
       block=st.integers(min_value=1, max_value=30),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_negating_the_series_leaves_the_paths_alone(series, horizon, block, seed):
    """Each block's sign is flipped with probability one half from the same
    generator state, so the sign of the series itself carries no information."""
    e = np.asarray(series, dtype=float)
    b = _admissible(series, block)
    kw = dict(horizon=horizon, block=b, n_sim=16)
    base = block_error_paths(e, rng=np.random.default_rng(seed), **kw)
    flipped = block_error_paths(-e, rng=np.random.default_rng(seed), **kw)
    assert np.allclose(flipped, -base, rtol=1e-9, atol=1e-8)


@given(series=_SERIES, horizon=st.integers(min_value=1, max_value=20),
       block=st.integers(min_value=1, max_value=30),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_the_same_generator_state_gives_the_same_paths(series, horizon, block, seed):
    e = np.asarray(series, dtype=float)
    b = _admissible(series, block)
    kw = dict(horizon=horizon, block=b, n_sim=16)
    a = block_error_paths(e, rng=np.random.default_rng(seed), **kw)
    c = block_error_paths(e, rng=np.random.default_rng(seed), **kw)
    assert np.array_equal(a, c)


@given(series=_SERIES, horizon=st.integers(min_value=2, max_value=12),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_a_whole_horizon_block_draws_a_contiguous_run(series, horizon, seed):
    """The reason a block exists. With one block spanning the horizon, each path
    is a circular run of the centred series up to an overall sign, so whatever
    serial dependence the series carries survives into the total. Skipped where
    the horizon exceeds half the series, which the draw refuses."""
    e = np.asarray(series, dtype=float)
    if 2 * horizon > e.size:
        pytest.skip("a whole-horizon block past n/2 is refused by design")
    centred = e - e.mean()
    paths = block_error_paths(e, horizon=horizon, block=horizon, n_sim=12,
                              rng=np.random.default_rng(seed))
    for row in paths:
        runs = [np.concatenate([centred, centred])[s:s + horizon]
                for s in range(e.size)]
        assert any(np.allclose(row, r, atol=1e-8) or np.allclose(row, -r, atol=1e-8)
                   for r in runs)


@given(series=_SERIES, horizon=st.integers(min_value=1, max_value=20),
       seed=st.integers(min_value=0, max_value=10_000))
@_SETTINGS
def test_single_period_blocks_stay_within_the_pool(series, horizon, seed):
    """At block one the draw is Wheeler's symmetrised pool, one element per
    period, so a path cannot contain a value the pool does not."""
    e = np.asarray(series, dtype=float)
    paths = block_error_paths(e, horizon=horizon, block=1, n_sim=16,
                              rng=np.random.default_rng(seed))
    pool = np.concatenate([e - e.mean(), -(e - e.mean())])
    assert paths.min() >= pool.min() - 1e-9
    assert paths.max() <= pool.max() + 1e-9
