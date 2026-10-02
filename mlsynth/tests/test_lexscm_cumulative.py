"""Tests for the cumulative-effect path on a simplex-weighted synthetic control.

TBR's cumulative posterior is closed form because AdID's counterfactual is affine
in the treated series, so summing the gaps keeps the Student-t pivot. Simplex
weights break that affinity, so the cumulative interval here is built by
inverting a block-sum pivot against held-out placebo residuals, and its
properties have to be asserted, not inherited.

The design behind these assertions, with the measured coverage, is in
``agents`` and the pull request; the numbers that matter:

* pooling placebo series and rescaling each to the treated unit's residual
  scale covers 0.915 at nominal 0.90, flat in horizon. Pooling without
  rescaling covers 0.93--0.95 because the pool is noisier than the unit under
  test; using the unit's own residuals alone covers 0.83 and falls to 0.72 by
  horizon 8, because a blank window yields too few distinct blocks.
* the interval is only centred when the donors can approximate the treated
  unit. Outside their convex hull coverage is 0.037 with intervals six times
  wider, so ``approximability`` is a precondition and not a diagnostic.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.fast_scm_helpers.post_inference import (
    approximability,
    cumulative_path,
)


@pytest.fixture
def pool() -> list[np.ndarray]:
    """Eleven held-out residual series, generous enough to form blocks from."""
    rng = np.random.default_rng(0)
    return [rng.normal(0.0, 1.0, 32) for _ in range(11)]


@pytest.fixture
def gaps() -> np.ndarray:
    """Eight post-period gaps carrying a constant effect of 2."""
    rng = np.random.default_rng(1)
    return 2.0 + rng.normal(0.0, 1.0, 8)


# --------------------------------------------------------------------- smoke

def test_path_has_one_row_per_horizon(gaps, pool):
    path = cumulative_path(gaps, pool)
    assert [p.horizon for p in path] == list(range(1, gaps.size + 1))
    assert all(np.isfinite([p.estimate, p.lower, p.upper]).all() for p in path)


# ----------------------------------------------------------------- invariants

def test_point_estimate_is_the_running_sum_of_gaps(gaps, pool):
    """The estimate is arithmetic; only the interval is inferential."""
    path = cumulative_path(gaps, pool)
    assert np.allclose([p.estimate for p in path], np.cumsum(gaps))


def test_interval_brackets_its_own_point_estimate(gaps, pool):
    for p in cumulative_path(gaps, pool):
        assert p.lower <= p.estimate <= p.upper


def test_higher_confidence_gives_a_wider_interval(gaps, pool):
    narrow = cumulative_path(gaps, pool, level=0.80)
    wide = cumulative_path(gaps, pool, level=0.99)
    for a, b in zip(narrow, wide):
        assert b.upper - b.lower >= a.upper - a.lower


def test_shifting_every_gap_shifts_the_interval_by_the_same_amount(gaps, pool):
    """The pivot is a location statistic, so a constant shift must translate it."""
    base = cumulative_path(gaps, pool)
    moved = cumulative_path(gaps + 5.0, pool)
    for h, (a, b) in enumerate(zip(base, moved), start=1):
        assert b.lower == pytest.approx(a.lower + 5.0 * h)
        assert b.upper == pytest.approx(a.upper + 5.0 * h)


def test_rescaling_shrinks_the_influence_of_a_noisier_pool(gaps):
    """Standardising is what stops a noisy pool inflating the interval."""
    rng = np.random.default_rng(2)
    noisy = [rng.normal(0.0, 1.0, 32)] + [rng.normal(0.0, 4.0, 32) for _ in range(8)]
    on = cumulative_path(gaps, noisy, standardize=True)
    off = cumulative_path(gaps, noisy, standardize=False)
    assert all(a.upper - a.lower < b.upper - b.lower for a, b in zip(on, off))


def test_a_single_post_period_still_returns_a_path(pool):
    path = cumulative_path(np.array([1.5]), pool)
    assert len(path) == 1 and path[0].estimate == pytest.approx(1.5)


# ---------------------------------------------------------------- edge cases

def test_series_shorter_than_the_horizon_are_skipped_not_fatal(gaps):
    """A short placebo cannot form a long block; the rest of the pool carries it."""
    rng = np.random.default_rng(3)
    ragged = [rng.normal(0, 1, 32), rng.normal(0, 1, 3), rng.normal(0, 1, 32)]
    path = cumulative_path(gaps, ragged)
    assert all(np.isfinite([p.lower, p.upper]).all() for p in path)


def test_pool_too_short_for_any_horizon_is_refused(gaps):
    """Short but non-degenerate, so this is the block length and not the spread."""
    rng = np.random.default_rng(6)
    short = [rng.normal(0.0, 1.0, 2), rng.normal(0.0, 1.0, 2)]
    with pytest.raises(MlsynthDataError, match="block"):
        cumulative_path(gaps, short)


def test_constant_pool_has_no_spread_and_is_refused(gaps):
    with pytest.raises(MlsynthDataError, match="degenerate|spread"):
        cumulative_path(gaps, [np.zeros(32), np.zeros(32)])


# ------------------------------------------------------------------ failures

def test_empty_pool_is_refused(gaps):
    with pytest.raises(MlsynthDataError, match="placebo"):
        cumulative_path(gaps, [])


def test_no_post_gaps_is_refused(pool):
    with pytest.raises(MlsynthDataError, match="post"):
        cumulative_path(np.array([]), pool)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_level_outside_the_unit_interval_is_refused(gaps, pool, bad):
    with pytest.raises(MlsynthConfigError, match="level"):
        cumulative_path(gaps, pool, level=bad)


def test_non_finite_gaps_are_refused(pool):
    with pytest.raises(MlsynthDataError, match="finite"):
        cumulative_path(np.array([1.0, np.nan, 3.0]), pool)


# ------------------------------------------------------------ approximability

def test_a_centred_blank_window_passes(gaps):
    rng = np.random.default_rng(4)
    check = approximability(rng.normal(0.0, 1.0, 32))
    assert check.ok and abs(check.t_stat) < 3.0


def test_an_offset_blank_window_fails_and_says_so(gaps):
    """Out of the donors' hull the gap is not centred, and the interval is biased."""
    rng = np.random.default_rng(5)
    check = approximability(12.0 + rng.normal(0.0, 1.0, 32))
    assert not check.ok
    assert abs(check.t_stat) > 3.0 and check.p_value < 0.01


def test_approximability_needs_more_than_one_period():
    with pytest.raises(MlsynthDataError, match="period"):
        approximability(np.array([1.0]))


def test_non_finite_blank_window_is_refused():
    with pytest.raises(MlsynthDataError, match="finite"):
        approximability(np.array([1.0, np.inf, 2.0, 3.0]))


def test_flat_blank_window_cannot_be_tested_for_an_offset():
    with pytest.raises(MlsynthDataError, match="spread"):
        approximability(np.full(16, 4.0))


def test_non_finite_pool_is_refused(gaps):
    rng = np.random.default_rng(7)
    bad = [rng.normal(0.0, 1.0, 32), np.concatenate([rng.normal(0, 1, 31), [np.nan]])]
    with pytest.raises(MlsynthDataError, match="finite"):
        cumulative_path(gaps, bad)
