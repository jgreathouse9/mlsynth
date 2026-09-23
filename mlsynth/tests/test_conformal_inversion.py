"""Inverting a conformal test: which nulls the confidence set keeps.

A test-inversion confidence set is defined by duality. Sweep a grid of candidate
effects, test each one at level ``alpha``, and keep the ones the test does not
reject; the interval is the range of what survives. So the set is
``{tau : p(tau) > alpha}``, and the boundary matters more here than it does for
a continuous test statistic.

The reason is that a permutation p-value is discrete. The conformal reference
set has a finite number of members -- ``T0 + 1`` residuals for a single-period
inversion, ``T`` cyclic blocks for the moving-block scheme -- so ``p`` can only
take the values ``k / n``. When ``alpha`` is itself one of those values, and
``alpha = 0.1`` with ``n = 20`` is, the difference between ``>`` and ``>=`` is
not a tie-breaking nicety: it is a whole level of the reference set, kept or
dropped, on a set of candidates with positive measure. Keeping ``p == alpha``
keeps nulls the test rejects, and the interval that results is not the inversion
of the test reported alongside it.

The authors' own implementation takes the strict rule --
``ci_grid[ps_temp > alpha]`` in ``scinference::confidence_interval`` (v1.0.0,
``567c688``) -- and on the JASA gonorrhea panel the inclusive rule widens every
one of the six per-period intervals by 15 to 40 percent.

These are the unit and property tests for the rule itself; the end-to-end
agreement with the authors' package is pinned in
``test_vanillasc_conformal_cwz.py`` and in the ``cwz_conformal`` benchmark case.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.conformal import confidence_set_bounds

_SETTINGS = settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def test_smoke_returns_the_range_of_surviving_nulls():
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p = np.array([0.02, 0.30, 0.90, 0.30, 0.02])
    lo, hi = confidence_set_bounds(grid, p, alpha=0.1)
    assert (lo, hi) == (-1.0, 1.0)
    assert np.isfinite([lo, hi]).all()


# --------------------------------------------------------------------------
# unit -- the duality rule, at the boundary
# --------------------------------------------------------------------------

def test_a_null_whose_p_value_equals_alpha_is_rejected():
    """``p == alpha`` is a rejection, so the candidate is outside the set.

    This is the whole defect. With ``T0 + 1 = 20`` residuals every attainable
    p-value is a multiple of ``0.05``, so ``alpha = 0.1`` is attained exactly
    and the inclusive rule keeps a null the test rejected.
    """
    grid = np.array([-1.0, 0.0, 1.0])
    p = np.array([0.10, 0.50, 0.10])
    lo, hi = confidence_set_bounds(grid, p, alpha=0.1)
    assert (lo, hi) == (0.0, 0.0)


def test_the_set_is_exactly_the_grid_points_the_test_does_not_reject():
    rng = np.random.default_rng(0)
    grid = np.linspace(-3.0, 3.0, 61)
    p = rng.uniform(0.0, 1.0, size=grid.size)
    lo, hi = confidence_set_bounds(grid, p, alpha=0.25)
    kept = grid[p > 0.25]
    assert (lo, hi) == (float(kept.min()), float(kept.max()))


def test_bounds_are_the_hull_of_the_kept_set_not_its_interior():
    """A gap inside the accepted region does not split the interval.

    The reported object is an interval, so a non-convex accepted set is
    summarised by its range. Recording the convention here because the
    alternative -- returning the largest contiguous run -- is a plausible
    reading of the same code.
    """
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.9, 0.01, 0.01, 0.9])
    assert confidence_set_bounds(grid, p, alpha=0.1) == (0.0, 3.0)


# --------------------------------------------------------------------------
# edge
# --------------------------------------------------------------------------

def test_an_empty_confidence_set_is_reported_as_nan():
    grid = np.array([-1.0, 0.0, 1.0])
    lo, hi = confidence_set_bounds(grid, np.array([0.01, 0.02, 0.01]), alpha=0.1)
    assert np.isnan(lo) and np.isnan(hi)


def test_alpha_below_the_achievable_level_rejects_nothing():
    """Below ``1 / n`` no null can be rejected, so the set is the whole grid.

    A permutation p-value is bounded below by ``1 / n`` because the observed
    statistic is a member of its own reference set. Asking for a level finer
    than the reference set can express is answerable only by keeping
    everything, which is what an honest inversion returns.
    """
    grid = np.linspace(-5.0, 5.0, 21)
    p = np.full(grid.size, 1.0 / 20.0)          # the floor of a 20-member set
    lo, hi = confidence_set_bounds(grid, p, alpha=0.01)
    assert (lo, hi) == (-5.0, 5.0)


def test_single_grid_point():
    assert confidence_set_bounds(np.array([2.5]), np.array([0.4]), alpha=0.1) == (2.5, 2.5)
    lo, hi = confidence_set_bounds(np.array([2.5]), np.array([0.04]), alpha=0.1)
    assert np.isnan(lo) and np.isnan(hi)


def test_an_unsorted_grid_is_summarised_by_its_range_not_its_order():
    grid = np.array([1.0, -1.0, 0.0])
    p = np.array([0.9, 0.9, 0.01])
    assert confidence_set_bounds(grid, p, alpha=0.1) == (-1.0, 1.0)


# --------------------------------------------------------------------------
# failure -- reported, not swallowed
# --------------------------------------------------------------------------

def test_mismatched_lengths_raise_a_translated_error():
    with pytest.raises(MlsynthDataError, match="same length"):
        confidence_set_bounds(np.array([0.0, 1.0]), np.array([0.5]), alpha=0.1)


def test_an_empty_grid_raises_a_translated_error():
    with pytest.raises(MlsynthDataError, match="at least one candidate"):
        confidence_set_bounds(np.array([]), np.array([]), alpha=0.1)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, np.nan])
def test_alpha_outside_the_unit_interval_raises(bad):
    with pytest.raises(MlsynthDataError, match="alpha"):
        confidence_set_bounds(np.array([0.0]), np.array([0.5]), alpha=bad)


def test_a_non_finite_p_value_raises_rather_than_being_compared():
    """``nan > alpha`` is ``False``, so a NaN would silently drop a candidate."""
    with pytest.raises(MlsynthDataError, match="finite"):
        confidence_set_bounds(np.array([0.0, 1.0]), np.array([0.5, np.nan]), alpha=0.1)


# --------------------------------------------------------------------------
# properties
# --------------------------------------------------------------------------

_pvals = st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=40)


@given(p=_pvals, alpha=st.floats(0.01, 0.5))
@_SETTINGS
def test_every_reported_bound_is_a_null_the_test_does_not_reject(p, alpha):
    """The duality invariant, stated on the output.

    Whatever the p-value path, an endpoint the function reports must itself be
    a surviving candidate. This is what ties the reported band to the reported
    p-values, and it is the assertion whose absence let the two disagree.
    """
    p = np.asarray(p, dtype=float)
    grid = np.arange(p.size, dtype=float)
    lo, hi = confidence_set_bounds(grid, p, alpha=alpha)
    if np.isnan(lo):
        assert not (p > alpha).any()
        return
    for bound in (lo, hi):
        assert p[grid == bound][0] > alpha


@given(p=_pvals, alpha=st.floats(0.01, 0.5))
@_SETTINGS
def test_the_set_shrinks_as_the_level_tightens(p, alpha):
    """Monotone in ``alpha``: a stricter level cannot admit more nulls."""
    p = np.asarray(p, dtype=float)
    grid = np.arange(p.size, dtype=float)
    lo_a, hi_a = confidence_set_bounds(grid, p, alpha=alpha)
    lo_b, hi_b = confidence_set_bounds(grid, p, alpha=min(alpha * 2.0, 0.99))
    if np.isnan(lo_b):
        return                                   # tighter set empty: nothing to nest
    assert not np.isnan(lo_a)
    assert lo_a <= lo_b and hi_a >= hi_b


@given(p=_pvals, alpha=st.floats(0.01, 0.5), shift=st.floats(-100.0, 100.0))
@_SETTINGS
def test_the_bounds_translate_with_the_grid(p, alpha, shift):
    """Shifting every candidate shifts the interval and nothing else."""
    p = np.asarray(p, dtype=float)
    grid = np.arange(p.size, dtype=float)
    lo, hi = confidence_set_bounds(grid, p, alpha=alpha)
    lo_s, hi_s = confidence_set_bounds(grid + shift, p, alpha=alpha)
    if np.isnan(lo):
        assert np.isnan(lo_s)
        return
    assert lo_s == pytest.approx(lo + shift)
    assert hi_s == pytest.approx(hi + shift)
