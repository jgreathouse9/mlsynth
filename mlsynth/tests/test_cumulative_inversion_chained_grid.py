"""The grid search chains its refits, and the band it returns does not move.

``cumulative_conformal_by_inversion(search="grid")`` evaluates the conformal
p-value once per candidate effect. Each evaluation solves the same simplex
program on a design that barely moves between neighbours -- only the post block
shifts, by one grid step -- and solved cold, every candidate rebuilds its active
set from nothing. ``conformal_pvalue_sweep`` exists to chain them, and this wires
the grid route to it.

There are exactly two things to assert, and the order matters. The band must be
identical, because a seed is a starting point for the solver and never a
parameter of the estimand: if the interval moves, the change is not an
optimisation. And the work must actually fall, because a wiring that silently
reverts to cold solves leaves every number right and only the saving gone --
which no assertion on the band can see.

Levels: smoke, unit invariants, edge.
"""
import numpy as np
import pytest

from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
from mlsynth.utils.conformal import cumulative_conformal_by_inversion
from mlsynth.utils.conformal.inversion import confidence_set_bounds

_KW = dict(refit="sc", conformal_type="block", alpha=0.1)


def _panel(J=20, pre=40, horizon=8, effect=0.3, seed=0):
    rng = np.random.default_rng(seed)
    T = pre + horizon
    f = np.column_stack([np.cumsum(rng.normal(0, 1, T)) for _ in range(2)])
    lam = rng.dirichlet(np.ones(J - 1), size=2).T * 2.0
    Y0 = f @ lam.T + 0.5 * rng.standard_normal((T, J - 1))
    y = Y0 @ rng.dirichlet(np.ones(J - 1)) + 0.5 * rng.standard_normal(T)
    y[pre:] += effect
    return y, Y0, pre, horizon


def _oracle_bounds(y, Y0, pre, horizon, grid, alpha):
    """The interval as one cold call per candidate would give it."""
    pvalues = []
    for tau in grid:
        shifted = y[:pre + horizon].copy()
        shifted[pre:] -= tau
        pvalues.append(conformal_pvalue(shifted, Y0[:pre + horizon], pre,
                                        refit="sc", conformal_type="block"))
    return confidence_set_bounds(np.asarray(grid), np.asarray(pvalues), alpha)


def _pivots_of(call):
    """Active-set pivots spent inside ``call``.

    ``simplex_qp`` imports the solver at call time, so replacing the module
    attribute is seen by the code under test; the replacement forwards to the
    real solver and only records its work.
    """
    from mlsynth.utils.bilevel import active_set

    real = active_set.solve_simplex_qp
    spent = []

    def counting(B, A, **kw):
        want_info = kw.pop("return_info", False)
        w, info = real(B, A, return_info=True, **kw)
        spent.append(info["pivots"])
        return (w, info) if want_info else w

    active_set.solve_simplex_qp = counting
    try:
        call()
    finally:
        active_set.solve_simplex_qp = real
    return sum(spent)


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def test_the_grid_route_still_returns_a_band():
    y, Y0, pre, h = _panel()
    band = cumulative_conformal_by_inversion(
        y, Y0, pre, h, search="grid", grid=np.linspace(-1.0, 1.5, 21), **_KW)
    assert band.search == "grid"
    assert band.has_interior_gaps in (True, False)


# --------------------------------------------------------------------------- #
# unit invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chaining_does_not_move_the_band(seed):
    """The claim the wiring lives or dies by, asserted exactly. A conformal
    p-value is a rank among a finite reference set, so any discrepancy is a
    different rank and a different interval."""
    y, Y0, pre, h = _panel(seed=seed)
    grid = np.linspace(-1.5, 2.0, 25)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                             grid=grid, **_KW)
    lo, hi = _oracle_bounds(y, Y0, pre, h, grid, 0.1)
    assert band.per_period_lower == lo
    assert band.per_period_upper == hi


def test_the_grid_route_spends_far_fewer_pivots():
    """The work assertion. Without it, a wiring that reverts to cold solves
    leaves every number right and only the saving gone."""
    y, Y0, pre, h = _panel(J=30, seed=4)
    grid = np.linspace(-1.5, 2.0, 41)
    chained = _pivots_of(lambda: cumulative_conformal_by_inversion(
        y, Y0, pre, h, search="grid", grid=grid, **_KW))
    cold = _pivots_of(lambda: _oracle_bounds(y, Y0, pre, h, grid, 0.1))
    assert chained < cold / 3, f"chained {chained} pivots against cold {cold}"


def test_a_shuffled_grid_costs_no_more_than_a_sorted_one():
    """The saving comes from consecutive solves being neighbours, which the
    sweep arranges by sorting internally."""
    y, Y0, pre, h = _panel(J=30, seed=4)
    grid = np.linspace(-1.5, 2.0, 41)
    shuffled = grid.copy()
    np.random.default_rng(0).shuffle(shuffled)
    kw = dict(search="grid", **_KW)
    a = _pivots_of(lambda: cumulative_conformal_by_inversion(
        y, Y0, pre, h, grid=grid, **kw))
    b = _pivots_of(lambda: cumulative_conformal_by_inversion(
        y, Y0, pre, h, grid=shuffled, **kw))
    assert b <= a * 1.25


def test_the_reported_gaps_are_unchanged_by_chaining():
    """``has_interior_gaps`` is read off the same p-values, so it cannot depend
    on how they were computed."""
    y, Y0, pre, h = _panel(seed=5)
    grid = np.linspace(-2.0, 2.0, 33)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                             grid=grid, **_KW)
    pvalues = []
    for tau in grid:
        shifted = y[:pre + h].copy()
        shifted[pre:] -= tau
        pvalues.append(conformal_pvalue(shifted, Y0[:pre + h], pre, refit="sc",
                                        conformal_type="block"))
    kept = np.asarray(pvalues) > 0.1
    if kept.any():
        first = int(np.argmax(kept))
        last = int(kept.size - 1 - np.argmax(kept[::-1]))
        assert band.has_interior_gaps == bool((~kept[first:last + 1]).any())


def test_the_bisecting_route_is_untouched():
    """Only the grid branch is wired; bisection walks outward one candidate at a
    time and has no sweep to chain."""
    y, Y0, pre, h = _panel(seed=2)
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, **_KW)
    assert band.search == "bisect"
    assert band.has_interior_gaps is None


# --------------------------------------------------------------------------- #
# edge
# --------------------------------------------------------------------------- #
def test_a_two_candidate_grid_still_agrees_with_cold_calls():
    y, Y0, pre, h = _panel(seed=3)
    grid = np.array([-0.5, 1.0])
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                             grid=grid, **_KW)
    lo, hi = _oracle_bounds(y, Y0, pre, h, grid, 0.1)
    assert (band.per_period_lower, band.per_period_upper) == (lo, hi)


def test_a_grid_no_candidate_survives_is_still_empty():
    """An empty accepted set is a real answer, and chaining must not turn it
    into a spurious interval."""
    y, Y0, pre, h = _panel(seed=6)
    grid = np.linspace(500.0, 600.0, 9)      # nowhere near any plausible effect
    band = cumulative_conformal_by_inversion(y, Y0, pre, h, search="grid",
                                             grid=grid, **_KW)
    assert band.is_empty
    assert np.isnan(band.lower) and np.isnan(band.upper)
