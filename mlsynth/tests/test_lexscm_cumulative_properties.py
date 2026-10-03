r"""Generative properties for the cumulative path on a constrained synthetic control.

The example tests fix one panel. These assert what the construction means for
every panel it accepts. The interval inverts a pivot on the sum over blocks of
held-out residuals, so it is equivariant in location and in scale, it is blind
to the order the placebo series arrive in, and it never narrows as the level
rises. The aggregate is the weighted mean of the per-unit paths by Abadie and
Zhao's equation (11), and that has to hold for every point on the simplex, not
only the one the fixtures use.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.fast_scm_helpers.post_inference import (
    approximability,
    cumulative_path,
    population_cumulative,
    unit_level_cumulative,
)

SETTINGS = settings(max_examples=40, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def gaps_and_pool(draw):
    """Post-period gaps and a pool of held-out series long enough to block."""
    n_post = draw(st.integers(1, 8))
    n_blank = draw(st.integers(12, 40))
    n_series = draw(st.integers(2, 8))
    rng = np.random.default_rng(draw(st.integers(0, 2 ** 31 - 1)))
    scale = draw(st.floats(0.2, 5.0))
    post = draw(st.floats(-5.0, 5.0)) + rng.normal(0.0, scale, n_post)
    pool = [rng.normal(0.0, scale * draw(st.floats(0.5, 2.0)), n_blank)
            for _ in range(n_series)]
    return post, pool


@st.composite
def unit_panels(draw):
    """Per-unit gaps with weights drawn from the simplex."""
    n_units = draw(st.integers(1, 4))
    n_post = draw(st.integers(1, 8))
    n_blank = draw(st.integers(12, 40))
    rng = np.random.default_rng(draw(st.integers(0, 2 ** 31 - 1)))
    post = rng.normal(0.0, draw(st.floats(0.3, 3.0)), (n_post, n_units))
    blank = rng.normal(0.0, draw(st.floats(0.3, 3.0)), (n_blank, n_units))
    raw = np.array([draw(st.floats(0.05, 1.0)) for _ in range(n_units)])
    return post, blank, raw / raw.sum()


@given(gaps_and_pool(), st.floats(-20.0, 20.0))
@SETTINGS
def test_a_shift_in_the_gaps_moves_the_path_by_the_same_amount(case, shift):
    """Location equivariance. At horizon h a constant shift moves the cumulative
    by h times it, and the pivot is built from the pool, which did not move."""
    post, pool = case
    base = cumulative_path(post, pool)
    moved = cumulative_path(post + shift, pool)
    for h, (a, b) in enumerate(zip(base, moved), start=1):
        assert b.estimate == pytest.approx(a.estimate + shift * h, abs=1e-6, rel=1e-9)
        assert b.lower == pytest.approx(a.lower + shift * h, abs=1e-6, rel=1e-9)
        assert b.upper == pytest.approx(a.upper + shift * h, abs=1e-6, rel=1e-9)


@given(gaps_and_pool(), st.floats(0.1, 10.0))
@SETTINGS
def test_rescaling_the_outcome_rescales_the_path(case, factor):
    """Scale equivariance. The interval carries the outcome's units, so changing
    them changes nothing about which effects are covered."""
    post, pool = case
    base = cumulative_path(post, pool)
    scaled = cumulative_path(post * factor, [s * factor for s in pool])
    for a, b in zip(base, scaled):
        assert b.estimate == pytest.approx(a.estimate * factor, rel=1e-9, abs=1e-9)
        assert b.lower == pytest.approx(a.lower * factor, rel=1e-9, abs=1e-9)
        assert b.upper == pytest.approx(a.upper * factor, rel=1e-9, abs=1e-9)


@given(gaps_and_pool(), st.integers(0, 2 ** 31 - 1))
@SETTINGS
def test_the_order_of_the_placebo_series_does_not_matter(case, seed):
    """The pool is a set. Only the first series is distinguished, because it
    sets the scale the rest are standardised onto, so the tail may permute."""
    post, pool = case
    if len(pool) < 3:
        return
    tail = pool[1:]
    order = np.random.default_rng(seed).permutation(len(tail))
    shuffled = [pool[0]] + [tail[i] for i in order]
    for a, b in zip(cumulative_path(post, pool), cumulative_path(post, shuffled)):
        assert b.lower == pytest.approx(a.lower, rel=1e-9, abs=1e-9)
        assert b.upper == pytest.approx(a.upper, rel=1e-9, abs=1e-9)


@given(gaps_and_pool())
@SETTINGS
def test_a_higher_level_is_never_a_narrower_interval(case):
    """Monotonicity in the level, which quantiles of a fixed sample guarantee."""
    post, pool = case
    paths = [cumulative_path(post, pool, level=lv) for lv in (0.50, 0.80, 0.95, 0.99)]
    for h in range(len(post)):
        widths = [p[h].upper - p[h].lower for p in paths]
        assert all(b >= a - 1e-9 for a, b in zip(widths, widths[1:]))


@given(unit_panels())
@SETTINGS
def test_the_aggregate_is_the_weighted_mean_of_its_parts(panel):
    """Equation (11), over the simplex and not only at one point on it."""
    post, blank, w = panel
    out = unit_level_cumulative(post, blank, w)
    for h in range(post.shape[0]):
        parts = np.array([out.per_unit[k][h].estimate for k in range(w.size)])
        assert out.aggregate[h].estimate == pytest.approx(float(w @ parts),
                                                          rel=1e-9, abs=1e-9)


@given(unit_panels(), st.floats(-10.0, 10.0))
@SETTINGS
def test_centring_is_blind_to_a_per_unit_level(panel, shift):
    """A unit's fitted level appears in both windows, so centring removes it and
    the result cannot depend on it. This is what centring is for."""
    post, blank, w = panel
    offsets = shift * np.arange(1, w.size + 1)
    base = unit_level_cumulative(post, blank, w, center=True)
    moved = unit_level_cumulative(post + offsets, blank + offsets, w, center=True)
    for a, b in zip(base.aggregate, moved.aggregate):
        assert b.estimate == pytest.approx(a.estimate, abs=1e-6, rel=1e-9)
        assert b.lower == pytest.approx(a.lower, abs=1e-6, rel=1e-9)
        assert b.upper == pytest.approx(a.upper, abs=1e-6, rel=1e-9)


@given(unit_panels())
@SETTINGS
def test_the_population_interval_never_narrows_the_treated_one(panel):
    """The representation error is mean zero, so it widens and never moves."""
    post, blank, w = panel
    f = np.full(10, 0.1)
    idx = np.arange(w.size)
    pop = population_cumulative(post, blank, w, f, idx)
    treated = unit_level_cumulative(post, blank, w)
    for a, b in zip(treated.aggregate, pop.aggregate):
        assert b.estimate == pytest.approx(a.estimate, rel=1e-9, abs=1e-9)
        assert b.upper - b.lower >= a.upper - a.lower - 1e-9


@given(st.integers(0, 2 ** 31 - 1), st.floats(0.5, 4.0), st.floats(-20.0, 20.0))
@SETTINGS
def test_the_offset_test_scales_with_the_gap_and_not_with_its_units(seed, scale, shift):
    """The statistic is a ratio, so it is invariant to the outcome's units and
    moves only with a real offset."""
    gaps = np.random.default_rng(seed).normal(0.0, 1.0, 32)
    base = approximability(gaps)
    rescaled = approximability(gaps * scale)
    assert rescaled.t_stat == pytest.approx(base.t_stat, rel=1e-7, abs=1e-9)
    if abs(shift) > 1e-6:
        moved = approximability(gaps + shift)
        assert abs(moved.t_stat - base.t_stat) > 0.0


@given(st.integers(0, 2 ** 31 - 1), st.integers(3, 60), st.floats(-0.95, 0.95),
       st.floats(0.5, 4.0))
@SETTINGS
def test_the_effective_count_stays_inside_its_two_bounds(seed, n, rho, scale):
    """The effective sample size is bounded by the floor and by the window.

    Two is the floor, because a Student-t needs one degree of freedom. The
    period count is the ceiling, because a window of ``n`` periods cannot carry
    the information of more than ``n`` independent ones -- the bound the example
    tests assert at one window length and this asserts over the whole domain of
    lengths, dependences and scales the function accepts.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, scale, n)
    gaps = e.copy()
    if abs(rho) > 1e-9:
        gaps[0] = e[0] / np.sqrt(1.0 - rho ** 2)
        for t in range(1, n):
            gaps[t] = rho * gaps[t - 1] + e[t]
    if gaps.std(ddof=1) <= 0.0:
        return
    check = approximability(gaps)
    assert 2.0 <= check.effective_n <= float(n)


@given(st.integers(0, 2 ** 31 - 1), st.integers(4, 60), st.floats(0.5, 4.0))
@SETTINGS
def test_clipping_can_only_shrink_the_statistic(seed, n, scale):
    """Bounding the effective count never makes the gate more willing to refuse.

    The statistic is ``bias / (sd / sqrt(n_eff))``, monotone in ``n_eff``, so
    capping the count can only move the statistic toward zero. A design the
    clipped test refuses would have been refused by the unclipped one too, which
    is what makes the change safe: it removes false refusals and adds none.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.normal(0.4, scale, n)
    if gaps.std(ddof=1) <= 0.0:
        return
    check = approximability(gaps)
    unclipped_n = max(n * (1.0 - check.serial_correlation)
                      / (1.0 + check.serial_correlation), 2.0)
    unclipped_t = check.bias / (check.scale / np.sqrt(unclipped_n))
    assert abs(check.t_stat) <= abs(unclipped_t) + 1e-9
