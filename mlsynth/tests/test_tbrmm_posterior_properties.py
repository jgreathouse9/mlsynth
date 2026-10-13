"""Generative properties for TBRMM's posterior.

The example tests fix a panel and so can only find what that panel's geometry
exposes. These assert what the posterior means for every panel.

Three are exact, which is what makes them sharp. The ATT bounds are the
cumulative bounds over a known constant, so the ratio holds to machine
precision. A constant injected into the treated post periods moves the interval
by exactly the matching amount, because the fit never sees the post window and
the regressor does not move. And scaling the panel scales the bounds, because
the posterior carries the outcome's units.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy import stats

from mlsynth.utils.tbr_helpers.design.estimate import measure_design

SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def panels(draw, min_treated: int = 1):
    n_treated = draw(st.integers(min_treated, 3))
    n_control = draw(st.integers(2, 5))
    n_pre = draw(st.integers(15, 40))
    n_post = draw(st.integers(2, 12))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    n_units, T = n_treated + n_control, n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) * draw(st.floats(0.5, 4.0)) + 100.0
    level = rng.uniform(5.0, 50.0, n_units)
    loading = rng.uniform(0.5, 1.5, n_units)
    Y = level[None, :] + loading[None, :] * factor[:, None] + rng.normal(0, 1.0, (T, n_units))
    units = [f"u{j}" for j in range(n_units)]
    pre, post = Y[:n_pre], Y[n_pre:]
    assume(float(np.std(pre[:, n_treated:].mean(axis=1))) > 1e-6)
    return pre, post, units, units[:n_treated], units[n_treated:]


@SETTINGS
@given(panels())
def test_the_att_bounds_are_the_cumulative_bounds_rescaled(data):
    """Division by a known constant: exact, and by geos as well as periods."""
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control)

    k = eff.n_treated * eff.n_post
    assert eff.posterior.att_lower == pytest.approx(eff.posterior.total_lower / k, rel=1e-12)
    assert eff.posterior.att_upper == pytest.approx(eff.posterior.total_upper / k, rel=1e-12)
    for m in eff.market_effects:
        assert m.posterior.att_lower == pytest.approx(
            m.posterior.total_lower / eff.n_post, rel=1e-12)


@SETTINGS
@given(panels())
def test_the_interval_always_brackets_its_point_estimate(data):
    """A two-sided interval around a location contains that location."""
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control)

    assert eff.posterior.total_lower <= eff.total_effect <= eff.posterior.total_upper
    assert eff.posterior.att_lower <= eff.att <= eff.posterior.att_upper
    for m in eff.market_effects:
        assert m.posterior.total_lower <= m.total_effect <= m.posterior.total_upper


@SETTINGS
@given(panels(), st.floats(-400, 400, allow_nan=False, allow_infinity=False))
def test_an_injected_constant_slides_the_interval_without_resizing_it(data, tau):
    """The fit never sees the post window, so only the location moves."""
    pre, post, units, treated, control = data
    cols = [units.index(u) for u in treated]
    base, _, _ = measure_design(pre, post, units, treated, control)
    lifted = post.copy()
    lifted[:, cols] += tau
    moved, _, _ = measure_design(pre, lifted, units, treated, control)

    shift = tau * base.n_post * base.n_treated
    assert moved.posterior.total_lower - base.posterior.total_lower == pytest.approx(
        shift, rel=1e-6, abs=1e-6)
    assert moved.posterior.scale == pytest.approx(base.posterior.scale, rel=1e-9)


@SETTINGS
@given(panels(), st.floats(0.25, 40.0, allow_nan=False, allow_infinity=False))
def test_scaling_the_panel_scales_the_interval(data, c):
    """The posterior carries the outcome's units; the direction is unitless."""
    pre, post, units, treated, control = data
    base, _, _ = measure_design(pre, post, units, treated, control)
    scaled, _, _ = measure_design(pre * c, post * c, units, treated, control)

    assert scaled.posterior.scale == pytest.approx(base.posterior.scale * c, rel=1e-6)
    assert scaled.posterior.total_lower == pytest.approx(
        base.posterior.total_lower * c, rel=1e-6)
    assert scaled.posterior.prob_direction == pytest.approx(
        base.posterior.prob_direction, rel=1e-6, abs=1e-9)


@SETTINGS
@given(panels(), st.floats(0.5, 0.80), st.floats(0.90, 0.995))
def test_a_wider_level_nests_a_narrower_one(data, lo_level, hi_level):
    """Raising the level can only add mass, never move a bound inward."""
    pre, post, units, treated, control = data
    narrow, _, _ = measure_design(pre, post, units, treated, control, level=lo_level)
    wide, _, _ = measure_design(pre, post, units, treated, control, level=hi_level)

    assert wide.posterior.total_lower <= narrow.posterior.total_lower
    assert wide.posterior.total_upper >= narrow.posterior.total_upper
    assert wide.posterior.scale == pytest.approx(narrow.posterior.scale, rel=1e-12)


@SETTINGS
@given(panels())
def test_the_degrees_of_freedom_are_the_pretest_length_less_two(data):
    """Eqn 1 fits two parameters, whatever the geo or the level."""
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control)

    assert eff.posterior.df == pre.shape[0] - 2
    assert all(m.posterior.df == pre.shape[0] - 2 for m in eff.market_effects)


@SETTINGS
@given(panels())
def test_direction_agrees_with_the_sign_of_the_estimate(data):
    """prob_direction is mass on the estimate's own side, so never below a half."""
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control)

    for q in [eff.posterior] + [m.posterior for m in eff.market_effects]:
        assert 0.5 <= q.prob_direction <= 1.0


@SETTINGS
@given(panels(min_treated=2))
def test_the_group_scale_is_not_the_markets_in_quadrature(data):
    """Correlated residuals: the group fit prices them, a quadrature sum cannot.

    Asserting only that the two differ would pass on a wrong implementation that
    differed for some other reason, so this pins the group scale to a direct fit
    of the summed series -- the thing a quadrature sum is standing in for.
    """
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control)
    cols = [units.index(u) for u in treated]

    direct, _, _ = measure_design(
        np.column_stack([pre[:, cols].sum(axis=1), pre[:, [units.index(u) for u in control]]]),
        np.column_stack([post[:, cols].sum(axis=1), post[:, [units.index(u) for u in control]]]),
        ["sum"] + list(control), ["sum"], list(control))
    assert eff.posterior.scale == pytest.approx(direct.posterior.scale, rel=1e-9)
    assert eff.total_effect == pytest.approx(direct.total_effect, rel=1e-9)


@SETTINGS
@given(panels(), st.floats(0.80, 0.99))
def test_the_bounds_are_cut_on_the_t_on_n_minus_two_degrees_of_freedom(data, level):
    """Section 9.1's quantile, pinned against the normal's.

    Every other property here is a ratio, a nesting relation or a shift, and the
    normal satisfies all of them, so the distribution the bounds are cut on needs
    an assertion of its own. At the pretest lengths a geo test has, the t is
    visibly the wider of the two, and using the normal would understate the
    interval on every panel.
    """
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control, level=level)
    q = eff.posterior

    half = (q.total_upper - q.total_lower) / 2.0
    assert half == pytest.approx(
        stats.t.ppf(0.5 * (1.0 + level), q.df) * q.scale, rel=1e-10)
    assert half > stats.norm.ppf(0.5 * (1.0 + level)) * q.scale
