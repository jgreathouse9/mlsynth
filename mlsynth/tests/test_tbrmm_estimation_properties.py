"""Generative property tests for TBRMM's market-level estimation phase.

The measured effect is Li and Van den Bulte's equation (2.4) fitted per treated
geo on the pretest and projected through the post window, pooled by averaging
the per-geo effects (their Appendix C). Five things follow from that definition
for every panel, not for the particular ones a fixture happens to pick, so they
are asserted over the domain.

Why this layer. A fixture test fixes the panel and so can only find a fault the
fixture's geometry exposes. The properties below are the estimator's meaning
written as invariants, and three of them are exact, not approximate,
which is what makes them sharp:

``att`` equals the mean of ``market_effects`` because that is how Appendix C
pools, and it is the property that lets a market-level table be published beside
a headline. If it ever fails the two numbers contradict each other on the page.

A constant added to the treated geos' post periods moves every per-market effect
by exactly that constant. The regression is fitted on the pretest alone, so the
injection cannot move the counterfactual, and the difference is the injection
itself with nothing left over. This is the recovery claim with the ground truth
set by construction.

Scaling the panel scales the effect and leaves the percent effect alone, which
says the estimator carries the outcome's units and does not invent a scale of
its own.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.tbrmm_helpers.estimate import measure_design


SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

@st.composite
def panels(draw, min_treated: int = 1):
    """A pretest matrix, a post matrix and a treated/control split.

    Units share a drifting common factor with unit-specific loadings, which is
    the geometry a geo panel has and the one that makes the control average a
    usable regressor. The control average is required to move over the pretest,
    since a constant one identifies no scale and the estimator refuses it.
    """
    n_treated = draw(st.integers(min_treated, 3))
    n_control = draw(st.integers(2, 5))
    n_pre = draw(st.integers(12, 40))
    n_post = draw(st.integers(1, 12))
    seed = draw(st.integers(0, 2**31 - 1))

    rng = np.random.default_rng(seed)
    n_units = n_treated + n_control
    T = n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) * draw(st.floats(0.5, 4.0)) + 100.0
    level = rng.uniform(5.0, 50.0, n_units)
    loading = rng.uniform(0.5, 1.5, n_units)
    Y = level[None, :] + loading[None, :] * factor[:, None] + rng.normal(0, 1.0, (T, n_units))

    units = [f"u{j}" for j in range(n_units)]
    treated, control = units[:n_treated], units[n_treated:]
    pre, post = Y[:n_pre], Y[n_pre:]
    assume(float(np.std(pre[:, n_treated:].mean(axis=1))) > 1e-6)
    return pre, post, units, treated, control


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@SETTINGS
@given(panels())
def test_pooled_effect_is_the_mean_of_its_parts(data):
    """Appendix C's pooling rule, exactly."""
    pre, post, units, treated, control = data
    effect, _, _ = measure_design(pre, post, units, treated, control)

    parts = np.array([m.att for m in effect.market_effects], dtype=float)
    assert parts.size == len(treated)
    assert effect.att == pytest.approx(float(parts.mean()), rel=1e-9, abs=1e-9)


@SETTINGS
@given(panels(), st.floats(-500, 500, allow_nan=False, allow_infinity=False))
def test_an_injected_constant_moves_every_market_by_exactly_that_constant(data, tau):
    """The fit never sees the post window, so the injection survives intact."""
    pre, post, units, treated, control = data
    treated_cols = [units.index(u) for u in treated]

    clean, _, _ = measure_design(pre, post, units, treated, control)
    lifted = post.copy()
    lifted[:, treated_cols] += tau
    moved, _, _ = measure_design(pre, lifted, units, treated, control)

    assert moved.att - clean.att == pytest.approx(tau, rel=1e-7, abs=1e-7)
    for before, after in zip(clean.market_effects, moved.market_effects):
        assert after.att - before.att == pytest.approx(tau, rel=1e-7, abs=1e-7)


@SETTINGS
@given(panels(), st.floats(0.25, 40.0, allow_nan=False, allow_infinity=False))
def test_scaling_the_panel_scales_the_effect_and_not_the_percent(data, c):
    """The effect carries the outcome's units; the percent effect does not."""
    pre, post, units, treated, control = data
    base, _, _ = measure_design(pre, post, units, treated, control)
    scaled, _, _ = measure_design(pre * c, post * c, units, treated, control)

    assert scaled.att == pytest.approx(base.att * c, rel=1e-7, abs=1e-7)
    if base.att_percent is not None and scaled.att_percent is not None:
        assert scaled.att_percent == pytest.approx(base.att_percent, rel=1e-6, abs=1e-6)


@SETTINGS
@given(panels(), st.floats(-300, 300, allow_nan=False, allow_infinity=False))
def test_shifting_a_geos_whole_series_leaves_its_effect_alone(data, k):
    """A level shift in both windows is absorbed by the fitted intercept."""
    pre, post, units, treated, control = data
    col = units.index(treated[0])

    base, _, _ = measure_design(pre, post, units, treated, control)
    pre2, post2 = pre.copy(), post.copy()
    pre2[:, col] += k
    post2[:, col] += k
    shifted, _, _ = measure_design(pre2, post2, units, treated, control)

    assert shifted.market_effects[0].att == pytest.approx(
        base.market_effects[0].att, rel=1e-6, abs=1e-6)
    assert shifted.market_effects[0].delta2 == pytest.approx(
        base.market_effects[0].delta2, rel=1e-6, abs=1e-6)


@SETTINGS
@given(panels(min_treated=2))
def test_the_treated_order_does_not_move_the_pooled_effect(data):
    """Pooling is an average over geos, so their listed order is immaterial."""
    pre, post, units, treated, control = data

    forward, _, _ = measure_design(pre, post, units, treated, control)
    backward, _, _ = measure_design(pre, post, units, list(reversed(treated)), control)

    assert backward.att == pytest.approx(forward.att, rel=1e-9, abs=1e-9)
    assert [m.unit for m in backward.market_effects] == list(reversed(treated))


@SETTINGS
@given(panels())
def test_the_total_is_summed_over_geos_and_periods(data):
    """`total_effect` is the program's incremental total, not an average."""
    pre, post, units, treated, control = data
    effect, _, _ = measure_design(pre, post, units, treated, control)

    expected = sum(m.att * effect.n_post for m in effect.market_effects)
    assert effect.total_effect == pytest.approx(expected, rel=1e-7, abs=1e-7)


@SETTINGS
@given(panels())
def test_the_reported_trajectories_reproduce_the_pooled_effect(data):
    """The contract's two series carry the same number the effect reports."""
    pre, post, units, treated, control = data
    effect, treated_path, control_path = measure_design(pre, post, units, treated, control)

    gap_post = (treated_path - control_path)[pre.shape[0]:]
    assert float(gap_post.mean()) == pytest.approx(effect.att, rel=1e-7, abs=1e-7)
    assert treated_path.size == pre.shape[0] + post.shape[0]


# ---------------------------------------------------------------------------
# Two properties that pin the counterfactual itself
# ---------------------------------------------------------------------------
#
# The properties above are all differences or ratios, and a counterfactual can
# be wrong without disturbing any of them: injecting a constant moves the gap by
# that constant whatever the gap was measured against. These two assert the
# level, which is where a mis-specified counterfactual lives.

@SETTINGS
@given(st.floats(0.3, 2.5, allow_nan=False, allow_infinity=False),
       st.floats(-40.0, 40.0, allow_nan=False, allow_infinity=False),
       st.integers(0, 2**31 - 1))
def test_the_fitted_scale_recovers_a_known_control_relation(b, a, seed):
    """delta_2 is estimated, not assumed: build the relation and read it back.

    Forcing the scale to one is the reduction to plain difference-in-differences
    that equation (2.4) exists to avoid, and it is invisible to every difference
    or ratio property, so it is asserted directly.
    """
    rng = np.random.default_rng(seed)
    n_pre, n_post, n_control = 60, 8, 3
    control = 100.0 + np.cumsum(rng.normal(size=n_pre + n_post)) * 3.0
    controls = control[:, None] + rng.normal(0, 0.05, (n_pre + n_post, n_control))
    treated = a + b * controls.mean(axis=1) + rng.normal(0, 0.05, n_pre + n_post)

    Y = np.column_stack([treated, controls])
    units = ["t", "c0", "c1", "c2"]
    effect, _, _ = measure_design(Y[:n_pre], Y[n_pre:], units, ["t"], units[1:])

    assert effect.market_effects[0].delta2 == pytest.approx(b, rel=0.02, abs=0.02)
    assert effect.att == pytest.approx(0.0, abs=0.25)


@SETTINGS
@given(st.floats(20.0, 400.0, allow_nan=False, allow_infinity=False),
       st.floats(0.4, 2.0, allow_nan=False, allow_infinity=False),
       st.integers(0, 2**31 - 1))
def test_control_side_movement_is_not_charged_to_the_treatment(jump, b, seed):
    """Everything moves in the post window and nothing was treated.

    The counterfactual has to follow the realized control path, so a shift the
    control geos share with the treated one leaves no effect behind. Projecting
    the pretest controls through the window instead would book the whole shift
    as treatment.
    """
    rng = np.random.default_rng(seed)
    n_pre, n_post, n_control = 60, 10, 3
    base = 100.0 + np.cumsum(rng.normal(size=n_pre + n_post)) * 2.0
    base[n_pre:] += jump                       # a common post-window shift
    controls = base[:, None] + rng.normal(0, 0.05, (n_pre + n_post, n_control))
    treated = 5.0 + b * controls.mean(axis=1) + rng.normal(0, 0.05, n_pre + n_post)

    Y = np.column_stack([treated, controls])
    units = ["t", "c0", "c1", "c2"]
    effect, _, _ = measure_design(Y[:n_pre], Y[n_pre:], units, ["t"], units[1:])

    # The shift puts the regressor outside its pretest range, and the
    # prediction error of a projection grows with that distance, so the
    # tolerance does too. It stays far below the shift itself, which is what a
    # counterfactual built from the pretest controls would report.
    assert effect.att == pytest.approx(0.0, abs=0.02 * jump + 0.2)
