r"""Generative properties for TBRMM's HAC variance.

The example tests fix one panel. These assert what the Newey-West construction
means for every panel: it is a reweighted quadratic form in the pretest
residuals, so it carries the outcome's units, it reduces to the diagonal
sandwich when the taper reaches no lags, and it changes only the width of an
interval and never its centre.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.tbr_helpers.design.estimate import _hac_scale, _newey_west, measure_design

SETTINGS = settings(max_examples=50, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def panels(draw):
    n_treated = draw(st.integers(1, 2))
    n_control = draw(st.integers(2, 5))
    n_pre = draw(st.integers(20, 50))
    n_post = draw(st.integers(2, 12))
    rho = draw(st.floats(-0.85, 0.85))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    n_units, T = n_treated + n_control, n_pre + n_post
    factor = np.cumsum(rng.normal(size=T)) * draw(st.floats(0.5, 3.0)) + 100.0
    lv = rng.uniform(5.0, 50.0, n_units)
    ld = rng.uniform(0.5, 1.5, n_units)
    Y = lv[None, :] + ld[None, :] * factor[:, None] + rng.normal(0, 1.0, (T, n_units))
    e = np.zeros(T); inn = rng.normal(0, 2.0, T)
    for t in range(1, T):
        e[t] = rho * e[t - 1] + inn[t]
    Y[:, 0] += e
    units = [f"u{j}" for j in range(n_units)]
    pre, post = Y[:n_pre], Y[n_pre:]
    assume(float(np.std(pre[:, n_treated:].mean(axis=1))) > 1e-6)
    return pre, post, units, units[:n_treated], units[n_treated:]


@SETTINGS
@given(panels())
def test_no_lags_leaves_only_the_diagonal(data):
    """At bandwidth zero the taper reaches nothing, so the meat is the sandwich."""
    pre, post, units, treated, control = data
    x = pre[:, [units.index(u) for u in control]].mean(axis=1)
    y = pre[:, units.index(treated[0])]
    X = np.column_stack([np.ones_like(x), x])
    u = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    lrv, meat = _newey_west(u, X, 0)
    dfc = u.size / (u.size - 2.0)
    assert lrv == pytest.approx(float(u @ u) / u.size * dfc, rel=1e-10)
    assert meat == pytest.approx((X * u[:, None]).T @ (X * u[:, None]) * dfc, rel=1e-10)


@SETTINGS
@given(panels(), st.integers(0, 6))
def test_the_scale_is_positive_and_finite(data, lag):
    """A Bartlett taper keeps the form positive semi-definite, so the root is real."""
    pre, post, units, treated, control = data
    assume(lag < pre.shape[0] - 2)
    ci = [units.index(u) for u in control]
    s = _hac_scale(pre[:, units.index(treated[0])], pre[:, ci].mean(axis=1),
                   post[:, ci].mean(axis=1), post.shape[0], lag)
    assert np.isfinite(s) and s > 0.0


@SETTINGS
@given(panels(), st.floats(0.25, 30.0, allow_nan=False, allow_infinity=False))
def test_the_scale_carries_the_outcomes_units(data, c):
    """Scaling the panel scales the standard deviation of a sum by the same factor."""
    pre, post, units, treated, control = data
    ci = [units.index(u) for u in control]; ti = units.index(treated[0])
    args = (pre[:, ti], pre[:, ci].mean(axis=1), post[:, ci].mean(axis=1),
            post.shape[0], 3)
    base = _hac_scale(*args)
    scaled = _hac_scale(pre[:, ti] * c, pre[:, ci].mean(axis=1) * c,
                        post[:, ci].mean(axis=1) * c, post.shape[0], 3)
    assert scaled == pytest.approx(base * c, rel=1e-7)


@SETTINGS
@given(panels())
def test_the_correction_moves_the_width_and_not_the_centre(data):
    """Only the variance changes, so every point estimate is untouched."""
    pre, post, units, treated, control = data
    iid, _, _ = measure_design(pre, post, units, treated, control)
    hac, _, _ = measure_design(pre, post, units, treated, control, variance="hac")

    assert hac.att == pytest.approx(iid.att, rel=1e-12)
    assert hac.total_effect == pytest.approx(iid.total_effect, rel=1e-12)
    assert hac.delta2 == pytest.approx(iid.delta2, rel=1e-12)
    for a, b in zip(iid.market_effects, hac.market_effects):
        assert a.att == pytest.approx(b.att, rel=1e-12)


@SETTINGS
@given(panels())
def test_the_interval_still_brackets_its_estimate(data):
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control, variance="hac")
    q = eff.posterior
    assert q.total_lower <= eff.total_effect <= q.total_upper
    assert q.variance == "hac" and q.bandwidth is not None


@SETTINGS
@given(panels())
def test_the_group_parameters_are_the_summed_series_own_fit(data):
    """delta1/delta2 on the effect describe the group, not any one market."""
    pre, post, units, treated, control = data
    eff, _, _ = measure_design(pre, post, units, treated, control)
    ti = [units.index(u) for u in treated]
    x = pre[:, [units.index(u) for u in control]].mean(axis=1)
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, pre[:, ti].sum(axis=1), rcond=None)[0]
    assert eff.delta1 == pytest.approx(float(coef[0]), rel=1e-9)
    assert eff.delta2 == pytest.approx(float(coef[1]), rel=1e-9)
