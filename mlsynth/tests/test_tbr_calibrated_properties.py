"""Properties of TBR's calibrated inference, over the input domain (#1043).

Each invariant the calibration promises, asserted generatively:

* the ADID forecast from origin ``E`` is TBR's own posterior predictive for the
  cumulative effect, centre, scale and degrees of freedom;
* the mixture quantile's bisection returns a point where the mixture CDF has
  reached the target and a point one tolerance below where it has not, so the
  generalised quantile lies within the stated tolerance;
* the distance trend's rescaling multiplies each squared error by the trend at
  the experiment over the trend at its own distance, and is the identity at the
  experiment's distance;
* the calibrated interval is ordered and nests in ``level``;
* placebo groups have the real group's size, carry the forced geos, satisfy the
  rules, are distinct and exclude the real group.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from scipy import special

from mlsynth.tests.test_tbr_design_rules import _rules, problems
from mlsynth.utils.tbr_helpers import calibration as cal
from mlsynth.utils.tbr_helpers.design.search import TREATMENT
from mlsynth.utils.tbr_helpers.posterior import cumulative_posterior, fit_pretest

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def regressions(draw):
    """A pair of series with a fitted relation and noise, and an origin."""
    n = draw(st.integers(6, 40))
    h = draw(st.integers(1, 12))
    seed = draw(st.integers(0, 2 ** 31 - 1))
    rng = np.random.default_rng(seed)
    x = draw(st.floats(1.0, 1e4)) * (1.0 + rng.normal(size=n + h).cumsum() * 0.1)
    y = (draw(finite) + draw(st.floats(-5.0, 5.0)) * x
         + draw(st.floats(0.01, 50.0)) * rng.normal(size=n + h))
    return y, x, n, h


@settings(max_examples=150, deadline=None)
@given(regressions())
def test_the_adid_forecast_is_tbrs_posterior_predictive(case):
    y, x, n, h = case
    f = cal.adid_forecast(y, x, n, h)
    fit = fit_pretest(y[:n], x[:n])
    assume(not fit.rank_deficient)
    loc, scale = cumulative_posterior(fit, y[n:n + h], x[n:n + h])
    observed = np.cumsum(y[n:n + h])
    np.testing.assert_allclose(f.centre, observed - loc, rtol=1e-7,
                               atol=1e-7 * np.abs(observed).max())
    np.testing.assert_allclose(f.scale, scale, rtol=1e-7)
    assert f.df == fit.df == n - 2


@st.composite
def mixtures(draw):
    m = draw(st.integers(0, 30))
    z = np.asarray(draw(st.lists(st.floats(-8.0, 8.0), min_size=m, max_size=m)))
    centre = draw(finite)
    scale = draw(st.floats(1e-3, 1e3))
    df = draw(st.integers(1, 200))
    w0 = 1.0 if m == 0 else draw(st.floats(0.0, 1.0))
    q = draw(st.floats(0.01, 0.99))
    return centre, scale, df, np.sort(centre + scale * z), w0, q


@settings(max_examples=300, deadline=None)
@given(mixtures())
def test_the_mixture_quantile_is_within_tolerance_of_its_target(case):
    centre, scale, df, points, w0, q = case
    v = cal.mixture_quantile(q, centre, scale, df, points, w0)
    step = cal.QUANTILE_TOL * scale
    assert cal.mixture_cdf(v, centre, scale, df, points, w0) >= q
    assert cal.mixture_cdf(v - step, centre, scale, df, points, w0) < q
    if w0 == 1.0:          # a continuous CDF meets the target itself
        t = special.stdtr(df, (v - centre) / scale)
        assert abs(t - q) <= 1e-6


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(-6.0, 6.0), min_size=1, max_size=20),
       st.floats(0.01, 5.0), st.floats(0.0, 2.0), st.integers(2, 60))
def test_the_trend_rescaling_identity(zs, alpha, beta, target):
    z = np.asarray(zs)
    d = np.arange(z.size, dtype=float) % target
    out = cal.rescale_to_experiment(z, d, alpha, beta, float(target))
    ratio = (alpha + beta * target) / np.maximum(alpha + beta * d, cal.TREND_FLOOR)
    np.testing.assert_allclose(out ** 2, z ** 2 * ratio, rtol=1e-12, atol=1e-300)
    assert np.all(np.sign(out) == np.sign(z))
    at = cal.rescale_to_experiment(z, np.full(z.size, float(target)), alpha,
                                   beta, float(target))
    np.testing.assert_allclose(at, z, rtol=1e-12)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(-6.0, 6.0), min_size=2, max_size=40),
       st.integers(1, 30))
def test_the_trend_slope_is_never_negative_and_otherwise_ols(zs, period):
    z = np.asarray(zs)
    d = np.arange(z.size, dtype=float) % period
    assume(np.ptp(d) > 0)
    alpha, beta = cal.distance_trend(d, z)
    assert beta >= 0.0
    b_ols, a_ols = np.polyfit(d, z ** 2, 1)
    if b_ols > 0:
        assert (alpha, beta) == pytest.approx((a_ols, b_ols), rel=1e-6, abs=1e-9)
    # the trend at the experiment is positive whatever the fit
    assert alpha + beta * (d.max() + 1) > 0 or np.all(z ** 2 == 0)


@settings(max_examples=150, deadline=None)
@given(mixtures(), st.floats(0.0, 50.0), st.integers(1, 60),
       st.floats(0.3, 0.6), st.floats(0.05, 0.35))
def test_the_calibrated_interval_nests_in_level(case, n0, n_eff, low, gap):
    centre, scale, df, points, _, _ = case
    assume(points.size > 0)
    z = (points - centre) / scale
    inner = cal.calibrated_interval(centre, scale, df, z, n0, n_eff, low)
    outer = cal.calibrated_interval(centre, scale, df, z, n0, n_eff, low + gap)
    # Each endpoint is its quantile to within the bisection's tolerance, so two
    # quantiles that coincide (an ECDF jump both levels land on) can come back
    # a tolerance apart in either order.
    slack = cal.QUANTILE_TOL * scale
    assert inner[0] <= inner[1] + slack
    assert outer[0] <= inner[0] + slack and inner[1] <= outer[1] + slack


@settings(max_examples=150, deadline=None)
@given(problems(), st.integers(1, 3), st.integers(1, 30), st.integers(0, 99))
def test_placebo_groups_are_admissible_distinct_and_not_the_real_one(
        problem, k, n, seed):
    roles, spec, _ = problem
    rules = _rules(roles, spec)
    forced = [j for j, r in enumerate(roles) if r == frozenset({TREATMENT})]
    treatable = [j for j, r in enumerate(roles) if TREATMENT in r]
    free = [j for j in treatable if j not in forced]
    k = max(k, len(forced))
    assume(len(forced) + len(free) >= k)
    admissible = [sorted(forced + list(c))
                  for c in itertools.combinations(free, k - len(forced))
                  if rules.admits(forced + list(c))]
    assume(admissible)
    real = admissible[0]
    groups, n_adm = cal.placebo_groups(real, roles, rules, n, seed=seed)
    assert n_adm == len(admissible) - 1
    assert len(groups) == min(n, n_adm)
    assert len({tuple(g) for g in groups}) == len(groups)
    for g in groups:
        assert g == sorted(g) and g != real and len(g) == k
        assert set(forced) <= set(g) and rules.admits(g)
