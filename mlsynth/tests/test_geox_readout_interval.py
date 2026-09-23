"""The realized readout reports an interval, not only a p-value.

``InferenceResults`` has carried ``ci_lower`` / ``ci_upper`` all along and the
GEOX readout never filled them, so ``report.att_ci`` was ``None`` and coverage
-- the headline metric of any inference comparison -- was not a property of what
GEOX ships. Both nulls can produce one: the placebo procedure reports a scale,
and a conformal test can be inverted over a grid of constant effects.

The two intervals are not the same object and the tests keep them apart. The
placebo interval is ``att +/- z sigma``, symmetric by construction. The
conformal interval is the set of constant effects the test does not reject, so
it need not be symmetric and its agreement with the p-value is what makes it an
interval at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from mlsynth.utils.geox_helpers.engines import resolve_engine

T, PRE, J = 60, 45, 10


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(3)
    f = np.zeros(T)
    e = rng.normal(size=T)
    f[0] = e[0]
    for t in range(1, T):
        f[t] = 0.5 * f[t - 1] + e[t]
    load = rng.uniform(0.8, 1.2, size=J)
    Y0 = 500.0 + 40.0 * np.outer(f, load) + rng.normal(scale=8.0, size=(T, J))
    y = 500.0 + 40.0 * f * load.mean() + rng.normal(scale=8.0, size=T)
    return y, Y0


def _readout(series, engine, inference, alpha=0.1, **kw):
    y, Y0 = series
    eng = resolve_engine(engine)
    fit = eng.fit_once(y, Y0, PRE, PRE, T - 1, 1, augment="ridge",
                       fixed_effects=True)
    p, details = eng.point_inference(fit, y, Y0, PRE, PRE, T - 1,
                                     inference=inference, alpha=alpha,
                                     n_draws=80, seed=0, **kw)
    return eng.att(fit, y, PRE, T - 1), p, details


# --- the placebo interval -------------------------------------------------


@pytest.mark.parametrize("engine", ["sdid", "augsynth"])
def test_placebo_reports_the_normal_interval(series, engine):
    att, _, d = _readout(series, engine, "placebo")
    z = norm.ppf(0.95)
    assert d["ci_lower"] == pytest.approx(att - z * d["sigma"])
    assert d["ci_upper"] == pytest.approx(att + z * d["sigma"])


@pytest.mark.parametrize("engine", ["sdid", "augsynth"])
def test_the_interval_brackets_the_estimate(series, engine):
    att, _, d = _readout(series, engine, "placebo")
    assert d["ci_lower"] < att < d["ci_upper"]


@pytest.mark.parametrize("engine", ["sdid", "augsynth"])
def test_a_smaller_alpha_widens_it(series, engine):
    _, _, wide = _readout(series, engine, "placebo", alpha=0.01)
    _, _, narrow = _readout(series, engine, "placebo", alpha=0.20)
    assert (wide["ci_upper"] - wide["ci_lower"]
            > narrow["ci_upper"] - narrow["ci_lower"])


@pytest.mark.parametrize("engine", ["sdid", "augsynth"])
def test_the_interval_agrees_with_the_p_value(series, engine):
    """Excluding zero and rejecting at ``alpha`` are the same statement.

    Both come from ``|att| / sigma`` against the same critical value, so an
    interval that excluded zero while the p-value did not reject would mean the
    readout reports two different tests.
    """
    att, p, d = _readout(series, engine, "placebo", alpha=0.1)
    excludes_zero = not (d["ci_lower"] <= 0.0 <= d["ci_upper"])
    assert excludes_zero == (p <= 0.1)


# --- the conformal interval, by bracketing --------------------------------


def _additive(true_effect, seed=5):
    """A panel with a genuinely CONSTANT effect, which is what is inverted for."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    e = rng.normal(size=T)
    f[0] = e[0]
    for t in range(1, T):
        f[t] = 0.5 * f[t - 1] + e[t]
    load = rng.uniform(0.9, 1.1, size=J)
    Y0 = 5000.0 + 300.0 * np.outer(f, load) + rng.normal(scale=60.0, size=(T, J))
    y = 5000.0 + 300.0 * f * load.mean() + rng.normal(scale=60.0, size=T)
    y[PRE:] += true_effect
    return y, Y0


def _interval(y, Y0, alpha=0.1):
    from mlsynth.utils.bilevel.ridge_inference import conformal_att_interval

    eng = resolve_engine("augsynth")
    fit = eng.fit_once(y, Y0, PRE, PRE, T - 1, 1, augment="ridge",
                       fixed_effects=True)
    return conformal_att_interval(
        y, Y0, PRE, alpha=alpha, ns=200, fixed_effects=True,
        lambda_=fit.extras["lambda_"]), eng.att(fit, y, PRE, T - 1)


@pytest.mark.parametrize("true_effect", [0.0, 400.0, 1200.0])
def test_the_conformal_interval_covers_a_constant_effect(true_effect):
    """It brackets the truth, and tracks it instead of sitting still."""
    (lo, hi), att = _interval(*_additive(true_effect))
    assert lo <= true_effect <= hi
    assert lo <= att <= hi


def test_it_excludes_zero_exactly_when_the_effect_is_real():
    (lo0, hi0), _ = _interval(*_additive(0.0))
    (lo1, hi1), _ = _interval(*_additive(1200.0))
    assert lo0 <= 0.0 <= hi0                 # no effect: zero is not excluded
    assert not (lo1 <= 0.0 <= hi1)           # a large one: it is


def test_the_endpoints_are_where_the_test_changes_answer():
    """Bracketing means the bounds are crossings, not grid points."""
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue

    y, Y0 = _additive(400.0)
    (lo, hi), _ = _interval(y, Y0)
    eng = resolve_engine("augsynth")
    lam = eng.fit_once(y, Y0, PRE, PRE, T - 1, 1, augment="ridge",
                       fixed_effects=True).extras["lambda_"]

    def p_at(tau0):
        shifted = np.array(y, dtype=float)
        shifted[PRE:] -= tau0
        return conformal_pvalue(shifted, Y0, PRE, ns=200, seed=0,
                                conformal_type="block", fixed_effects=True,
                                lambda_=lam)

    width = hi - lo
    assert p_at(lo + 0.05 * width) > 0.1     # inside is accepted
    assert p_at(hi - 0.05 * width) > 0.1
    assert p_at(lo - 0.30 * width) <= 0.1    # outside is rejected
    assert p_at(hi + 0.30 * width) <= 0.1


def _recast(scenario, seed):
    """A Recast panel with its fitted penalty, and the post counterfactual mean."""
    from mlsynth.utils.geox_helpers.simulate import simulate_recast_panel

    wide, treated, cf, pre = simulate_recast_panel(scenario, 0.0, seed)
    y = wide[treated].to_numpy()
    Y0 = wide.drop(columns=[treated]).to_numpy()
    eng = resolve_engine("augsynth")
    fit = eng.fit_once(y, Y0, pre, pre, wide.shape[0] - 1, 1, augment="ridge",
                       fixed_effects=True)
    return (y, Y0, pre, dict(ns=300, seed=0, conformal_type="block",
                             fixed_effects=True,
                             lambda_=fit.extras["lambda_"]),
            float(np.mean(cf[pre:])))


def _a4(seed):
    """An A4 panel -- 30 pre-days, so the test is weak."""
    y, Y0, pre, kw, _ = _recast("A4", seed)
    return y, Y0, pre, kw


# --- an empty set is a claim about every candidate, not about one ---------


def test_a_rejected_point_estimate_does_not_make_the_set_empty():
    """The confidence set is ``{tau : p(tau) > alpha}``, and ``p`` does not peak
    at the pre-period estimate.

    ``conformal_pvalue`` refits on every period, so shifting the post block by a
    candidate changes the fit as well as the residual, and the most conforming
    candidate can sit well away from the estimate. On Recast A1 seed 23 the
    estimate is -4.7% and is rejected at ``p = 0.010``, while the test accepts
    the whole band around -30%. Concluding empty from the estimate alone throws
    that band away, and the interval misses the truth for a reason that has
    nothing to do with the truth.
    """
    from mlsynth.utils.bilevel.ridge_inference import (
        conformal_att_interval, conformal_pvalue,
    )

    y, Y0, pre, kw, cfm = _recast("A1", 23)
    inside = -0.30 * cfm

    shifted = np.array(y, dtype=float)
    shifted[pre:] -= inside
    assert conformal_pvalue(shifted, Y0, pre, **kw) > 0.05   # it is accepted

    lo, hi = conformal_att_interval(y, Y0, pre, alpha=0.05, **kw)
    assert not (np.isnan(lo) or np.isnan(hi))
    assert lo <= inside <= hi


def test_a_set_the_test_rejects_everywhere_is_still_empty():
    """Relocating the centre must not manufacture an interval.

    On Recast A1 seed 45 the statistic is minimised at the point estimate and
    the test rejects even there, so no constant effect conforms.
    """
    from mlsynth.utils.bilevel.ridge_inference import conformal_att_interval

    y, Y0, pre, kw, _ = _recast("A1", 45)
    lo, hi = conformal_att_interval(y, Y0, pre, alpha=0.05, **kw)
    assert np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("seed", [0, 2, 6])
def test_a_finite_bound_is_always_a_crossing(seed):
    """The search span starts the doubling; it does not cap the answer.

    Recast's A4 panels are where this bites: 30 pre-days leave the test so
    little power that a crossing can sit tens of pre-period sigmas out, or not
    exist at all. A search that stops doubling and returns its own stopping
    point reports a bound narrower than the truth, which under-covers. Seed 6 is
    such a panel; seeds 0 and 2 have genuine crossings.
    """
    from mlsynth.utils.bilevel.ridge_inference import (
        conformal_att_interval, conformal_pvalue,
    )

    y, Y0, pre, kw = _a4(seed)
    lo, hi = conformal_att_interval(y, Y0, pre, alpha=0.05, **kw)
    assert not (np.isnan(lo) or np.isnan(hi))       # the centre is accepted

    def p_at(tau0):
        shifted = np.array(y, dtype=float)
        shifted[pre:] -= tau0
        return conformal_pvalue(shifted, Y0, pre, **kw)

    step = 0.01 * (hi - lo)
    for bound, out in ((lo, -1), (hi, +1)):
        if not np.isfinite(bound):
            continue                                # unbounded: nothing to check
        assert p_at(bound - out * step) > 0.05      # inside is accepted ...
        assert p_at(bound + out * step) <= 0.05     # ... outside is rejected


def test_a_panel_the_test_never_rejects_on_reports_no_bound():
    """A4 seed 6 is the case, and it used to come back as two numbers.

    The block reference set is the ``T`` cyclic shifts of the residual path.
    With 15 of 45 periods in the post block, a third of those shifts overlap it,
    so a candidate driven arbitrarily far still leaves five of them at least as
    extreme and the p-value plateaus at ``5 / 45``, never reaching ``0.05``.
    Nothing is excluded in either direction and the set is the whole line.
    """
    from mlsynth.utils.bilevel.ridge_inference import (
        conformal_att_interval, conformal_pvalue,
    )

    y, Y0, pre, kw = _a4(6)
    far = 1e4 * float(np.mean(np.abs(y[pre:])))
    for tau0 in (-far, far):
        shifted = np.array(y, dtype=float)
        shifted[pre:] -= tau0
        assert conformal_pvalue(shifted, Y0, pre, **kw) > 0.05
    assert conformal_att_interval(y, Y0, pre, alpha=0.05, **kw) == (-np.inf,
                                                                   np.inf)


def test_a_level_the_test_cannot_reach_gives_an_unbounded_interval():
    """Block conformal compares against ``T`` cyclic shifts, one of which is the
    observed path itself, so its p-value cannot fall below ``1 / T``. Below that
    level no candidate is ever rejected and the confidence set is the whole
    line. Reporting a finite bound there would invent a rejection.
    """
    from mlsynth.utils.bilevel.ridge_inference import conformal_att_interval

    y, Y0 = _additive(0.0)
    assert 1.0 / T > 0.01                     # the level is below the floor
    lo, hi = conformal_att_interval(
        y, Y0, PRE, alpha=0.01, ns=200, fixed_effects=True,
        conformal_type="block")
    assert lo == -np.inf and hi == np.inf


def test_the_readout_reports_an_unbounded_side_as_infinite():
    """``None`` would make an unbounded set indistinguishable from an empty one."""
    eng = resolve_engine("augsynth")
    y, Y0 = _additive(0.0)
    fit = eng.fit_once(y, Y0, PRE, PRE, T - 1, 1, augment="ridge",
                       fixed_effects=True)
    _, d = eng.point_inference(fit, y, Y0, PRE, PRE, T - 1,
                               inference="conformal", conformal_type="block",
                               ns=200, seed=0, alpha=0.01,
                               augment="ridge", fixed_effects=True)
    assert d["ci_lower"] == -np.inf and d["ci_upper"] == np.inf


def test_a_multiplicative_lift_has_no_constant_effect_interval():
    """An empty set is the right answer, not a search failure.

    The null inverted is a *constant* effect. A multiplicative lift on a
    trending panel is not constant in levels, so once the noise is small enough
    to resolve the difference no candidate conforms and the set is empty. That
    is the test being right, and it is why the readout can report absent.
    """
    y, Y0 = _additive(0.0)
    y[PRE:] *= 1.25                          # proportional, not additive
    (lo, hi), _ = _interval(y, Y0)
    assert np.isnan(lo) and np.isnan(hi)


# --- the readout populates the result model -------------------------------


@pytest.mark.parametrize("engine, inference", [
    ("sdid", "placebo"), ("augsynth", "placebo"),
])
def test_the_report_exposes_att_ci(engine, inference):
    """Through the estimator, on the standardized accessor."""
    import pandas as pd

    from mlsynth import GEOX
    from mlsynth.config_models import GEOXConfig

    rng = np.random.default_rng(11)
    n_geo, n_t, pre = 12, 60, 45
    f = np.zeros(n_t)
    e = rng.normal(size=n_t)
    f[0] = e[0]
    for t in range(1, n_t):
        f[t] = 0.5 * f[t - 1] + e[t]
    load = rng.uniform(0.8, 1.2, size=n_geo)
    Y = 3000.0 + 200.0 * np.outer(f, load) + rng.normal(scale=60.0,
                                                        size=(n_t, n_geo))
    Y[pre:, 0] *= 1.10                       # a real lift on the treated geo
    long = (pd.DataFrame(Y, columns=[f"g{i:02d}" for i in range(n_geo)])
            .rename_axis("t").reset_index()
            .melt(id_vars="t", var_name="geo", value_name="Y"))
    long["post"] = (long["t"] >= pre).astype(int)

    res = GEOX(GEOXConfig(
        df=long, unitid="geo", time="t", outcome="Y", post_col="post",
        treatment_size=1, to_be_treated=["g00"], durations=[10],
        effect_sizes=[0.0, 0.1], n_backtests=2, n_draws=40, seed=0, n_jobs=1,
        n_validation_backtests=0, alpha=0.05, power_threshold=0.1,
        engine=engine, inference=inference,
    )).fit()

    ci = res.report.att_ci
    assert ci is not None, "att_ci is None; the readout did not fill the interval"
    assert ci[0] < ci[1]
    assert res.report.inference.confidence_level == pytest.approx(0.95)
