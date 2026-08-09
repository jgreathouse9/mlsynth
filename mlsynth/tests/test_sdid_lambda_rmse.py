"""The pre-period fit statistic SDID's own objective minimises.

``rmse_pre`` averages the pre-period gap over every pre-period with equal
weight. Synthetic DiD does not fit that way: it chooses time weights
``lambda`` over the pre-periods and matches a lambda-weighted average, so on a
real panel most pre-periods carry no weight at all. The reported number is
therefore dominated by periods the estimator deliberately set aside, and it runs
systematically higher than the loss the fit was actually chosen to reduce.

This adds the lambda-weighted counterpart alongside it. Two properties give it
its meaning:

* it is a dispersion around the lambda-weighted mean gap, not around zero.
  SDID's level term ``lambda' y_pre - lambda' Y0_pre omega`` absorbs exactly
  that mean, so the statistic measures what the fit leaves after the level is
  taken out -- and it comes to the same thing whether the counterfactual
  carries the level term already or not. That matters here: this estimator's
  default counterfactual is ``Y0 omega`` without it, so the raw pre-period gaps
  sit at a near-constant offset that has nothing to do with fit quality;
* it is therefore invariant to shifting the counterfactual by a constant, which
  is the property that makes it comparable across the two conventions.

``rmse_pre`` itself is untouched: it is pinned by the SDID replications, and the
two numbers answer different questions -- how well the fit tracks all of
history, against how well it satisfies its own criterion.

Under staggered adoption each cohort carries its own ``lambda`` over its own
pre-period, so the statistic is formed per cohort and aggregated with the same
treated-unit weights the aggregate trajectory uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.config_models import SDIDConfig


def _panel(n_units=12, n_periods=30, t_treat=22, n_treated=1, effect=4.0,
           seed=0):
    """A block-treatment panel with a common factor, so lambda has structure."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    factor = 8.0 * np.sin(2 * np.pi * t / 9.0)
    rows = []
    for i in range(n_units):
        level = 100.0 + 12.0 * i
        loading = 0.5 + 0.12 * i
        y = level + loading * factor + rng.normal(0.0, 1.5, n_periods)
        treated = i < n_treated
        for k, period in enumerate(t):
            val = y[k] + (effect if treated and period >= t_treat else 0.0)
            rows.append({"unit": i, "time": int(period), "Y": float(val),
                         "W": int(treated and period >= t_treat)})
    return pd.DataFrame(rows)


def _fit(df, **kw):
    cfg = dict(df=df, outcome="Y", treat="W", unitid="unit", time="time",
               display_graphs=False)
    cfg.update(kw)
    return SDID(SDIDConfig(**cfg)).fit()


@pytest.fixture(scope="module")
def result():
    return _fit(_panel())


# ---------------------------------------------------------------------------
# It is reported
# ---------------------------------------------------------------------------

class TestReported:
    def test_present_and_finite(self, result):
        extra = result.fit_diagnostics.additional_metrics
        assert "rmse_pre_lambda" in extra
        assert np.isfinite(extra["rmse_pre_lambda"])

    def test_rmse_pre_is_unchanged(self, result):
        # The unweighted statistic is pinned by the SDID replications.
        gap = np.asarray(result.time_series.estimated_gap, dtype=float)
        n_pre = int(np.sum(np.asarray(result.time_series.time_periods)
                           < result.time_series.intervention_time))
        expected = float(np.sqrt(np.mean(gap[:n_pre] ** 2)))
        assert result.fit_diagnostics.rmse_pre == pytest.approx(expected,
                                                                rel=1e-9)


# ---------------------------------------------------------------------------
# It is the right quantity
# ---------------------------------------------------------------------------

class TestDefinition:
    def test_matches_the_lambda_weighted_formula(self, result):
        cohort = next(iter(result.cohorts.values()))
        lam = np.asarray(cohort.time_weights, dtype=float)
        gap = (np.asarray(cohort.actual, dtype=float)
               - np.asarray(cohort.counterfactual, dtype=float))[:lam.shape[0]]
        centred = gap - float(lam @ gap)
        expected = float(np.sqrt(lam @ (centred ** 2)))
        assert result.fit_diagnostics.additional_metrics[
            "rmse_pre_lambda"] == pytest.approx(expected, rel=1e-9)

    def test_invariant_to_the_level_convention(self, result):
        # This estimator's counterfactual omits SDID's level term, so the raw
        # pre-period gaps carry a near-constant offset. The statistic has to be
        # blind to it, or it would report the offset instead of the fit.
        from mlsynth.utils.sdid_helpers.orchestration import (
            lambda_weighted_pre_rmse)

        class _Shifted:
            def __init__(self, c, by):
                self.n_treated = c.n_treated
                self.time_weights = c.time_weights
                self.actual = np.asarray(c.actual, dtype=float)
                self.counterfactual = np.asarray(c.counterfactual,
                                                 dtype=float) + by

        cohorts = dict(result.cohorts)
        base = lambda_weighted_pre_rmse(cohorts)
        shifted = lambda_weighted_pre_rmse(
            {k: _Shifted(c, 250.0) for k, c in cohorts.items()})
        assert shifted == pytest.approx(base, rel=1e-9)

    def test_weights_are_a_simplex(self, result):
        cohort = next(iter(result.cohorts.values()))
        lam = np.asarray(cohort.time_weights, dtype=float)
        assert lam.min() >= -1e-9
        assert float(lam.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_it_differs_from_the_unweighted_statistic(self, result):
        # If the two agreed, lambda would be uniform and there would be no
        # reason to report both.
        extra = result.fit_diagnostics.additional_metrics
        assert extra["rmse_pre_lambda"] != pytest.approx(
            result.fit_diagnostics.rmse_pre, rel=1e-6)


# ---------------------------------------------------------------------------
# Invariances the statistic must inherit
# ---------------------------------------------------------------------------

class TestInvariance:
    def test_scales_with_the_outcome(self):
        # A dispersion in outcome units: doubling the panel doubles it.
        df = _panel()
        big = df.copy()
        big["Y"] = big["Y"] * 2.0
        a = _fit(df).fit_diagnostics.additional_metrics["rmse_pre_lambda"]
        b = _fit(big).fit_diagnostics.additional_metrics["rmse_pre_lambda"]
        assert b == pytest.approx(2.0 * a, rel=1e-6)

    def test_unmoved_by_a_common_level_shift(self):
        # SDID differences out levels, so adding a constant to every unit
        # changes neither the weights nor the pre-period dispersion.
        df = _panel()
        shifted = df.copy()
        shifted["Y"] = shifted["Y"] + 500.0
        a = _fit(df).fit_diagnostics.additional_metrics["rmse_pre_lambda"]
        b = _fit(shifted).fit_diagnostics.additional_metrics["rmse_pre_lambda"]
        assert b == pytest.approx(a, rel=1e-6)


# ---------------------------------------------------------------------------
# Staggered adoption: one lambda per cohort
# ---------------------------------------------------------------------------

class TestStaggered:
    def _staggered(self):
        rng = np.random.default_rng(5)
        t = np.arange(34)
        factor = 8.0 * np.sin(2 * np.pi * t / 9.0)
        onset = {0: 22, 1: 27}          # two cohorts
        rows = []
        for i in range(14):
            level = 100.0 + 10.0 * i
            y = level + (0.5 + 0.1 * i) * factor + rng.normal(0, 1.5, 34)
            start = onset.get(i)
            for k, period in enumerate(t):
                on = start is not None and period >= start
                rows.append({"unit": i, "time": int(period),
                             "Y": float(y[k] + (5.0 if on else 0.0)),
                             "W": int(on)})
        return pd.DataFrame(rows)

    def test_aggregates_over_cohorts(self):
        res = _fit(self._staggered())
        assert len(res.cohorts) >= 2
        val = res.fit_diagnostics.additional_metrics["rmse_pre_lambda"]
        assert np.isfinite(val) and val > 0

    def test_matches_the_treated_weighted_combination(self):
        # The same weighting the aggregate trajectory uses, so the statistic
        # describes the path actually reported.
        res = _fit(self._staggered())
        per, wts = [], []
        for c in res.cohorts.values():
            lam = np.asarray(c.time_weights, dtype=float)
            gap = (np.asarray(c.actual, dtype=float)
                   - np.asarray(c.counterfactual, dtype=float))[:lam.shape[0]]
            centred = gap - float(lam @ gap)
            per.append(float(lam @ (centred ** 2)))
            wts.append(max(int(c.n_treated), 1))
        expected = float(np.sqrt(np.average(per, weights=wts)))
        assert res.fit_diagnostics.additional_metrics[
            "rmse_pre_lambda"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

class TestDegenerate:
    def test_absent_when_time_weights_are(self, monkeypatch):
        # A cohort without time weights cannot contribute; the statistic is
        # reported as None instead of guessing a uniform lambda.
        from mlsynth.utils.sdid_helpers import orchestration as orch
        res = _fit(_panel())
        for c in res.cohorts.values():
            object.__setattr__(c, "time_weights", None)
        val = orch.lambda_weighted_pre_rmse(res.cohorts)
        assert val is None

    def test_single_weighted_period_leaves_no_dispersion(self):
        # All weight on one period: the level term matches that period exactly,
        # so the weighted dispersion around the weighted mean is zero. The
        # statistic says the fit satisfies its own criterion perfectly, which
        # on one period it does -- and is why it is reported next to the
        # unweighted RMSE and not instead of it.
        from mlsynth.utils.sdid_helpers.orchestration import (
            lambda_weighted_pre_rmse)

        class _C:
            n_treated = 1
            time_weights = np.array([0.0, 1.0, 0.0])
            actual = np.array([10.0, 12.0, 14.0, 20.0])
            counterfactual = np.array([10.0, 9.0, 14.0, 15.0])

        assert lambda_weighted_pre_rmse({0: _C()}) == pytest.approx(0.0,
                                                                    abs=1e-12)
