"""GEOX realization: reading out the experiment the design chose.

``GEOX.fit()`` returns a :class:`DesignResult` whose ``report`` slot stays
``None`` until outcomes exist. ``realize_design`` fills it: apply the chosen
region's pre-period SDID fit across the full panel and package the realized
effect into the standardized result models.

The realization has to use the same estimator and the same null the design
scored itself with, because GEOX's whole claim is that the design is chosen
by the estimator that will analyse the result. So the readout is SDID plus the
placebo standard error, not a different model that happens to be available. It
also has to inherit the design's donor pool: with a spillover constraint active
the design excluded a treated market's conflict-neighbours, and reading out
against the full complement would answer a different question than the MDE
promised.

``how`` is tested alongside, since it sets the scale everything here reports in.

Tests are TDD-first and layered: premise, invariants of the readout, edge cases
(no post period, single post period, constrained donor pool), and failures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOX
from mlsynth.config_models import BaseEstimatorResults, GEOXConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.geox_helpers.realize import realize_design
from mlsynth.utils.geox_helpers.shaping import aggregate_treated

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def wide(panel):
    return geoex_dataprep(panel, "location", "date", "Y")["Ywide"]


@pytest.fixture(scope="module")
def region(wide):
    return frozenset(list(wide.columns)[:2])


# ---------------------------------------------------------------------------
# Structure of the realized report
# ---------------------------------------------------------------------------

class TestRealizedReport:
    @pytest.fixture(scope="class")
    def report(self, wide, region):
        n_pre = wide.shape[0] - 14
        return realize_design(wide, region, n_pre, n_draws=25, seed=0)

    def test_is_an_effect_result(self, report):
        assert isinstance(report, BaseEstimatorResults)

    def test_series_span_the_whole_panel(self, report, wide):
        ts = report.time_series
        for arr in (ts.observed_outcome, ts.counterfactual_outcome,
                    ts.estimated_gap):
            assert np.asarray(arr).shape[0] == wide.shape[0]
        assert len(ts.time_periods) == wide.shape[0]

    def test_gap_is_observed_minus_counterfactual(self, report):
        ts = report.time_series
        np.testing.assert_allclose(
            np.asarray(ts.estimated_gap),
            np.asarray(ts.observed_outcome) - np.asarray(ts.counterfactual_outcome),
            atol=1e-9)

    def test_att_is_the_post_period_mean_gap(self, report, wide):
        n_pre = wide.shape[0] - 14
        gap = np.asarray(report.time_series.estimated_gap)
        assert report.effects.att == pytest.approx(float(gap[n_pre:].mean()),
                                                   abs=1e-9)

    def test_intervention_time_is_the_split(self, report, wide):
        n_pre = wide.shape[0] - 14
        assert report.time_series.intervention_time == wide.index[n_pre]

    def test_carries_both_sdid_weight_vectors(self, report):
        w = report.weights
        assert w.donor_weights and w.time_weights

    def test_fit_diagnostics_are_finite(self, report):
        fd = report.fit_diagnostics
        assert np.isfinite(fd.rmse_pre)
        assert np.isfinite(fd.additional_metrics["scaled_l2"])

    def test_inference_uses_the_designs_own_null(self, report):
        inf = report.inference
        assert inf.method == "placebo"
        assert 0.0 <= inf.p_value <= 1.0
        assert np.isfinite(inf.details["sigma"])


# ---------------------------------------------------------------------------
# The readout must match the estimator the design scored with
# ---------------------------------------------------------------------------

class TestConsistencyWithTheDesign:
    def test_counterfactual_matches_a_direct_sdid_fit(self, wide, region):
        from mlsynth.utils.geox_helpers.engine import sdid_fit_once
        from mlsynth.utils.geox_helpers.shaping import donor_matrix

        n_pre = wide.shape[0] - 14
        report = realize_design(wide, region, n_pre, n_draws=10, seed=0)
        treated = aggregate_treated(wide, region, how="mean").to_numpy()
        donors = donor_matrix(wide, region).to_numpy()
        fit = sdid_fit_once(treated, donors, n_pre, n_pre, wide.shape[0] - 1,
                            n_tr=len(region))
        np.testing.assert_allclose(
            np.asarray(report.time_series.counterfactual_outcome),
            fit.counterfactual, atol=1e-9)

    def test_p_value_is_reproducible_at_a_fixed_seed(self, wide, region):
        n_pre = wide.shape[0] - 14
        kw = dict(n_draws=15, seed=7)
        a = realize_design(wide, region, n_pre, **kw)
        b = realize_design(wide, region, n_pre, **kw)
        assert a.inference.p_value == b.inference.p_value

    def test_excluded_markets_stay_out_of_the_donor_pool(self, wide, region):
        units = [u for u in wide.columns if u not in region]
        drop = units[0]
        n_pre = wide.shape[0] - 14
        report = realize_design(wide, region, n_pre, exclude=[drop],
                                n_draws=10, seed=0)
        assert str(drop) not in report.weights.donor_weights


# ---------------------------------------------------------------------------
# how: the reporting scale
# ---------------------------------------------------------------------------

class TestHow:
    def test_sum_scales_the_paths_by_the_region_size(self, wide, region):
        n_pre = wide.shape[0] - 14
        mean = realize_design(wide, region, n_pre, how="mean", n_draws=10, seed=0)
        summed = realize_design(wide, region, n_pre, how="sum", n_draws=10, seed=0)
        k = len(region)
        np.testing.assert_allclose(
            np.asarray(summed.time_series.observed_outcome),
            np.asarray(mean.time_series.observed_outcome) * k, rtol=1e-9)
        assert summed.effects.att == pytest.approx(mean.effects.att * k, rel=1e-9)

    def test_scale_does_not_move_the_p_value(self, wide, region):
        # The test statistic is scale-free, so reporting units must not decide
        # significance.
        n_pre = wide.shape[0] - 14
        mean = realize_design(wide, region, n_pre, how="mean", n_draws=15, seed=3)
        summed = realize_design(wide, region, n_pre, how="sum", n_draws=15, seed=3)
        assert summed.inference.p_value == pytest.approx(mean.inference.p_value)

    def test_bad_how_raises(self, wide, region):
        with pytest.raises(MlsynthConfigError, match="how"):
            realize_design(wide, region, wide.shape[0] - 14, how="median")


class TestHowOnTheConfig:
    def _cfg(self, panel, **kw):
        base = dict(df=panel, unitid="location", time="date", outcome="Y",
                    treatment_size=2, durations=[14], effect_sizes=[0.0, 0.2],
                    n_backtests=2, n_draws=5, n_validation_backtests=0, seed=0)
        base.update(kw)
        return GEOXConfig(**base)

    def test_defaults_to_mean(self, panel):
        assert self._cfg(panel).how == "mean"

    def test_rejects_anything_else(self, panel):
        with pytest.raises(MlsynthConfigError, match="how"):
            self._cfg(panel, how="median")

    def test_sum_scales_the_deployable_design(self, panel):
        a = GEOX(self._cfg(panel, how="mean")).fit()
        b = GEOX(self._cfg(panel, how="sum")).fit()
        # Same region chosen; SDID differences out the level, so the scale
        # changes what is reported and not which markets win.
        assert a.selected_units == b.selected_units


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_post_period(self, wide, region):
        n_pre = wide.shape[0] - 1
        report = realize_design(wide, region, n_pre, n_draws=10, seed=0)
        assert np.isfinite(report.effects.att)
        assert len(report.time_series.observed_outcome) == wide.shape[0]

    def test_no_post_period_raises(self, wide, region):
        with pytest.raises(MlsynthConfigError, match="post"):
            realize_design(wide, region, wide.shape[0], n_draws=10, seed=0)

    def test_pre_periods_out_of_range_raises(self, wide, region):
        with pytest.raises(MlsynthConfigError, match="pre_periods"):
            realize_design(wide, region, 0, n_draws=10, seed=0)

    def test_unknown_market_raises(self, wide):
        with pytest.raises(MlsynthDataError):
            realize_design(wide, frozenset({"atlantis"}), wide.shape[0] - 14,
                           n_draws=10, seed=0)

    def test_excluding_every_donor_raises(self, wide, region):
        others = [u for u in wide.columns if u not in region]
        with pytest.raises(MlsynthDataError, match="No donor units remain"):
            realize_design(wide, region, wide.shape[0] - 14, exclude=others,
                           n_draws=10, seed=0)

    def test_cpic_reports_the_realized_spend(self, wide, region):
        n_pre = wide.shape[0] - 14
        report = realize_design(wide, region, n_pre, cpic=7.5, n_draws=10, seed=0)
        gap = np.asarray(report.time_series.estimated_gap)
        expected = 7.5 * float(gap[n_pre:].sum()) * len(region)
        assert report.weights.summary_stats["cost"] == pytest.approx(expected)

    def test_cost_absent_without_cpic(self, wide, region):
        report = realize_design(wide, region, wide.shape[0] - 14, n_draws=10,
                                seed=0)
        assert report.weights.summary_stats["cost"] is None


# ---------------------------------------------------------------------------
# Wiring: GEOX fills report when the panel carries a post period
# ---------------------------------------------------------------------------

class TestFitPopulatesTheReport:
    def _post_panel(self, panel):
        df = panel.copy()
        cutoff = df["date"].sort_values().unique()[-14]
        df["post"] = (df["date"] >= cutoff).astype(int)
        return df

    def test_report_is_none_without_a_post_period(self, panel):
        res = GEOX(GEOXConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=[0.0, 0.2],
            n_backtests=2, n_draws=5, n_validation_backtests=0, seed=0)).fit()
        assert res.report is None

    def test_report_is_filled_when_post_periods_exist(self, panel):
        res = GEOX(GEOXConfig(
            df=self._post_panel(panel), unitid="location", time="date",
            outcome="Y", post_col="post", treatment_size=2, durations=[7],
            effect_sizes=[0.0, 0.2], n_backtests=2, n_draws=5,
            n_validation_backtests=0, seed=0)).fit()
        assert res.report is not None
        assert isinstance(res.report, BaseEstimatorResults)
        assert np.isfinite(res.report.effects.att)
        assert res.report.inference.method == "placebo"

    def test_realized_region_is_the_selected_one(self, panel):
        res = GEOX(GEOXConfig(
            df=self._post_panel(panel), unitid="location", time="date",
            outcome="Y", post_col="post", treatment_size=2, durations=[7],
            effect_sizes=[0.0, 0.2], n_backtests=2, n_draws=5,
            n_validation_backtests=0, seed=0)).fit()
        donors = set(res.report.weights.donor_weights)
        assert not (donors & set(res.selected_units))



# ---------------------------------------------------------------------------
# The readout across engines
#
# `realize_design` was written when SDID was the only engine, whose `extras`
# happen to be all floats (`zeta`, `bias_correction`, `pre_rmspe_lambda`). The
# augsynth engine carries what its estimator has: a penalty that may be absent,
# an augmentation named by string, a flag. `extras` is the engine's own slot and
# the readout does not get to assume a type for it -- the pipeline's premise is
# that everything outside the fit is engine-independent, and a readout that
# runs on one engine and raises on the other is that premise failing.
#
# These parametrize over both engines so the whole readout, not one field, is
# held to it. The five designs are built once and shared: a design on the
# augsynth path costs about seven seconds, and rebuilding one per test put
# thirty-five fits in a suite that needs five.
# ---------------------------------------------------------------------------

_ENGINE_KWARGS = {
    "sdid": {"engine": "sdid"},
    "augsynth": {"engine": "augsynth", "inference": "placebo"},
    "augsynth-conformal": {"engine": "augsynth", "inference": "conformal",
                           "ns": 40},
    "augsynth-plain": {"engine": "augsynth", "inference": "placebo",
                       "augment": None},
    "augsynth-nofe": {"engine": "augsynth", "inference": "placebo",
                      "fixed_effects": False},
}


def _post_panel(panel):
    df = panel.copy()
    cutoff = df["date"].sort_values().unique()[-14]
    df["post"] = (df["date"] >= cutoff).astype(int)
    return df


def _design(panel, **over):
    kwargs = dict(
        df=_post_panel(panel), unitid="location", time="date", outcome="Y",
        post_col="post", treatment_size=2, durations=[7],
        effect_sizes=[0.0, 0.2], n_backtests=2, n_draws=5,
        n_validation_backtests=0, seed=0)
    kwargs.update(over)
    return GEOX(GEOXConfig(**kwargs)).fit()


@pytest.fixture(scope="module")
def readouts(panel):
    """One realized design per engine/inference pairing, built once.

    Every test in this section reads the same five designs and none of them
    mutates one, so the fits are shared. Seeds are fixed, so a shared design is
    the same object a per-test build would have produced.
    """
    return {label: _design(panel, **over).report
            for label, over in _ENGINE_KWARGS.items()}


class TestReadoutRunsOnEveryEngine:
    @pytest.mark.parametrize("label", sorted(_ENGINE_KWARGS))
    def test_the_report_is_produced(self, readouts, label):
        """Smoke: every engine and inference pairing reads out."""
        assert readouts[label] is not None
        assert np.isfinite(readouts[label].effects.att)

    @pytest.mark.parametrize("label", sorted(_ENGINE_KWARGS))
    def test_the_paths_are_finite_and_panel_length(self, readouts, label):
        ts = readouts[label].time_series
        n = len(ts.time_periods)
        for path in (ts.observed_outcome, ts.counterfactual_outcome,
                     ts.estimated_gap):
            assert len(path) == n
            assert np.all(np.isfinite(path))

    @pytest.mark.parametrize("label", sorted(_ENGINE_KWARGS))
    def test_the_p_value_is_a_probability(self, readouts, label):
        assert 0.0 <= readouts[label].inference.p_value <= 1.0


class TestEngineExtrasSurviveTheReadout:
    """`extras` reach `summary_stats` with their own types intact.

    The readout reports what the engine fitted, so a penalty that was absent
    has to arrive as absent and not as a number standing in for it. Coercing
    the slot to float is what made the augsynth readout unreachable.
    """

    def test_sdid_extras_are_reported(self, readouts):
        stats = readouts["sdid"].weights.summary_stats
        for key in ("zeta", "bias_correction", "pre_rmspe_lambda"):
            assert key in stats
            assert isinstance(stats[key], float)

    def test_augsynth_string_extra_is_reported_as_itself(self, readouts):
        assert readouts["augsynth"].weights.summary_stats["augment"] == "ridge"

    def test_augsynth_flag_extra_stays_a_bool(self, readouts):
        assert readouts["augsynth"].weights.summary_stats["fixed_effects"] is True

    def test_a_flag_turned_off_is_reported_off(self, readouts):
        stats = readouts["augsynth-nofe"].weights.summary_stats
        assert stats["fixed_effects"] is False

    def test_an_absent_penalty_is_reported_absent(self, readouts):
        """`augment=None` is plain SCM, which has no ridge penalty.

        `float(None)` raises, and any stand-in value would report a penalty
        that was never applied.
        """
        stats = readouts["augsynth-plain"].weights.summary_stats
        assert stats["augment"] is None
        assert stats["lambda_"] is None

    def test_a_present_penalty_is_a_number(self, readouts):
        stats = readouts["augsynth"].weights.summary_stats
        assert isinstance(stats["lambda_"], float)
        assert np.isfinite(stats["lambda_"])

    @pytest.mark.parametrize("label", sorted(_ENGINE_KWARGS))
    def test_the_shared_summary_keys_are_present_whichever_engine(
            self, readouts, label):
        stats = readouts[label].weights.summary_stats
        for key in ("how", "treated_markets", "engine", "cpic", "cost"):
            assert key in stats

    @pytest.mark.parametrize("label", sorted(_ENGINE_KWARGS))
    def test_no_numpy_scalars_leak_into_the_summary(self, readouts, label):
        """Plain Python scalars, so the report pickles and serialises.

        Every other numeric field in the result models is cast on the way in;
        `extras` is the one slot fed straight from an engine, so it is the one
        that can carry a `np.float64` out.
        """
        stats = readouts[label].weights.summary_stats
        leaked = {k: type(v).__name__ for k, v in stats.items()
                  if isinstance(v, np.generic)}
        assert leaked == {}


class TestExtrasNormalisationUnit:
    """The conversion itself, away from a fit."""

    def test_numpy_scalars_become_python_scalars(self):
        from mlsynth.utils.geox_helpers.realize import _report_extras

        out = _report_extras({"a": np.float64(1.5), "b": np.int64(3),
                              "c": np.bool_(True)})
        assert out == {"a": 1.5, "b": 3, "c": True}
        assert all(not isinstance(v, np.generic) for v in out.values())

    def test_strings_none_and_bools_pass_through(self):
        from mlsynth.utils.geox_helpers.realize import _report_extras

        out = _report_extras({"augment": "ridge", "lambda_": None,
                              "fixed_effects": False})
        assert out == {"augment": "ridge", "lambda_": None,
                       "fixed_effects": False}
        assert out["fixed_effects"] is False

    def test_an_empty_slot_is_an_empty_dict(self):
        from mlsynth.utils.geox_helpers.realize import _report_extras

        assert _report_extras({}) == {}

    def test_nan_survives(self):
        """augsynth reports `pre_rmspe_lambda` as nan: it weights donors only."""
        from mlsynth.utils.geox_helpers.realize import _report_extras

        assert np.isnan(_report_extras({"x": float("nan")})["x"])


class TestTheTwoEnginesDisagree:
    """A guard on the parametrization above.

    If both engines produced the same report the tests would pass while
    testing one estimator twice.
    """

    def test_the_engines_give_different_answers(self, readouts):
        assert readouts["sdid"].effects.att != readouts["augsynth"].effects.att

    def test_only_sdid_reports_time_weights(self, readouts):
        assert readouts["sdid"].weights.time_weights is not None
        assert readouts["augsynth"].weights.time_weights is None
