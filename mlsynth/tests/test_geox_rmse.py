"""Estimation error as a GEOX design criterion, beside the MDE.

The MDE asks the smallest effect a design can detect. It says nothing about how
far the estimate will land from the truth once an effect is there, and the two
come apart whenever the estimator is biased under the design -- for synthetic
control, the treated region sitting outside the donor hull.

The error is already determined by quantities the backtest computes. Both
engines define the ATT as the mean gap over the treatment window, and neither
builds its counterfactual from the treated post block, so injecting a
multiplicative lift ``y[post] *= (1 + e)`` moves the estimate by exactly
``e * mean(y[post])`` -- which is the effect that was injected. The estimate
minus the truth is therefore

    tau_hat(e) - e * baseline = tau_0,

the backtest's own placebo ATT, for every effect size and on both the analytic
and the re-inject path. That identity is the first thing tested here: it is what
makes the criterion free, and if it ever stops holding the reported RMSE stops
meaning what it says.

Over the backtests of one (candidate, duration) the three reported numbers are
``bias = mean(tau_0)``, ``error_sd = std(tau_0)`` and ``rmse = sqrt(mean(tau_0^2))``,
with ``rmse^2 = bias^2 + error_sd^2``. The squaring is the whole difference from
``abs_lift_in_zero``, which the composite rank already carries: that term is
built after :func:`compute_power` has averaged over backtests, so it measures
bias and cancels error that alternates in sign. A design whose placebo ATT runs
+5, -5, +5, -5 is unbiased and unreliable, and only the RMSE says so.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOX
from mlsynth.config_models import GEOXConfig
from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.geox_helpers.aggregate import compute_accuracy, compute_power
from mlsynth.utils.geox_helpers.engines import resolve_engine
from mlsynth.utils.geox_helpers.orchestration import (
    PlanningReadout, planning_backtests, planning_readout)
from mlsynth.utils.geox_helpers.shaping import aggregate_treated, donor_matrix
from mlsynth.utils.geox_helpers.simulate import simulate_backtest

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"
ALPHA = 0.1
GRID = [-0.2, -0.1, 0.0, 0.1, 0.2]


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def wide(panel):
    return geoex_dataprep(panel, "location", "date", "Y")["Ywide"]


@pytest.fixture(scope="module")
def arrays(wide):
    """One candidate region's treated series, donors and backtest window."""
    region = frozenset(list(wide.columns)[:2])
    y = aggregate_treated(wide, region, how="mean").to_numpy()
    Y0 = donor_matrix(wide, region).to_numpy()
    n_pre = wide.shape[0] - 14
    return y, Y0, n_pre, n_pre, wide.shape[0] - 1, len(region)


# ---------------------------------------------------------------------------
# The identity the criterion rests on
# ---------------------------------------------------------------------------

class TestEstimationErrorIdentity:
    """``tau_hat(e) - e * baseline == tau_0``, exactly."""

    @pytest.mark.parametrize("engine,inference", [
        ("sdid", "placebo"),
        ("augsynth", "placebo"),
        ("augsynth", "conformal"),
    ])
    @pytest.mark.parametrize("analytic", [True, False])
    def test_error_equals_the_placebo_att(self, arrays, engine, inference,
                                          analytic):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine(engine)
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        tau0 = eng.att(fit, y, start, end)
        baseline = float(np.mean(y[start:end + 1]))

        swept = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, GRID,
                                   n_draws=15, n_tr=n_tr, seed=1,
                                   analytic=analytic, alpha=ALPHA,
                                   inference=inference)

        for es, tau in zip(GRID, swept["tau"]):
            error = tau - es * baseline
            assert error == pytest.approx(tau0, rel=1e-9, abs=1e-9), (
                f"{engine}/{inference} analytic={analytic}: error at e={es} is "
                f"{error:.6f}, placebo ATT is {tau0:.6f}")

    @pytest.mark.parametrize("engine", ["sdid", "augsynth"])
    def test_sweep_reports_the_error(self, arrays, engine):
        # The sweep already knows tau_0; it has to hand it back, because the
        # user-supplied effect grid need not contain zero for it to be read off.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine(engine)
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        swept = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, GRID,
                                   n_draws=15, n_tr=n_tr, seed=1, alpha=ALPHA)
        assert swept["tau0"] == pytest.approx(eng.att(fit, y, start, end))

    def test_backtest_rows_carry_the_error_and_sigma(self, wide):
        region = frozenset(list(wide.columns)[:2])
        y = aggregate_treated(wide, region, how="mean").to_numpy()
        Y0 = donor_matrix(wide, region).to_numpy()
        rows = simulate_backtest(y, Y0, wide.shape[0], 14, 1, GRID,
                                 n_draws=15, n_tr=len(region), seed=1,
                                 alpha=ALPHA)
        assert rows
        for row in rows:
            assert np.isfinite(row["estimation_error"])
            assert np.isfinite(row["placebo_sigma"])
        errors = {row["estimation_error"] for row in rows}
        assert len(errors) == 1, "the error does not depend on the effect size"

    def test_an_engine_that_omits_tau0_is_reported(self, wide, monkeypatch):
        # A silent nan would report the design as unmeasurable instead of
        # naming the engine that broke the protocol.
        import dataclasses

        import mlsynth.utils.geox_helpers.simulate as simulate_module

        eng = resolve_engine("sdid")

        def sweep_without_tau0(*args, **kwargs):
            swept = eng.sweep_p_values(*args, **kwargs)
            return {k: v for k, v in swept.items() if k != "tau0"}

        monkeypatch.setattr(
            simulate_module, "resolve_engine",
            lambda name: dataclasses.replace(
                eng, sweep_p_values=sweep_without_tau0))
        region = frozenset(list(wide.columns)[:2])
        y = aggregate_treated(wide, region, how="mean").to_numpy()
        Y0 = donor_matrix(wide, region).to_numpy()
        with pytest.raises(MlsynthEstimationError, match="tau0"):
            simulate_backtest(y, Y0, wide.shape[0], 14, 1, GRID,
                              n_draws=15, n_tr=len(region), seed=1, alpha=ALPHA)

    def test_error_is_the_effect_the_grid_omits(self, wide):
        # A grid without zero still reports the same error as one with it: the
        # error is read off the fit, not off a grid point.
        region = frozenset(list(wide.columns)[:2])
        y = aggregate_treated(wide, region, how="mean").to_numpy()
        Y0 = donor_matrix(wide, region).to_numpy()
        kw = dict(n_draws=15, n_tr=len(region), seed=1, alpha=ALPHA)
        with_zero = simulate_backtest(y, Y0, wide.shape[0], 14, 1,
                                      [-0.1, 0.0, 0.1], **kw)
        without = simulate_backtest(y, Y0, wide.shape[0], 14, 1,
                                    [0.3, 0.4], **kw)
        assert (without[0]["estimation_error"]
                == pytest.approx(with_zero[0]["estimation_error"]))


# ---------------------------------------------------------------------------
# The aggregation, on cubes built by hand
# ---------------------------------------------------------------------------

def _cube(errors_by_candidate, *, duration=14, effect_sizes=(0.0, 0.1),
          sigma=1.0):
    """A long cube with prescribed per-backtest errors.

    ``errors_by_candidate`` maps a candidate label to its per-backtest error
    sequence. Every other column is filled with a constant, so a test that
    reads one of them is reading something it set.
    """
    rows = []
    for candidate, errors in errors_by_candidate.items():
        for sim, error in enumerate(errors, start=1):
            for es in effect_sizes:
                rows.append({
                    "candidate": candidate, "duration": duration, "sim": sim,
                    "effect_size": float(es),
                    "estimation_error": float(error),
                    "placebo_sigma": sigma,
                    "p_value": 0.5,
                    "placebo_mean_effect": float(error) + float(es) * 100.0,
                    "detected_lift": (float(error) + float(es) * 100.0) / 100.0,
                    "scaled_l2": 0.5, "pre_rmspe": 1.0,
                    "pre_rmspe_lambda": 1.0, "investment": float("nan"),
                })
    return pd.DataFrame(rows)


class TestAccuracyAggregation:
    def test_rmse_decomposes_into_bias_and_spread(self):
        table = compute_accuracy(_cube({"A": [1.0, 3.0, 5.0, 7.0]}))
        row = table.iloc[0]
        assert row["bias"] == pytest.approx(4.0)
        assert row["error_sd"] == pytest.approx(np.std([1.0, 3.0, 5.0, 7.0]))
        assert row["rmse"] == pytest.approx(
            np.sqrt(np.mean(np.square([1.0, 3.0, 5.0, 7.0]))))
        assert row["rmse"] ** 2 == pytest.approx(
            row["bias"] ** 2 + row["error_sd"] ** 2)

    def test_alternating_error_separates_from_constant_error(self):
        # The claim the criterion exists to make. Two designs, the same error
        # magnitude every backtest: one keeps its sign, the other flips.
        cube = _cube({"steady": [5.0, 5.0, 5.0, 5.0],
                      "flipping": [5.0, -5.0, 5.0, -5.0]})
        table = compute_accuracy(cube).set_index("candidate")

        # Averaging first cannot tell them apart on bias ...
        assert abs(table.loc["flipping", "bias"]) < abs(table.loc["steady", "bias"])
        # ... and the rank's own recovery term is built on that average, so it
        # calls the unreliable design the better one.
        averaged = compute_power(cube, alpha=ALPHA).set_index("candidate")
        assert (abs(averaged.loc["flipping", "detected_lift"]).min()
                < abs(averaged.loc["steady", "detected_lift"]).min())
        # Squaring first says they are equally wrong, which they are.
        assert table.loc["flipping", "rmse"] == pytest.approx(
            table.loc["steady", "rmse"])

    def test_accuracy_does_not_depend_on_the_effect_grid(self):
        errors = {"A": [1.0, -2.0, 3.0]}
        coarse = compute_accuracy(_cube(errors, effect_sizes=(0.1,)))
        fine = compute_accuracy(
            _cube(errors, effect_sizes=(-0.2, -0.1, 0.0, 0.1, 0.2)))
        for column in ("bias", "error_sd", "rmse"):
            assert coarse[column].iloc[0] == pytest.approx(fine[column].iloc[0])

    def test_calibration_ratio_is_rmse_over_the_placebo_sigma(self):
        table = compute_accuracy(_cube({"A": [2.0, -2.0]}, sigma=4.0))
        row = table.iloc[0]
        assert row["sigma_mean"] == pytest.approx(4.0)
        assert row["calibration_ratio"] == pytest.approx(row["rmse"] / 4.0)

    def test_calibration_ratio_absent_when_the_engine_reports_no_sigma(self):
        # The conformal path permutes instead of drawing placebos, so there is
        # no sigma to calibrate against. Reported absent, not guessed.
        table = compute_accuracy(_cube({"A": [1.0, 2.0]}, sigma=float("nan")))
        assert np.isnan(table["sigma_mean"].iloc[0])
        assert np.isnan(table["calibration_ratio"].iloc[0])
        assert np.isfinite(table["rmse"].iloc[0])

    @pytest.mark.parametrize("errors,bias,sd,rmse", [
        ([4.0], 4.0, 0.0, 4.0),                    # one backtest: no spread
        ([0.0, 0.0], 0.0, 0.0, 0.0),               # a design that never errs
        ([-3.0, 3.0], 0.0, 3.0, 3.0),              # pure spread, no bias
    ])
    def test_degenerate_backtest_sets(self, errors, bias, sd, rmse):
        row = compute_accuracy(_cube({"A": errors})).iloc[0]
        assert row["bias"] == pytest.approx(bias)
        assert row["error_sd"] == pytest.approx(sd)
        assert row["rmse"] == pytest.approx(rmse)

    def test_empty_cube_returns_the_columns(self):
        table = compute_accuracy(pd.DataFrame())
        assert table.empty
        for column in ("candidate", "duration", "bias", "error_sd", "rmse",
                       "sigma_mean", "calibration_ratio"):
            assert column in table.columns

    def test_a_failed_backtest_is_reported_not_dropped(self):
        # A fit that did not converge leaves a nan error. Dropping it would
        # report the accuracy of the backtests that happened to work.
        table = compute_accuracy(_cube({"A": [1.0, float("nan"), 3.0]}))
        assert np.isnan(table["rmse"].iloc[0])
        assert np.isnan(table["bias"].iloc[0])

    def test_a_cube_without_the_error_column_raises(self):
        cube = _cube({"A": [1.0]}).drop(columns=["estimation_error"])
        with pytest.raises(MlsynthEstimationError, match="estimation_error"):
            compute_accuracy(cube)

    def test_a_cube_without_a_sigma_column_still_reports_the_error(self):
        # No standard error is a missing comparison, not a missing measurement.
        cube = _cube({"A": [1.0, 3.0]}).drop(columns=["placebo_sigma"])
        row = compute_accuracy(cube).iloc[0]
        assert row["rmse"] == pytest.approx(np.sqrt(5.0))
        assert np.isnan(row["sigma_mean"])
        assert np.isnan(row["calibration_ratio"])


# ---------------------------------------------------------------------------
# The design, end to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result(panel):
    return GEOX(GEOXConfig(
        df=panel, unitid="location", time="date", outcome="Y",
        treatment_size=2, durations=[14], effect_sizes=GRID,
        n_backtests=3, n_draws=25, n_validation_backtests=2,
        alpha=ALPHA, seed=0)).fit()


class TestShortlist:
    def test_shortlist_carries_the_accuracy_columns(self, result):
        for column in ("bias", "error_sd", "rmse", "sigma_mean",
                       "calibration_ratio"):
            assert column in result.power.columns, column

    def test_rmse_is_non_negative_and_finite(self, result):
        rmse = result.power["rmse"].dropna()
        assert not rmse.empty
        assert (rmse >= 0).all()
        assert np.isfinite(rmse).all()

    def test_rmse_dominates_its_components(self, result):
        s = result.power.dropna(subset=["rmse", "bias", "error_sd"])
        assert not s.empty
        assert (s["rmse"] >= s["bias"].abs() - 1e-9).all()
        assert (s["rmse"] >= s["error_sd"] - 1e-9).all()

    def test_decomposition_holds_on_real_backtests(self, result):
        s = result.power.dropna(subset=["rmse", "bias", "error_sd"])
        np.testing.assert_allclose(
            s["rmse"] ** 2, s["bias"] ** 2 + s["error_sd"] ** 2, rtol=1e-9)

    def test_candidate_designs_carry_the_accuracy(self, result):
        scored = [c for c in result.search.candidates if c.rmse is not None]
        assert scored
        for design in scored:
            assert np.isfinite(design.rmse) and design.rmse >= 0
            assert np.isfinite(design.bias)
            assert np.isfinite(design.error_sd)

    def test_the_ranking_is_untouched(self, panel, result):
        # The composite rank is what the GeoLift cross-validation pins. The
        # accuracy columns are reported beside it and take no part in it.
        again = GEOX(GEOXConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=GRID,
            n_backtests=3, n_draws=25, n_validation_backtests=0,
            alpha=ALPHA, seed=0)).fit()
        left = result.power[["candidate", "duration", "mde", "rank"]]
        right = again.power[["candidate", "duration", "mde", "rank"]]
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True), right.reset_index(drop=True))

    def test_calibration_ratio_is_the_error_over_the_placebo_scale(self, result):
        # The two scales are not the same measurement -- the placebo sigma is
        # drawn across donor markets, the error across backtest windows -- so
        # the ratio has no value it should sit at. What is pinned is what it
        # is: a positive, finite comparison of the one against the other.
        s = result.power.dropna(subset=["calibration_ratio"])
        assert not s.empty
        assert (s["calibration_ratio"] > 0).all()
        assert np.isfinite(s["calibration_ratio"]).all()
        np.testing.assert_allclose(s["calibration_ratio"],
                                   s["rmse"] / s["sigma_mean"], rtol=1e-9)


class TestPlanningAccuracy:
    def test_planning_rmse_is_reported_for_the_winner(self, result):
        assert "winner_rmse_planning" in result.metadata
        planning = result.metadata["winner_rmse_planning"]
        assert planning is not None
        assert planning >= 0 and np.isfinite(planning)
        assert result.search.winner.rmse_planning == pytest.approx(planning)

    def test_absent_when_no_validation_backtests(self, panel):
        res = GEOX(GEOXConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=GRID,
            n_backtests=3, n_draws=25, n_validation_backtests=0,
            alpha=ALPHA, seed=0)).fit()
        assert res.metadata["winner_rmse_planning"] is None
        assert res.search.winner.rmse_planning is None

    def test_planning_backtests_are_not_the_selecting_ones(self, result):
        # Held-back backtests sit deeper in history, so the winner's planning
        # RMSE is a different number from its in-search one except by accident.
        winner = result.search.winner
        assert winner.rmse is not None and winner.rmse_planning is not None
        assert winner.rmse_planning != pytest.approx(winner.rmse, rel=1e-12)


class TestPlanningReadout:
    """The two branches of the readout that a full design run does not reach."""

    @staticmethod
    def _config(panel, **overrides):
        settings = dict(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=GRID,
            n_backtests=3, n_draws=15, n_validation_backtests=2,
            alpha=ALPHA, seed=0)
        settings.update(overrides)
        return GEOXConfig(**settings)

    def test_a_panel_too_short_carries_no_held_back_backtests(self, panel, wide):
        # The deeper windows would run off the start of the panel. Reported as
        # nothing measured, which is what the caller turns into None.
        config = self._config(panel, n_backtests=wide.shape[0],
                              n_validation_backtests=4)
        region = frozenset(list(wide.columns)[:2])
        assert planning_backtests(wide, region, config).empty
        assert planning_readout(pd.DataFrame(), config) == PlanningReadout()

    def test_an_undetectable_winner_still_reports_its_error(self, panel):
        # Nothing detectable means no planning MDE. The error does not depend
        # on detectability, so it is still read -- at the duration the design
        # deploys at, which is the longest requested.
        config = self._config(panel, durations=[7, 14])
        undetectable = _cube({"A": [2.0, 4.0]}, duration=7)
        undetectable = pd.concat(
            [undetectable, _cube({"A": [1.0, 3.0]}, duration=14)],
            ignore_index=True)
        readout = planning_readout(undetectable, config)
        assert readout.mde is None
        assert readout.rmse == pytest.approx(np.sqrt(5.0))


class TestConformalEngine:
    @pytest.fixture(scope="class")
    def conformal(self, panel):
        return GEOX(GEOXConfig(
            df=panel, unitid="location", time="date", outcome="Y",
            treatment_size=2, durations=[14], effect_sizes=GRID,
            n_backtests=2, n_draws=15, n_validation_backtests=0,
            engine="augsynth", inference="conformal",
            alpha=ALPHA, seed=0)).fit()

    def test_rmse_is_reported_without_a_sigma(self, conformal):
        s = conformal.power
        assert np.isfinite(s["rmse"].dropna()).all()
        assert s["calibration_ratio"].isna().all()


def test_the_criterion_costs_no_extra_fits(wide, monkeypatch):
    # The identity is what makes the RMSE free. If a future change made the
    # error depend on the effect size, the sweep would have to refit per grid
    # point; this pins that it does not, by counting the fits behind one cube.
    import dataclasses

    import mlsynth.utils.geox_helpers.simulate as simulate_module

    eng = resolve_engine("sdid")
    calls = []

    def counting_fit_once(*args, **kwargs):
        calls.append(1)
        return eng.fit_once(*args, **kwargs)

    counting = dataclasses.replace(eng, fit_once=counting_fit_once)
    monkeypatch.setattr(simulate_module, "resolve_engine",
                        lambda name: counting)
    region = frozenset(list(wide.columns)[:2])
    y = aggregate_treated(wide, region, how="mean").to_numpy()
    Y0 = donor_matrix(wide, region).to_numpy()
    simulate_module.simulate_backtest(
        y, Y0, wide.shape[0], 14, 1, list(itertools.chain(GRID, [0.3, 0.4])),
        n_draws=15, n_tr=len(region), seed=1, alpha=ALPHA)
    assert len(calls) == 1, "one backtest fit, whatever the grid's length"
