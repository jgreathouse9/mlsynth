"""Augmented SCM as an SDIDGEO scoring engine.

The seam exists so the estimator that scores candidates can be swapped without
the pipeline noticing. This is the second engine: augsynth (Ben-Michael, Feller
& Rothstein 2021) fitted through :class:`~mlsynth.utils.bilevel.engine.BilevelSCM`,
tested by the conformal permutation of Chernozhukov, Wuthrich & Zhu -- which is
what GeoLift itself uses.

Engine and inference are not one axis. Placebo reassignment needs only a
counterfactual built from donors, so it applies to either engine. Conformal
needs the pre-period residuals to be exchangeable across the matching window,
and synthetic DiD's time weights exist precisely to say pre-periods are not
interchangeable -- so conformal is available on augsynth alone, and the config
has to refuse the invalid pairing instead of silently producing a number.

The two procedures also differ in what the effect sweep can hoist. A placebo
standard error does not depend on the injected effect and is drawn once for the
grid; a conformal p-value re-permutes against the treated series and is
recomputed per effect size. Both paths are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.config_models import SDIDGEOConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.sdidgeo_helpers.engines import EngineFit, resolve_engine
from mlsynth.utils.sdidgeo_helpers.shaping import aggregate_treated, donor_matrix

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def wide(panel):
    from mlsynth.utils.datautils import geoex_dataprep
    return geoex_dataprep(panel, "location", "date", "Y")["Ywide"]


@pytest.fixture(scope="module")
def arrays(wide):
    region = frozenset(list(wide.columns)[:2])
    y = aggregate_treated(wide, region, how="mean").to_numpy()
    Y0 = donor_matrix(wide, region).to_numpy()
    n_pre = wide.shape[0] - 14
    return y, Y0, n_pre, n_pre, wide.shape[0] - 1, len(region)


# ---------------------------------------------------------------------------
# Resolution and the EngineFit contract
# ---------------------------------------------------------------------------

class TestResolution:
    def test_resolves(self):
        eng = resolve_engine("augsynth")
        assert eng.name == "augsynth"
        for fn in ("fit_once", "att", "sweep_p_values", "point_inference"):
            assert callable(getattr(eng, fn))

    def test_config_accepts_augsynth(self, panel):
        cfg = SDIDGEOConfig(df=panel, unitid="location", time="date",
                            outcome="Y", treatment_size=2, durations=[14],
                            effect_sizes=[0.0, 0.2], n_backtests=2, n_draws=5,
                            seed=0, engine="augsynth")
        assert cfg.engine == "augsynth"


class TestEngineFitContract:
    @pytest.fixture(scope="class")
    def fit(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        return resolve_engine("augsynth").fit_once(y, Y0, n_pre, start, end, n_tr)

    def test_shapes(self, fit, arrays):
        y, Y0, *_ = arrays
        assert isinstance(fit, EngineFit)
        assert fit.counterfactual.shape == y.shape
        assert fit.donor_weights.shape == (Y0.shape[1],)
        assert np.isfinite(fit.pre_rmspe) and np.isfinite(fit.scaled_l2)

    def test_has_no_time_weights(self, fit):
        # augsynth weights donors only; the slot stays empty so the shared
        # pipeline can tell the two engines apart without asking which it has.
        assert fit.time_weights is None

    def test_extras_carry_the_augsynth_specifics(self, fit):
        assert "intercept" in fit.extras
        assert "lambda_" in fit.extras and "augment" in fit.extras

    def test_counterfactual_is_intercept_plus_weighted_donors(self, fit, arrays):
        _, Y0, *_ = arrays
        expected = fit.extras["intercept"] + Y0 @ fit.donor_weights
        np.testing.assert_allclose(fit.counterfactual, expected, atol=1e-9)

    def test_att_is_the_mean_window_gap(self, fit, arrays):
        y, _, _, start, end, _ = arrays
        eng = resolve_engine("augsynth")
        expected = float(np.mean(y[start:end + 1]
                                 - fit.counterfactual[start:end + 1]))
        assert eng.att(fit, y, start, end) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# It must match the old GeoLift fit exactly
# ---------------------------------------------------------------------------

class TestMatchesTheGeoLiftFit:
    """The engine is an adapter, so it must add no arithmetic of its own."""

    def test_matches_a_direct_bilevel_fit(self, arrays):
        from mlsynth.utils.bilevel.engine import BilevelSCM
        y, Y0, n_pre, start, end, n_tr = arrays
        # GeoLift's default: ridge augmentation with unit fixed effects, so the
        # fit matches shapes and not levels.
        A, B = y[:n_pre] - y[:n_pre].mean(), Y0[:n_pre] - Y0[:n_pre].mean(axis=0)
        res = BilevelSCM(augment="ridge").fit(A, B)
        fit = resolve_engine("augsynth").fit_once(y, Y0, n_pre, start, end, n_tr)
        np.testing.assert_allclose(fit.donor_weights, np.asarray(res.W),
                                   atol=1e-10)

    def test_intercept_aligns_the_pre_period_means(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        fit = resolve_engine("augsynth").fit_once(y, Y0, n_pre, start, end, n_tr)
        expected = float(np.mean(y[:n_pre] - Y0[:n_pre] @ fit.donor_weights))
        assert fit.extras["intercept"] == pytest.approx(expected, abs=1e-9)

    def test_scaled_l2_is_the_augsynth_ratio(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        fit = resolve_engine("augsynth").fit_once(y, Y0, n_pre, start, end, n_tr)
        A = y[:n_pre] - y[:n_pre].mean()
        B = Y0[:n_pre] - Y0[:n_pre].mean(axis=0)
        w_unif = np.full(B.shape[1], 1.0 / B.shape[1])
        expected = (np.linalg.norm(B @ fit.donor_weights - A)
                    / np.linalg.norm(B @ w_unif - A))
        assert fit.scaled_l2 == pytest.approx(float(expected), rel=1e-9)

    def test_no_fixed_effects_leaves_a_zero_intercept(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        fit = resolve_engine("augsynth").fit_once(
            y, Y0, n_pre, start, end, n_tr, fixed_effects=False)
        assert fit.extras["intercept"] == 0.0


# ---------------------------------------------------------------------------
# Conformal inference, and what the sweep can and cannot hoist
# ---------------------------------------------------------------------------

class TestConformalSweep:
    def test_sweep_returns_one_p_value_per_effect(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        grid = [-0.2, 0.0, 0.2]
        out = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, grid,
                                 inference="conformal", ns=50, seed=0, n_tr=n_tr)
        assert len(out["p_value"]) == len(grid) == len(out["tau"])
        assert all(0.0 <= p <= 1.0 for p in out["p_value"])

    def test_conformal_p_values_differ_across_the_grid(self, arrays):
        # The defect a same-call-site substitution would have caused: one
        # p-value reused for every effect size. A conformal test re-permutes
        # against the injected series, so the grid must move.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        out = eng.sweep_p_values(fit, y, Y0, n_pre, start, end,
                                 [-0.3, 0.0, 0.3], inference="conformal",
                                 ns=50, seed=0, n_tr=n_tr)
        assert len(set(out["p_value"])) > 1

    def test_null_effect_is_least_significant(self, arrays):
        # A monotonicity the design's whole MDE logic rests on: a bigger
        # injected lift must not be harder to detect than no lift at all.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        out = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, [0.0, 0.5],
                                 inference="conformal", ns=100, seed=0, n_tr=n_tr)
        assert out["p_value"][1] <= out["p_value"][0]

    def test_reproducible_at_a_fixed_seed(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        kw = dict(inference="conformal", ns=40, seed=3, n_tr=n_tr)
        a = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, [0.0, 0.2], **kw)
        b = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, [0.0, 0.2], **kw)
        assert a["p_value"] == b["p_value"]

    def test_placebo_is_available_on_augsynth_too(self, arrays):
        # Placebo needs only a donor-built counterfactual, so it is not tied to
        # the engine. This is the combination that isolates the objective from
        # the inference.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        out = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, [0.0, 0.2],
                                 inference="placebo", n_draws=10, seed=0,
                                 n_tr=n_tr)
        assert all(np.isfinite(p) for p in out["p_value"])

    def test_point_inference_reports_its_method(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        p, details = eng.point_inference(fit, y, Y0, n_pre, start, end,
                                         inference="conformal", ns=50, seed=0,
                                         n_tr=n_tr)
        assert 0.0 <= p <= 1.0
        assert details["method"] == "conformal"


# ---------------------------------------------------------------------------
# The config must refuse the invalid pairing
# ---------------------------------------------------------------------------

class TestInferenceValidation:
    def _cfg(self, panel, **kw):
        base = dict(df=panel, unitid="location", time="date", outcome="Y",
                    treatment_size=2, durations=[14], effect_sizes=[0.0, 0.2],
                    n_backtests=2, n_draws=5, seed=0)
        base.update(kw)
        return SDIDGEOConfig(**base)

    def test_sdid_defaults_to_placebo(self, panel):
        assert self._cfg(panel).inference == "placebo"

    def test_augsynth_defaults_to_conformal(self, panel):
        # GeoLift's own choice, so the augsynth path reproduces it by default.
        assert self._cfg(panel, engine="augsynth").inference == "conformal"

    def test_sdid_with_conformal_is_rejected(self, panel):
        with pytest.raises(MlsynthConfigError, match="conformal"):
            self._cfg(panel, engine="sdid", inference="conformal")

    def test_augsynth_with_placebo_is_allowed(self, panel):
        assert self._cfg(panel, engine="augsynth",
                         inference="placebo").inference == "placebo"

    def test_unknown_inference_is_rejected(self, panel):
        with pytest.raises(MlsynthConfigError, match="inference"):
            self._cfg(panel, inference="bayes")

    def test_augsynth_only_knobs_rejected_on_sdid(self, panel):
        with pytest.raises(MlsynthConfigError, match="augsynth"):
            self._cfg(panel, engine="sdid", fixed_effects=False)


# ---------------------------------------------------------------------------
# End to end through the pipeline
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_augsynth_design_runs_and_selects(self, panel):
        import warnings
        from mlsynth import SDIDGEO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=2, durations=[14], effect_sizes=[0.0, 0.1, 0.2],
                n_backtests=2, n_draws=8, ns=50, n_validation_backtests=0,
                engine="augsynth", seed=0)).fit()
        assert res.selected_units is not None
        assert len(res.selected_units) == 2
        assert not res.power.empty

    def test_augsynth_weights_have_no_time_vector(self, panel):
        import warnings
        from mlsynth import SDIDGEO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=2, durations=[14], effect_sizes=[0.0, 0.2],
                n_backtests=2, n_draws=8, ns=40, n_validation_backtests=0,
                engine="augsynth", seed=0)).fit()
        assert res.design_weights.donor_weights
        assert not res.design_weights.time_weights


class TestPlanningMDEUsesTheChosenEngine:
    """The winner's re-scoring has to run on the engine the search ran on.

    ``planning_mde`` re-scores the winning region on validation backtests held
    out of the selection, and it is a separate code path from the scoring loop:
    it builds its own settings and calls ``simulate_backtest`` directly. Nothing
    in the engine tests reached it, because they all set
    ``n_validation_backtests=0`` -- which is how a NameError in that path
    survived a green run of every engine test file and only failed in CI.

    The planning MDE has to be comparable with the optimistic MDE it corrects,
    so it must be produced by the same engine and the same null.
    """

    def _fit(self, panel, **kw):
        import warnings
        from mlsynth import SDIDGEO
        base = dict(df=panel, unitid="location", time="date", outcome="Y",
                    treatment_size=2, durations=[7],
                    effect_sizes=[-0.2, 0.0, 0.2], n_backtests=2,
                    n_draws=8, n_validation_backtests=2, seed=0)
        base.update(kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SDIDGEO(SDIDGEOConfig(**base)).fit()

    @pytest.mark.parametrize("engine,extra", [
        ("sdid", {}),
        ("augsynth", dict(ns=40, augment="ridge", fixed_effects=True)),
    ])
    def test_validation_backtests_run_on_the_selected_engine(self, panel,
                                                             engine, extra):
        res = self._fit(panel, engine=engine, **extra)
        assert res.selected_units is not None
        # metadata carries both numbers, and the planning one is only present
        # when the validation backtests actually ran.
        assert "winner_mde_planning" in res.metadata
        assert "winner_mde_optimistic" in res.metadata

    def test_augsynth_planning_mde_is_finite_or_absent(self, panel):
        res = self._fit(panel, engine="augsynth", ns=40, augment="ridge",
                        fixed_effects=True)
        planning = res.metadata["winner_mde_planning"]
        assert planning is None or np.isfinite(planning)
