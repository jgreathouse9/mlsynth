"""Tests for the Synthetic Business Cycle (SBC) estimator.

Covers:
    * SBCConfig validation (h, p, weights_mode).
    * prepare_sbc_inputs (data prep + h+p validation).
    * Hamilton filter primitives.
    * Trend forecast extrapolation.
    * Simplex and unrestricted weight solvers.
    * Orchestration (full pipeline + summarize_effects).
    * SBC estimator class smoke + edge cases.
    * Plotter (smoke).
    * Immutability.

Reference: Shi, Xi, Xie (2025), arXiv:2505.22388.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from mlsynth import SBC
from mlsynth.config_models import SBCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)

from mlsynth.utils.sbc_helpers.structures import (
    HamiltonFit,
    SBCDesign,
    SBCInputs,
    SBCResults,
)
from mlsynth.utils.sbc_helpers.setup import prepare_sbc_inputs
from mlsynth.utils.sbc_helpers.hamilton import (
    _build_lag_design,
    cycle_matrix_pre,
    fit_hamilton_filter,
)
from mlsynth.utils.sbc_helpers.trend_forecast import forecast_treated_trend
from mlsynth.utils.sbc_helpers.synthetic import solve_sbc_weights
from mlsynth.utils.sbc_helpers.orchestration import (
    solve_sbc,
    summarize_effects,
)
from mlsynth.utils.sbc_helpers.plotter import plot_sbc


# =========================================================================
# FIXTURES
# =========================================================================

def _make_panel(
    n_units=6,
    T=60,
    T0=45,
    drift=0.05,
    seed=0,
    treated_effect=0.0,
):
    """Synthetic nonstationary panel: random walks with drift + AR(1) factor.

    Matches the spirit of the paper's Models 1-2 (Section 4):
    independent random walks with drifts plus a common stationary factor.
    """
    rng = np.random.default_rng(seed)

    common = np.zeros(T)
    phi = 0.5
    for t in range(1, T):
        common[t] = phi * common[t - 1] + rng.standard_normal()

    Y = np.zeros((T, n_units))
    for i in range(n_units):
        drift_i = drift + 0.01 * rng.standard_normal()
        shocks = rng.standard_normal(T)
        rw = np.cumsum(drift_i + shocks)
        load = rng.standard_normal()
        Y[:, i] = rw + load * common

    if treated_effect != 0.0:
        Y[T0:, 0] += treated_effect

    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({
                "unitid": f"u{i:02d}",
                "time": t,
                "y": Y[t, i],
                "treat": int(i == 0 and t >= T0),
            })
    return pd.DataFrame(rows), Y


@pytest.fixture
def panel_df():
    df, _ = _make_panel()
    return df


@pytest.fixture
def panel_with_effect():
    df, _ = _make_panel(treated_effect=3.0)
    return df


@pytest.fixture
def inputs(panel_df):
    return prepare_sbc_inputs(
        df=panel_df, outcome="y", unitid="unitid", time="time",
        treat="treat", h=2, p=2,
    )


# =========================================================================
# CONFIG
# =========================================================================

class TestSBCConfig:

    def test_defaults(self, panel_df):
        cfg = SBCConfig(df=panel_df, outcome="y", unitid="unitid",
                         time="time", treat="treat")
        assert cfg.h == 2
        assert cfg.p == 2
        assert cfg.weights_mode == "simplex"
        assert cfg.display_graphs is False

    def test_h_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            SBCConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", treat="treat", h=0)

    def test_p_must_be_positive(self, panel_df):
        with pytest.raises(Exception):
            SBCConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", treat="treat", p=0)

    def test_invalid_weights_mode_rejected(self, panel_df):
        with pytest.raises(Exception):
            SBCConfig(df=panel_df, outcome="y", unitid="unitid",
                       time="time", treat="treat", weights_mode="gibberish")

    def test_invalid_dict_wraps(self, panel_df):
        with pytest.raises(MlsynthConfigError, match="Invalid SBC configuration"):
            SBC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                  "time": "time", "treat": "treat", "h": 0})


# =========================================================================
# DATA PREP
# =========================================================================

class TestPrepareSbcInputs:

    def test_basic_shapes(self, panel_df):
        inp = prepare_sbc_inputs(panel_df, "y", "unitid", "time", "treat",
                                   h=2, p=2)
        assert inp.Y_full.shape == (60, 6)
        assert inp.T == 60
        assert inp.T0 == 45
        assert inp.N == 6
        assert inp.treated_unit_name == "u00"

    def test_target_is_column_zero(self, panel_df, inputs):
        np.testing.assert_array_equal(inputs.Y_full[:, 0], inputs.y_target)

    def test_too_few_pre_periods_rejected(self):
        # 3 pre periods can't hold h=2 + p=2 = 4 lag observations.
        rows = []
        for i in range(3):
            for t in range(5):
                rows.append({
                    "unitid": f"u{i}", "time": t, "y": float(i + t),
                    "treat": int(i == 0 and t >= 3),
                })
        df = pd.DataFrame(rows)
        with pytest.raises(MlsynthDataError, match="at least h \\+ p"):
            prepare_sbc_inputs(df, "y", "unitid", "time", "treat", h=2, p=2)


# =========================================================================
# HAMILTON FILTER
# =========================================================================

class TestHamilton:

    def test_lag_design_shapes(self):
        y = np.arange(10, dtype=float)
        X, target, idx = _build_lag_design(y, h=2, p=2)
        # First valid t is h + p - 1 = 3, so T_eff = 10 - 3 = 7
        assert X.shape == (7, 3)
        assert target.shape == (7,)
        np.testing.assert_array_equal(X[:, 0], 1.0)
        np.testing.assert_array_equal(idx, np.arange(3, 10))

    def test_lag_design_rejects_short(self):
        with pytest.raises(MlsynthDataError):
            _build_lag_design(np.zeros(3), h=2, p=2)  # T < h+p

    def test_fit_basic(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(40)
        fit = fit_hamilton_filter(y, h=2, p=2)
        assert isinstance(fit, HamiltonFit)
        assert fit.coefficients.shape == (3,)   # intercept + p slopes
        assert fit.trend_pre.shape == (40,)
        assert fit.cycle_pre.shape == (40,)
        # First h + p - 1 = 3 entries should be NaN
        assert np.all(np.isnan(fit.trend_pre[:3]))
        assert np.all(np.isnan(fit.cycle_pre[:3]))
        # Remaining entries should be finite
        assert np.all(np.isfinite(fit.trend_pre[3:]))

    def test_fit_invalid_h_p(self):
        with pytest.raises(MlsynthEstimationError):
            fit_hamilton_filter(np.zeros(20), h=0, p=2)
        with pytest.raises(MlsynthEstimationError):
            fit_hamilton_filter(np.zeros(20), h=2, p=0)

    def test_fit_recovers_constant_trend(self):
        # Pure constant series -> intercept should approximate the mean,
        # slopes should be near zero, cycle near zero.
        y = np.full(40, 5.0)
        fit = fit_hamilton_filter(y, h=2, p=2)
        valid = ~np.isnan(fit.cycle_pre)
        np.testing.assert_allclose(fit.cycle_pre[valid], 0.0, atol=1e-6)
        # On a constant series the trend equals the series.
        np.testing.assert_allclose(fit.trend_pre[valid], 5.0, atol=1e-6)

    def test_cycle_matrix_pre_shape(self, inputs):
        fits, cycles = cycle_matrix_pre(
            inputs.Y_full, T0=inputs.T0, h=2, p=2,
        )
        assert len(fits) == inputs.N
        assert cycles.shape == (inputs.T0, inputs.N)
        # First h + p - 1 = 3 rows should be NaN
        assert np.all(np.isnan(cycles[:3, :]))
        # The remaining rows should be finite for every column.
        assert np.all(np.isfinite(cycles[3:, :]))


# =========================================================================
# TREND FORECAST
# =========================================================================

class TestTrendForecast:

    def test_forecast_zero_horizon(self, inputs):
        fit = fit_hamilton_filter(inputs.y_target[:inputs.T0], h=2, p=2)
        out = forecast_treated_trend(inputs.y_target, fit,
                                      T0=inputs.T0, horizon=0)
        assert out.shape == (0,)

    def test_forecast_shape(self, inputs):
        fit = fit_hamilton_filter(inputs.y_target[:inputs.T0], h=2, p=2)
        horizon = inputs.T - inputs.T0
        out = forecast_treated_trend(inputs.y_target, fit,
                                      T0=inputs.T0, horizon=horizon)
        assert out.shape == (horizon,)
        assert np.all(np.isfinite(out))

    def test_forecast_includes_intercept(self):
        # The forecast extrapolates the full estimated trend, intercept
        # included (matching the authors' replication code). With slopes
        # at zero and a large intercept, the forecast equals the intercept.
        fake_fit = HamiltonFit(
            coefficients=np.array([1000.0, 0.0, 0.0]),
            trend_pre=np.array([np.nan, np.nan, np.nan, 0.0]),
            cycle_pre=np.array([np.nan, np.nan, np.nan, 0.0]),
            h=2, p=2,
        )
        y_target = np.zeros(10)
        out = forecast_treated_trend(y_target, fake_fit, T0=5, horizon=3)
        np.testing.assert_allclose(out, 1000.0, atol=1e-9)

    def test_forecast_intercept_plus_slopes(self):
        # alpha0 + alpha1 * y[t-h] + alpha2 * y[t-h-1].
        fake_fit = HamiltonFit(
            coefficients=np.array([5.0, 0.5, 0.25]),
            trend_pre=np.array([np.nan, np.nan, np.nan, 0.0]),
            cycle_pre=np.array([np.nan, np.nan, np.nan, 0.0]),
            h=2, p=2,
        )
        y_target = np.arange(10, dtype=float)  # y[t] = t
        out = forecast_treated_trend(y_target, fake_fit, T0=5, horizon=1)
        # step 0: t=5, lags y[3], y[2] -> 5 + 0.5*3 + 0.25*2 = 7.0
        np.testing.assert_allclose(out, [7.0], atol=1e-9)


# =========================================================================
# SCM-ON-CYCLES
# =========================================================================

class TestSolveSBCWeights:

    def test_simplex_basic(self):
        rng = np.random.default_rng(0)
        ct = rng.standard_normal(20)
        cd = rng.standard_normal((20, 4))
        w, b = solve_sbc_weights(ct, cd, weights_mode="simplex")
        assert w.shape == (4,)
        assert b is None
        assert np.all(w >= -1e-6)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_simplex_recovers_when_target_is_donor(self):
        rng = np.random.default_rng(0)
        cd = rng.standard_normal((30, 3))
        ct = cd[:, 1].copy()  # target == donor 1
        w, _ = solve_sbc_weights(ct, cd, weights_mode="simplex")
        # Weight should concentrate on donor 1
        assert w[1] > 0.9

    def test_unrestricted_basic(self):
        rng = np.random.default_rng(0)
        ct = rng.standard_normal(20)
        cd = rng.standard_normal((20, 4))
        w, b = solve_sbc_weights(ct, cd, weights_mode="unrestricted")
        assert w.shape == (4,)
        assert isinstance(b, float)

    def test_simplex_matches_cvxpy_reference(self):
        # The simplex solve now runs through the in-house FISTA QP
        # (bilevel.simplex.simplex_lstsq) rather than cvxpy. Guard against
        # drift by checking it against a direct cvxpy solve of the same
        # program: min ||ct - cd w||^2 s.t. w >= 0, sum(w) = 1.
        import cvxpy as cp

        rng = np.random.default_rng(7)
        ct = rng.standard_normal(40)
        cd = rng.standard_normal((40, 5))
        w, _ = solve_sbc_weights(ct, cd, weights_mode="simplex")

        wv = cp.Variable(5, nonneg=True)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(ct - cd @ wv)), [cp.sum(wv) == 1]
        )
        prob.solve(solver=cp.CLARABEL)
        w_ref = np.asarray(wv.value, dtype=float)

        # Objective values should agree even if the argmin is non-unique.
        obj = float(np.sum((ct - cd @ w) ** 2))
        obj_ref = float(np.sum((ct - cd @ w_ref) ** 2))
        assert abs(obj - obj_ref) < 1e-4
        np.testing.assert_allclose(w, w_ref, atol=1e-3)

    def test_simplex_single_donor(self):
        # Degenerate one-donor case: the only feasible weight is 1.0.
        ct = np.arange(10, dtype=float)
        cd = (2.0 * ct).reshape(-1, 1)
        w, b = solve_sbc_weights(ct, cd, weights_mode="simplex")
        assert w.shape == (1,)
        np.testing.assert_allclose(w, [1.0])
        assert b is None

    def test_unknown_mode_rejected(self):
        with pytest.raises(MlsynthEstimationError):
            solve_sbc_weights(np.zeros(5), np.zeros((5, 2)),
                              weights_mode="zigzag")

    def test_shape_mismatch_rejected(self):
        with pytest.raises(MlsynthEstimationError):
            solve_sbc_weights(np.zeros(5), np.zeros((10, 2)),
                              weights_mode="simplex")


# =========================================================================
# ORCHESTRATION
# =========================================================================

class TestSolveSbc:

    def test_runs_end_to_end(self, inputs):
        design = solve_sbc(inputs, h=2, p=2, weights_mode="simplex")
        assert isinstance(design, SBCDesign)
        assert design.weights.shape == (inputs.N - 1,)
        # Counterfactual is capped at the h-step forecast horizon.
        hz = min(2, inputs.T - inputs.T0)
        assert design.trend_forecast.shape == (hz,)
        assert design.cycle_forecast.shape == (hz,)
        assert design.counterfactual_post.shape == (hz,)
        assert design.pre_cycle_rmse >= 0
        # simplex => intercept is None
        assert design.intercept is None

    def test_runs_unrestricted(self, inputs):
        design = solve_sbc(inputs, h=2, p=2, weights_mode="unrestricted")
        assert isinstance(design.intercept, float)

    def test_no_post_period(self):
        # Build a panel with all "treated" so post window is empty.
        rows = []
        T = 30
        for i in range(4):
            for t in range(T):
                rows.append({
                    "unitid": f"u{i}", "time": t, "y": float(i + t * 0.1),
                    # Mark only unit 0 as treated, starting at the very end.
                    "treat": int(i == 0 and t >= T - 1),
                })
        df = pd.DataFrame(rows)
        inputs = prepare_sbc_inputs(df, "y", "unitid", "time", "treat",
                                      h=2, p=2)
        # Note: T0 = T - 1 = 29, post = 1 -> still has 1 post period.
        # Build a no-post case explicitly via SBCInputs:
        no_post_inputs = SBCInputs(
            Y_full=inputs.Y_full, T=inputs.T, T0=inputs.T,
            N=inputs.N, treated_unit_name=inputs.treated_unit_name,
            donor_names=inputs.donor_names, time_labels=inputs.time_labels,
            Ywide=inputs.Ywide, y_target=inputs.y_target,
        )
        design = solve_sbc(no_post_inputs, h=2, p=2, weights_mode="simplex")
        assert design.counterfactual_post.shape == (0,)
        assert design.trend_forecast.shape == (0,)

    def test_summarize_effects(self, inputs):
        design = solve_sbc(inputs, h=2, p=2, weights_mode="simplex")
        att, cf, te = summarize_effects(inputs, design)
        assert isinstance(att, float)
        assert cf.shape == (inputs.T,)
        assert te.shape == (inputs.T,)
        # Pre-period treatment effect should be exactly zero.
        np.testing.assert_array_equal(te[:inputs.T0], 0.0)
        # Counterfactual matches the observed treated series pre-treatment.
        np.testing.assert_array_equal(cf[:inputs.T0], inputs.y_target[:inputs.T0])

    def test_summarize_no_post_returns_nan(self):
        Y = np.zeros((10, 3))
        inputs = SBCInputs(
            Y_full=Y, T=10, T0=10, N=3,
            treated_unit_name="x", donor_names=["a", "b"],
            time_labels=np.arange(10), Ywide=None, y_target=Y[:, 0],
        )
        # A minimal design with empty post arrays.
        design = SBCDesign(
            weights=np.array([0.5, 0.5]), weights_mode="simplex",
            intercept=None,
            treated_hamilton=HamiltonFit(
                coefficients=np.zeros(3),
                trend_pre=np.zeros(10), cycle_pre=np.zeros(10),
                h=2, p=2,
            ),
            donor_hamiltons=[],
            trend_forecast=np.empty(0),
            cycle_forecast=np.empty(0),
            counterfactual_post=np.empty(0),
            pre_cycle_rmse=0.0,
        )
        att, cf, te = summarize_effects(inputs, design)
        assert np.isnan(att)


# =========================================================================
# SBC CLASS (public API)
# =========================================================================

class TestSBCClass:

    def test_fit_smoke(self, panel_df):
        res = SBC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "h": 2, "p": 2,
        }).fit()
        assert isinstance(res, SBCResults)
        assert isinstance(res.att, float)
        assert res.design.weights.shape[0] == 5

    def test_simplex_weights_sum_to_one(self, panel_df):
        res = SBC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "h": 2, "p": 2,
            "weights_mode": "simplex",
        }).fit()
        assert abs(sum(res.weights_by_donor.values()) - 1.0) < 1e-4

    def test_fit_unrestricted_mode(self, panel_df):
        res = SBC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "h": 2, "p": 2,
            "weights_mode": "unrestricted",
        }).fit()
        # Unrestricted mode allows negative weights and an intercept.
        assert isinstance(res.design.intercept, float)

    def test_pre_period_treatment_effect_is_zero(self, panel_df):
        res = SBC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "h": 2, "p": 2,
        }).fit()
        np.testing.assert_array_equal(
            res.treatment_effect[: res.inputs.T0], 0.0
        )

    def test_recovers_effect_sign(self, panel_with_effect):
        res = SBC({
            "df": panel_with_effect, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "h": 2, "p": 2,
        }).fit()
        # Injected +3 effect on the treated post window — ATT should be > 0.
        assert res.att > 0

    def test_unexpected_error_wrapped(self, monkeypatch, panel_df):
        def boom(*args, **kwargs):
            raise RuntimeError("kaboom")
        monkeypatch.setattr("mlsynth.estimators.sbc.solve_sbc", boom)
        with pytest.raises(MlsynthEstimationError):
            SBC({"df": panel_df, "outcome": "y", "unitid": "unitid",
                  "time": "time", "treat": "treat"}).fit()


# =========================================================================
# PLOTTER (smoke)
# =========================================================================

class TestPlotter:

    @pytest.fixture(autouse=True)
    def _matplotlib_agg(self, monkeypatch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)

    def test_plot_runs(self, panel_df):
        res = SBC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "h": 2, "p": 2,
            "display_graphs": False,
        }).fit()
        plot_sbc(res)


# =========================================================================
# IMMUTABILITY
# =========================================================================

class TestImmutability:

    def test_inputs_frozen(self, inputs):
        with pytest.raises(FrozenInstanceError):
            inputs.T0 = 99   # type: ignore[misc]

    def test_design_frozen(self, inputs):
        design = solve_sbc(inputs, h=2, p=2, weights_mode="simplex")
        with pytest.raises(FrozenInstanceError):
            design.weights = np.zeros(5)   # type: ignore[misc]

    def test_results_frozen(self, panel_df):
        res = SBC({
            "df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat",
        }).fit()
        # SBCResults is now a frozen pydantic EffectResult; field assignment
        # raises (and `att` is a read-only contract property).
        with pytest.raises(Exception):
            res.weights_by_donor = {}   # type: ignore[misc]

    def test_hamilton_fit_frozen(self):
        fit = fit_hamilton_filter(np.arange(20, dtype=float), h=2, p=2)
        with pytest.raises(FrozenInstanceError):
            fit.h = 99   # type: ignore[misc]
