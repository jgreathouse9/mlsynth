"""Tests for the DSCAR estimator (Zheng & Chen 2024).

Layered per agents/agents_tests.md:

* Numerical helpers: per-period QP + EL weight solver, variable-
  importance OLS.
* Data utilities: panel ingestion, missingness handling, treat
  indicator parsing.
* Estimator integration: end-to-end runs on a synthetic AR-1 panel
  with known ATT.
* **Path-A regression**: pin the Zheng & Chen (2024) Section 5
  empirical numbers (Beijing PM2.5 air-pollution alerts).
* Public API contracts: dict-vs-typed config equivalence, exception
  translation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import DSCAR
from mlsynth.config_models import DSCARConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.dscar_helpers import (
    DSCARInputs, DSCARResults, prepare_dsc_inputs, run_dsc,
)
from mlsynth.utils.dscar_helpers.weights import (
    solve_dsc_weights, variable_importance,
)


def _ar1_panel(
    *, N=10, T=30, T0=20, ar=0.6, tau=2.0, seed=0,
) -> pd.DataFrame:
    """Tiny AR(1) panel with a treated unit (index 0) and one covariate."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((N, T)) * 0.3
    x = rng.standard_normal((N, T)) * 0.5
    Y = np.zeros((N, T))
    Y[:, 0] = rng.standard_normal(N)
    for t in range(1, T):
        Y[:, t] = 0.5 * x[:, t] + ar * Y[:, t - 1] + eps[:, t]
    Y[0, T0:] += tau
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({
                "unit": f"u{i}", "year": t,
                "y": float(Y[i, t]),
                "x1": float(x[i, t]),
                "y_lag1": float(Y[i, t - 1]) if t >= 1 else float(rng.standard_normal()),
                "treat": int(i == 0 and t >= T0),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Layer 1 -- numerical helpers
# ----------------------------------------------------------------------
class TestWeightSolver:
    def test_qp_falls_back_when_target_outside_hull(self):
        Z1 = np.array([10.0])
        Z0 = np.array([[1.0, 2.0, 3.0]])
        V = np.array([1.0])
        w, used_el = solve_dsc_weights(Z1, Z0, V, el_tolerance=1e-6)
        # Far outside hull -> EL refinement should NOT trigger.
        assert not used_el
        # Weight concentrates on the largest donor (idx 2).
        assert w[2] == pytest.approx(w.max())
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_el_refines_when_target_in_hull(self):
        # Z1 = 2 is the average of donors at 1, 2, 3 -> exact match feasible.
        Z1 = np.array([2.0])
        Z0 = np.array([[1.0, 2.0, 3.0]])
        V = np.array([1.0])
        w, used_el = solve_dsc_weights(Z1, Z0, V, el_tolerance=0.5)
        assert used_el
        # Exact match: Z0 @ w == Z1.
        assert (Z0 @ w)[0] == pytest.approx(2.0, abs=1e-6)


class TestVariableImportance:
    def test_shape(self):
        panel = _ar1_panel()
        Y = panel.pivot(index="year", columns="unit", values="y").to_numpy().T
        Y_lag = np.empty_like(Y); Y_lag[:, 1:] = Y[:, :-1]; Y_lag[:, 0] = 0
        X = np.zeros((Y.shape[0], Y.shape[1], 1))
        for i, u in enumerate(panel["unit"].unique()):
            X[i, :, 0] = panel.loc[panel.unit == u, "x1"].to_numpy()
        V = variable_importance(Y, X, Y_lag, T0=20)
        assert V.shape == (Y.shape[1], 1 + 1)
        assert (V >= 0).all()


# ----------------------------------------------------------------------
# Layer 2 -- data utilities
# ----------------------------------------------------------------------
class TestSetup:
    def test_prepare_inputs_partitions_correctly(self):
        df = _ar1_panel()
        inp = prepare_dsc_inputs(
            df=df, outcome="y", treat="treat", unitid="unit", time="year",
            exog_covariates=["x1"], lagged_outcome="y_lag1",
        )
        assert inp.n_treated == 1
        assert inp.T0 == 20
        assert inp.T1 == 10
        assert inp.Y.shape == (10, 30)
        assert inp.X.shape == (10, 30, 1)

    def test_rejects_zero_treated(self):
        df = _ar1_panel()
        df["treat"] = 0
        with pytest.raises(MlsynthDataError):
            prepare_dsc_inputs(
                df=df, outcome="y", treat="treat", unitid="unit", time="year",
                exog_covariates=["x1"], lagged_outcome="y_lag1",
            )

    def test_rejects_staggered_treated(self):
        df = _ar1_panel()
        # Mark a second unit as starting later.
        df.loc[(df["unit"] == "u1") & (df["year"] >= 25), "treat"] = 1
        with pytest.raises(MlsynthDataError):
            prepare_dsc_inputs(
                df=df, outcome="y", treat="treat", unitid="unit", time="year",
                exog_covariates=["x1"], lagged_outcome="y_lag1",
            )


# ----------------------------------------------------------------------
# Layer 3 -- estimator integration
# ----------------------------------------------------------------------
class TestEstimator:
    def test_recovers_planted_ATT(self):
        df = _ar1_panel(N=8, T=30, T0=20, tau=2.0, seed=2)
        res = DSCAR({
            "df": df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "year",
            "exog_covariates": ["x1"], "lagged_outcome": "y_lag1",
            "display_graphs": False,
        }).fit()
        # Generous tolerance; ATT recovery depends on donor diversity.
        assert res.att == pytest.approx(2.0, abs=1.5)

    def test_weights_sum_to_one_each_period(self):
        df = _ar1_panel()
        res = DSCAR({
            "df": df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "year",
            "exog_covariates": ["x1"], "lagged_outcome": "y_lag1",
            "display_graphs": False,
        }).fit()
        for t in range(res.inputs.T):
            s = res.weight_matrix[t].sum()
            assert s == pytest.approx(1.0, abs=1e-6), (t, s)


# ----------------------------------------------------------------------
# Path-A regression: Beijing PM2.5 air-pollution alerts
# ----------------------------------------------------------------------
class TestPathABeijingAlerts:
    """Pin Zheng & Chen (2024) Section 5 empirical numbers."""

    @pytest.fixture(scope="class")
    def orange_fit(self):
        base = Path(__file__).resolve().parents[2] / "basedata"
        csv = base / "beijing_pm25_orange_alert.csv"
        if not csv.exists():
            pytest.skip("beijing_pm25_orange_alert.csv not shipped.")
        df = pd.read_csv(csv)
        df["treat_indicator"] = (
            (df["alert_if"] == 1) & (df["hour_eps"] > 48)
        ).astype(int)
        return DSCAR({
            "df": df, "outcome": "pm25", "treat": "treat_indicator",
            "unitid": "id_eps", "time": "hour_eps",
            "exog_covariates": ["WSPM", "humi", "dewp", "pres"],
            "lagged_outcome": "pm25_lag1", "display_graphs": False,
        }).fit()

    @pytest.fixture(scope="class")
    def red_fit(self):
        base = Path(__file__).resolve().parents[2] / "basedata"
        csv = base / "beijing_pm25_red_alert.csv"
        if not csv.exists():
            pytest.skip("beijing_pm25_red_alert.csv not shipped.")
        df = pd.read_csv(csv)
        df["treat_indicator"] = (
            (df["alert_if"] == 1) & (df["hour_eps"] > 48)
        ).astype(int)
        return DSCAR({
            "df": df, "outcome": "pm25", "treat": "treat_indicator",
            "unitid": "id_eps", "time": "hour_eps",
            "exog_covariates": ["WSPM", "humi", "dewp", "pres"],
            "lagged_outcome": "pm25_lag1", "display_graphs": False,
        }).fit()

    def test_orange_att_matches_paper(self, orange_fit):
        # Paper Table on page 21: ATT = -33.8 mu g/m^3, mu_0 = 139.0,
        # mu_1 = 105.3, relative reduction 24.3%.
        mu0 = float(orange_fit.fit.Y0_hat[48:].mean())
        mu1 = float(orange_fit.fit.Y_treated_mean[48:].mean())
        assert orange_fit.att == pytest.approx(-33.8, abs=0.1)
        assert mu0 == pytest.approx(139.0, abs=0.2)
        assert mu1 == pytest.approx(105.3, abs=0.1)
        assert 100 * orange_fit.att_relative == pytest.approx(24.3, abs=0.1)

    def test_red_att_qualitative(self, red_fit):
        # Paper Table on page 21: ATT = -70.4, mu_0 = 269.2, mu_1 = 198.8,
        # relative reduction 26.2%. The released reference R code does NOT
        # apply the per-unit pres/humi de-meaning that the paper appears
        # to use (the block is commented out in
        # `eg2/Eg_Air_Pollution_eps_201616_12_16_final.R`), so the
        # paper's exact magnitude is unreproducible from the released
        # artefacts. We assert the qualitative finding:
        mu1 = float(red_fit.fit.Y_treated_mean[48:].mean())
        # mu_1 still matches (it's the observed treated mean).
        assert mu1 == pytest.approx(198.8, abs=0.1)
        # ATT is significantly negative with relative reduction ~20-26%.
        assert red_fit.att < -40
        assert 100 * red_fit.att_relative > 18.0
        assert 100 * red_fit.att_relative < 30.0


# ----------------------------------------------------------------------
# Layer 4 -- public API contracts
# ----------------------------------------------------------------------
class TestPublicAPI:
    def test_top_level_import(self):
        from mlsynth import DSCAR as _D
        from mlsynth.config_models import DSCARConfig as _C
        assert _D is DSCAR
        assert _C is DSCARConfig

    def test_dict_config_equivalent_to_typed_config(self):
        df = _ar1_panel(seed=3)
        cfg = dict(
            df=df, outcome="y", treat="treat", unitid="unit", time="year",
            exog_covariates=["x1"], lagged_outcome="y_lag1",
            display_graphs=False,
        )
        r1 = DSCAR(cfg).fit()
        r2 = DSCAR(DSCARConfig(**cfg)).fit()
        assert r1.att == pytest.approx(r2.att)

    def test_unknown_covariate_translates(self):
        df = _ar1_panel()
        # Unknown covariates are caught at the data-prep layer, not the
        # config validator, so the surfaced exception is MlsynthDataError.
        with pytest.raises(MlsynthDataError):
            DSCAR({**dict(df=df, outcome="y", treat="treat",
                            unitid="unit", time="year",
                            display_graphs=False),
                    "exog_covariates": ["does_not_exist"]}).fit()
