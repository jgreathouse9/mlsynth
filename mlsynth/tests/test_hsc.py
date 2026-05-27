"""Tests for the Harmonic Synthetic Control (HSC) estimator.

Covers config validation, the helper math (metric / QP / forecaster), the
estimator integration, plotting, and -- crucially -- that the packaged
estimator reproduces the standalone helper computation exactly.

Reference: Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
from matplotlib.figure import Figure

from mlsynth import HSC
from mlsynth.config_models import HSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.hsc_helpers.formulation import (
    difference_operator,
    fit_donor_weights,
    roughness_matrix,
    smoother_and_metric,
)
from mlsynth.utils.hsc_helpers.forecast import arima110_forecast, forecast_smooth
from mlsynth.utils.hsc_helpers.optimization import fit_at_rho, select_rho_by_cv
from mlsynth.utils.hsc_helpers.plotter import plot_hsc
from mlsynth.utils.hsc_helpers.structures import HSCInputs, HSCResults


# ----------------------------------------------------------------------
# DGP helper (reduced version of the paper's Appendix C.1)
# ----------------------------------------------------------------------

def _integ_ar1(T, phi, innov):
    d = np.zeros(T)
    for t in range(1, T):
        d[t] = phi * d[t - 1] + innov[t]
    return np.cumsum(d)


def _simulate(seed, N0=12, T0=60, Tpost=6, kappa=2.0, rho_u=0.0):
    rng = np.random.default_rng(seed)
    T = T0 + Tpost
    F = np.column_stack([
        np.cumsum(rng.normal(0, 2, T)),
        _integ_ar1(T, 0.5, rng.normal(0, 2, T)),
        np.array([0.0] + list(np.cumsum(rng.normal(0, 1, T - 1)))),
    ])
    Lam = np.clip(rng.normal(0, 0.5, (N0, 3)), -2, 2)
    S = rng.choice(N0, 8, replace=False)
    lam0 = rng.dirichlet(np.ones(8) * 0.5) @ Lam[S]
    units = np.vstack([lam0, Lam])
    L = units @ F.T
    uc = rng.normal(0, np.sqrt(1 - 0.25 ** 2), T)
    E = np.zeros((N0 + 1, T))
    for j in range(N0 + 1):
        ui = rng.normal(0, np.sqrt(1 - 0.25 ** 2), T)
        U = np.sqrt(rho_u) * uc + np.sqrt(1 - rho_u) * ui
        E[j] = _integ_ar1(T, 0.25, U)
    alpha = np.concatenate([[0.0], rng.uniform(5, 15, N0)])
    Y = L + kappa * E + rng.normal(0, 1, (N0 + 1, T)) + alpha[:, None] + rng.normal(0, 1, T)[None, :]
    return Y, T0


def _to_df(Y, T0):
    rows = []
    for j in range(Y.shape[0]):
        for t in range(Y.shape[1]):
            rows.append({"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
                         "treat": int(j == 0 and t >= T0)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    Y, T0 = _simulate(0)
    return _to_df(Y, T0)


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------

class TestHSCConfig:
    def test_defaults(self, panel):
        cfg = HSCConfig(df=panel, outcome="y", unitid="unit", time="time", treat="treat")
        assert cfg.q == 1 and cfg.forecaster == "arima110" and cfg.cv_splits == 3
        assert cfg.rho_grid == [0.0, 0.2, 0.5, 0.8, 0.97]

    def test_rho_grid_out_of_range_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            HSCConfig(df=panel, outcome="y", unitid="unit", time="time",
                      treat="treat", rho_grid=[0.5, 1.5])

    def test_bad_q_rejected(self, panel):
        with pytest.raises(Exception):
            HSCConfig(df=panel, outcome="y", unitid="unit", time="time",
                      treat="treat", q=3)


# ----------------------------------------------------------------------
# Helper math
# ----------------------------------------------------------------------

class TestFormulation:
    def test_difference_operator_shape(self):
        assert difference_operator(10, 1).shape == (9, 10)
        assert difference_operator(10, 2).shape == (8, 10)

    def test_metric_boundaries(self):
        T, q = 12, 1
        S0, W0 = smoother_and_metric(T, q, 0.0)
        assert np.allclose(S0, np.eye(T)) and np.allclose(W0, roughness_matrix(T, q))
        S1, W1 = smoother_and_metric(T, q, 1.0)
        assert np.allclose(S1 + W1, np.eye(T))         # P0 + (I - P0) = I
        # W is symmetric PSD at an interior rho
        _, W = smoother_and_metric(T, q, 0.5)
        assert np.allclose(W, W.T)
        assert np.min(np.linalg.eigvalsh(W)) > -1e-8

    def test_fit_donor_weights_on_simplex(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 5)); Y = X @ np.array([.4, .3, .3, 0, 0]) + 0.01 * rng.standard_normal(20)
        w = fit_donor_weights(X, Y, np.eye(20))
        assert w.shape == (5,) and (w >= -1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-6


class TestForecast:
    def test_arima110_extrapolates_trend(self):
        x = np.arange(30.0)                 # linear drift
        fc = arima110_forecast(x, 5)
        assert fc.shape == (5,)
        assert fc[0] > x[-1] and np.all(np.diff(fc) > 0)   # keeps rising
    def test_dispatch_and_last(self):
        x = np.array([1.0, 2.0, 5.0])
        assert np.allclose(forecast_smooth(x, 3, "last"), 5.0)
        with pytest.raises(ValueError):
            forecast_smooth(x, 3, "nope")


# ----------------------------------------------------------------------
# Estimator integration
# ----------------------------------------------------------------------

class TestEstimator:
    def test_fit_returns_results(self, panel):
        res = HSC({"df": panel, "outcome": "y", "unitid": "unit",
                   "time": "time", "treat": "treat"}).fit()
        assert isinstance(res, HSCResults) and res.mode == "hsc"
        assert 0.0 <= res.selected_rho <= 1.0
        assert res.counterfactual_full.shape == (res.inputs.T,)
        assert res.treatment_effect.shape == (res.inputs.n_post,)
        assert abs(sum(res.weights_by_donor.values()) - 1.0) < 1e-6

    def test_q2_runs(self, panel):
        res = HSC({"df": panel, "outcome": "y", "unitid": "unit",
                   "time": "time", "treat": "treat", "q": 2}).fit()
        assert np.isfinite(res.att)

    def test_too_few_donors_raises(self):
        rng = np.random.default_rng(0)
        rows = []
        for j in range(2):                     # u0 treated, u1 the only donor
            for t in range(20):
                rows.append({"unit": f"u{j}", "time": t,
                             "y": float(t + 2 * j + rng.normal()),
                             "treat": int(j == 0 and t >= 15)})
        with pytest.raises(MlsynthDataError):
            HSC({"df": pd.DataFrame(rows), "outcome": "y", "unitid": "unit",
                 "time": "time", "treat": "treat"}).fit()


# ----------------------------------------------------------------------
# Exact reproduction: estimator == direct helper computation
# ----------------------------------------------------------------------

class TestExactReproduction:
    def test_package_matches_helpers(self, panel):
        cfg = dict(outcome="y", unitid="unit", time="time", treat="treat")
        res = HSC({"df": panel, **cfg}).fit()
        inp = res.inputs
        # recompute the counterfactual directly from the helper functions
        best_rho, _ = select_rho_by_cv(
            inp.X_pre, inp.Y_pre, q=1, rho_grid=[0.0, 0.2, 0.5, 0.8, 0.97], n_splits=3
        )
        omega, E = fit_at_rho(inp.X_pre, inp.Y_pre, best_rho, q=1)
        cf_post = inp.X_post @ omega + arima110_forecast(E, inp.n_post)
        assert best_rho == res.selected_rho
        assert np.allclose(cf_post, res.design.counterfactual_post, atol=1e-8)


# ----------------------------------------------------------------------
# Regime adaptation (the paper's headline) and inference / plotting
# ----------------------------------------------------------------------

class TestRegimeAdaptation:
    def test_rho_adapts_across_regimes(self):
        cfg = dict(outcome="y", unitid="unit", time="time", treat="treat")
        rho_common, rho_idio = [], []
        for s in range(8):
            Yc, T0 = _simulate(100 + s, rho_u=1.0, kappa=2.0)
            Yi, _ = _simulate(100 + s, rho_u=0.0, kappa=2.0)
            rho_common.append(HSC({"df": _to_df(Yc, T0), **cfg}).fit().selected_rho)
            rho_idio.append(HSC({"df": _to_df(Yi, T0), **cfg}).fit().selected_rho)
        # shared trend -> match on levels (higher rho); idiosyncratic -> lean to differencing
        assert np.mean(rho_common) > np.mean(rho_idio)


class TestPlot:
    def test_plot_returns_none_without_crash(self, panel):
        res = HSC({"df": panel, "outcome": "y", "unitid": "unit",
                   "time": "time", "treat": "treat"}).fit()
        plot_hsc(res)        # uses Agg; should not raise
        _plt.close("all")


class TestSdidRidge:
    def test_config_accepts_sdid(self, panel):
        cfg = HSCConfig(df=panel, outcome="y", unitid="unit", time="time",
                        treat="treat", ridge="sdid")
        assert cfg.ridge == "sdid"

    def test_negative_float_ridge_rejected(self, panel):
        with pytest.raises(MlsynthConfigError):
            HSCConfig(df=panel, outcome="y", unitid="unit", time="time",
                      treat="treat", ridge=-1.0)

    def test_sdid_coefficient_positive(self):
        from mlsynth.utils.hsc_helpers.formulation import sdid_ridge_coefficient
        rng = np.random.default_rng(0)
        X = np.cumsum(rng.normal(0, 1, (40, 6)), axis=0)
        assert sdid_ridge_coefficient(X, n_post=5) > 0.0

    def test_sdid_diversifies_weights(self, panel):
        base = dict(outcome="y", unitid="unit", time="time", treat="treat")
        w_rel = HSC({"df": panel, **base, "ridge": 1e-6}).fit()
        w_sdid = HSC({"df": panel, **base, "ridge": "sdid"}).fit()
        # SDID ridge spreads weight: max donor weight no larger than the
        # near-unregularized fit, and typically smaller.
        assert max(w_sdid.weights_by_donor.values()) <= max(w_rel.weights_by_donor.values()) + 1e-9
