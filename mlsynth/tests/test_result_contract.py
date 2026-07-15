"""Conformance tests for the standardized two-family result contract.

mlsynth has exactly two output families:

* :class:`~mlsynth.config_models.EffectResult` -- the observational report
  returned by effect estimators (ATT, counterfactual, weights, inference);
* :class:`~mlsynth.config_models.DesignResult` -- the research design returned
  by experimental-design estimators, whose ``report`` is an ``EffectResult``.

Both inherit :class:`~mlsynth.config_models.MlsynthResult`. This module pins
the contract on the three reference estimators (VanillaSC, FDID, TSSC -- all
observational) so that the unified-results claim is machine-checked. As the
migration rolls out, every estimator should be added to ``OBSERVATIONAL``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

# BFSC draws its posterior with NUTS (NumPyro), behind the ``[bayes]`` optional
# dependency. The module-scoped ``fitted`` fixture fits every OBSERVATIONAL
# estimator eagerly, so a skip-marked param would still be fit (and crash) in
# the fixture loop -- include BFSC only when the backend is importable.
_HAS_NUMPYRO = importlib.util.find_spec("numpyro") is not None

from mlsynth import (
    BFSC, BPSCS, BSCM, BVSS, CFM, CLUSTERSC, CSCIPCA, DPSC, DSCAR, ISCM, FDID, FMA, FSCM, HSC, LEXSCM, MAREX, MASC, MEDSC,
    MCNNM, MSQRT, MTGP, NSC, PDA, PROPSC, PROXIMAL, RESCM, RMSI, SBC, SCMO, SCUL, SDID,
    SequentialSDID, SHC, SNN, SparseSC, SPILLSYNTH, SPOTSYNTH, SSC, TASC, TSSC,
    VanillaSC,
)
from mlsynth.config_models import (
    BaseEstimatorResults,
    DesignResult,
    EffectResult,
    MlsynthResult,
)


def _make_panel(n_units=6, T=30, T0=20, seed=0, rho=0.6):
    """Factor + AR(1) panel with a single treated unit (unit 0)."""
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = rho * common[t - 1] + rng.standard_normal()
    Y = np.zeros((T, n_units))
    for i in range(n_units):
        load = rng.standard_normal()
        Y[:, i] = 10.0 + load * common + rng.standard_normal(T) * 0.4
    coords = rng.uniform(0, 1, size=(n_units, 2)); coords[0] = [0.0, 0.0]
    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                         "treat": int(i == 0 and t >= T0),
                         # per-unit spatial coords + a baseline covariate (BPSCS;
                         # ignored by every other estimator, which use only y/treat)
                         "lat": float(coords[i, 0]), "lon": float(coords[i, 1]),
                         "cov1": float(coords[i].sum())})
    return pd.DataFrame(rows)


def _make_compositional_panel(n_units=6, T=8, T0=5, n_treated=2, K=3, seed=1):
    """Balanced panel with a K-proportion compositional outcome (rows sum to 1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        load = rng.standard_normal(K)
        for t in range(T):
            lat = 0.3 * i + 0.2 * t * np.arange(K) + load * (1 + 0.1 * t)
            treated = i >= n_units - n_treated
            if treated and t >= T0:
                lat[0] += 1.0
            ex = np.exp(lat - lat.max())
            p = ex / ex.sum()
            row = {"unitid": f"u{i:02d}", "time": t,
                   "treat": int(treated and t >= T0)}
            for k in range(K):
                row[f"p{k + 1}"] = float(p[k])
            rows.append(row)
    return pd.DataFrame(rows)


def _make_cscipca_panel(n_units=8, T=30, T0=20, L=3, K=2, seed=2):
    """Instrumented-PCA panel with L covariate columns; unit 0 treated."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((K, T))
    X = rng.standard_normal((n_units, T, L))
    Gamma = rng.uniform(-0.3, 0.3, (L, K))
    Y = np.einsum("itl,lk,kt->it", X, Gamma, F) + 0.1 * rng.standard_normal((n_units, T))
    Y[0, T0:] += 2.0
    rows = []
    for i in range(n_units):
        for t in range(T):
            row = {"unitid": f"u{i:02d}", "time": t, "y": float(Y[i, t]),
                   "treat": int(i == 0 and t >= T0)}
            for l in range(L):
                row[f"x{l}"] = float(X[i, t, l])
            rows.append(row)
    return pd.DataFrame(rows)


def _make_medsc_panel(n_units=8, T=24, T0=16, seed=3):
    """Panel with an outcome plus a mediator column; unit 0 treated."""
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = 0.6 * common[t - 1] + rng.standard_normal()
    rows = []
    for i in range(n_units):
        load = rng.standard_normal()
        med = 1.0 + 0.05 * i + 0.02 * np.arange(T) + 0.05 * rng.standard_normal(T)
        y = 10.0 + load * common - 2.0 * med + rng.standard_normal(T) * 0.3
        treated = i == 0
        for t in range(T):
            yy = y[t] - 3.0 if (treated and t >= T0) else y[t]
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": float(yy),
                         "price": float(med[t]),
                         "treat": int(treated and t >= T0)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel_df():
    return _make_panel()


def _base_cfg(panel_df):
    return {"df": panel_df, "outcome": "y", "unitid": "unitid",
            "time": "time", "treat": "treat", "display_graphs": False}


# Estimators that should return an EffectResult (the observational family).
# Extend this list as the migration proceeds.
OBSERVATIONAL = [
    pytest.param(VanillaSC, {}, id="VanillaSC"),
    pytest.param(FDID, {}, id="FDID"),
    pytest.param(TSSC, {"draws": 80, "seed": 0}, id="TSSC"),
    pytest.param(SPOTSYNTH, {"inference": "frequentist"}, id="SPOTSYNTH"),
    pytest.param(RMSI, {}, id="RMSI"),
    pytest.param(SNN, {}, id="SNN"),
    pytest.param(MSQRT, {"lambda_": 0.5}, id="MSQRT"),
    pytest.param(MCNNM, {}, id="MCNNM"),
    pytest.param(SparseSC, {"outcome_lag_periods": [1, 2]}, id="SparseSC"),
    pytest.param(TASC, {"d": 2, "n_em_iter": 2}, id="TASC"),
    pytest.param(CLUSTERSC, {"clustering": False}, id="CLUSTERSC"),
    pytest.param(SBC, {}, id="SBC"),
    pytest.param(HSC, {}, id="HSC"),
    pytest.param(FSCM, {}, id="FSCM"),
    pytest.param(SCMO, {}, id="SCMO"),
    pytest.param(PROPSC, {"df": _make_compositional_panel(), "outcome": "p1",
                          "outcomes": ["p1", "p2", "p3"]}, id="PROPSC"),
    pytest.param(SCUL, {"inference": False, "number_initial_periods": 4,
                        "training_post_length": 5}, id="SCUL"),
    pytest.param(RESCM, {}, id="RESCM"),
    pytest.param(PDA, {}, id="PDA"),
    pytest.param(MASC, {}, id="MASC"),
    pytest.param(NSC, {"a": 0.0, "b": 0.0}, id="NSC"),
    pytest.param(FMA, {}, id="FMA"),
    pytest.param(CFM, {"n_factors": 1}, id="CFM"),
    pytest.param(CSCIPCA, {"df": _make_cscipca_panel(),
                           "covariates": ["x0", "x1", "x2"], "n_factors": 2,
                           "inference": False}, id="CSCIPCA"),
    pytest.param(MEDSC, {"df": _make_medsc_panel(), "mediator": "price",
                         "inference": False}, id="MEDSC"),
    pytest.param(SDID, {}, id="SDID"),
    pytest.param(SequentialSDID, {"n_bootstrap": 20, "seed": 0}, id="SequentialSDID"),
    pytest.param(SSC, {}, id="SSC"),
    pytest.param(BVSS, {"n_iter": 30, "burn_in": 10, "seed": 0}, id="BVSS"),
    pytest.param(BSCM, {"n_iter": 60, "burn_in": 30, "chains": 2, "seed": 0}, id="BSCM"),
    pytest.param(DSCAR, {}, id="DSCAR"),
    pytest.param(ISCM, {}, id="ISCM"),
    pytest.param(SHC, {}, id="SHC"),
    pytest.param(SPILLSYNTH, {"method": "cd", "affected_units": ["u01"]},
                 id="SPILLSYNTH"),
    pytest.param(DPSC, {"epsilon1": 50.0, "epsilon2": 50.0, "n_draws": 20, "seed": 0},
                 id="DPSC"),
    # PROXIMAL is a dispatcher (like TSSC): the headline variant -- the first
    # requested method -- is lifted into the standardized sub-models. SPSC needs
    # no proxies, so it is the cleanest headline for the generic panel.
    pytest.param(PROXIMAL, {"methods": ["SPSC"],
                            "donors": ["u01", "u02", "u03", "u04", "u05"]},
                 id="PROXIMAL"),
]

if _HAS_NUMPYRO:  # BFSC/MTGP need the ``[bayes]`` extra; skip the params without it
    OBSERVATIONAL.append(
        pytest.param(BFSC, {"n_factors": 2, "n_warmup": 50, "n_samples": 50,
                            "n_chains": 1, "seed": 0}, id="BFSC")
    )
    OBSERVATIONAL.append(
        pytest.param(MTGP, {"n_factors": 2, "n_warmup": 50, "n_samples": 50,
                            "n_chains": 1, "seed": 0}, id="MTGP")
    )
    OBSERVATIONAL.append(
        pytest.param(BPSCS, {"covariates": ["cov1"], "coords": ["lat", "lon"],
                            "n_warmup": 80, "n_samples": 80, "n_chains": 1,
                            "target_accept": 0.9, "max_tree_depth": 8, "seed": 0},
                     id="BPSCS")
    )


@pytest.fixture(scope="module")
def fitted(panel_df):
    """Fit each observational estimator once and cache the result."""
    out = {}
    for param in OBSERVATIONAL:
        Est, extra = param.values
        cfg = {**_base_cfg(panel_df), **extra}
        out[param.id] = Est(cfg).fit()
    return out


@pytest.mark.parametrize("Est, extra", OBSERVATIONAL)
def test_is_effect_result(Est, extra, fitted):
    res = fitted[Est.__name__]
    assert isinstance(res, MlsynthResult)
    assert isinstance(res, BaseEstimatorResults)
    assert isinstance(res, EffectResult)  # EffectResult is BaseEstimatorResults


@pytest.mark.parametrize("Est, extra", OBSERVATIONAL)
def test_standard_submodels_populated(Est, extra, fitted):
    """Every observational result exposes the shared sub-models."""
    res = fitted[Est.__name__]
    assert res.effects is not None and res.effects.att is not None
    assert res.time_series is not None
    assert res.time_series.counterfactual_outcome is not None
    assert res.weights is not None
    assert res.method_details is not None and res.method_details.method_name


@pytest.mark.parametrize("Est, extra", OBSERVATIONAL)
def test_flat_accessor_contract(Est, extra, fitted):
    """The flat read contract: att / att_ci / counterfactual / gap /
    donor_weights / pre_rmse behave identically across estimators.
    """
    res = fitted[Est.__name__]

    assert isinstance(res.att, float)
    assert res.att == pytest.approx(res.effects.att)

    cf = np.asarray(res.counterfactual)
    gap = np.asarray(res.gap)
    assert cf.shape == gap.shape
    assert cf.ndim == 1

    dw = res.donor_weights
    assert dw is None or isinstance(dw, dict)

    assert res.pre_rmse is None or isinstance(res.pre_rmse, float)

    ci = res.att_ci
    assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])


@pytest.mark.parametrize("Est, extra", OBSERVATIONAL)
def test_serializable(Est, extra, fitted):
    """A standardized result serializes through pydantic (reproducibility)."""
    res = fitted[Est.__name__]
    dumped = res.model_dump(include={"effects", "fit_diagnostics", "inference"})
    assert "effects" in dumped
    assert dumped["effects"]["att"] is not None


# --- the design family (skeleton; design estimators migrate onto this) -----

def test_design_result_is_mlsynth_result():
    assert issubclass(DesignResult, MlsynthResult)
    assert not issubclass(DesignResult, EffectResult)


def test_design_report_is_effect_result():
    """A design resolves to an EffectResult via `report`."""
    report = EffectResult()
    design = DesignResult(report=report)
    assert isinstance(design.report, EffectResult)
    assert isinstance(design, MlsynthResult)


# --- design-family conformance on real estimators ------------------------

def _make_design_panel(n_units=15, T=40, T_post=12, n_candidates=8, L=2,
                       sigma=0.1, seed=0, baseline=100.0):
    """Factor panel with a candidate-eligibility column + post indicator."""
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal((n_units, L))
    nu = rng.standard_normal((T, L))
    Y = baseline + nu @ gamma.T + sigma * rng.standard_normal((T, n_units))
    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                         "post": int(t >= T - T_post),
                         "candidate": int(i < n_candidates)})
    return pd.DataFrame(rows)


# Estimators that should return a DesignResult (the experimental-design family).
DESIGN = [
    pytest.param(
        LEXSCM,
        {"outcome": "y", "unitid": "unitid", "time": "time",
         "candidate_col": "candidate", "m": 2, "post_col": "post",
         "top_K": 3, "n_sims": 30, "n_post_grid": [2, 4, 6, 8],
         "mde_horizon": "late", "verbose": False},
        id="LEXSCM",
    ),
    pytest.param(
        MAREX,
        {"outcome": "y", "unitid": "unitid", "time": "time",
         "design": "standard", "post_col": "post", "m_eq": 3, "relaxed": True},
        id="MAREX",
    ),
]


@pytest.fixture(scope="module")
def fitted_designs():
    df = _make_design_panel()
    return {p.id: p.values[0]({"df": df, **p.values[1]}).fit() for p in DESIGN}


@pytest.mark.parametrize("Est, extra", DESIGN)
def test_is_design_result(Est, extra, fitted_designs):
    res = fitted_designs[Est.__name__]
    assert isinstance(res, MlsynthResult)
    assert isinstance(res, DesignResult)
    assert not isinstance(res, EffectResult)  # the design family, not the report


@pytest.mark.parametrize("Est, extra", DESIGN)
def test_design_resolves_to_effect_report(Est, extra, fitted_designs):
    """A realized design exposes its effect report as an EffectResult."""
    res = fitted_designs[Est.__name__]
    assert res.selected_units is not None and len(res.selected_units) > 0
    report = res.report
    assert isinstance(report, EffectResult)
    # The report satisfies the same flat read-contract as any effect estimator.
    assert report.effects is not None and report.effects.att is not None
    assert isinstance(report.att, float)
    assert report.att == pytest.approx(report.effects.att)
    cf = np.asarray(report.counterfactual)
    gap = np.asarray(report.gap)
    assert cf.ndim == 1 and cf.shape == gap.shape
    ci = report.att_ci
    assert ci is None or (len(ci) == 2 and ci[0] <= ci[1])
