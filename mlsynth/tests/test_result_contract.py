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

import numpy as np
import pandas as pd
import pytest

from mlsynth import (
    CLUSTERSC, FDID, FSCM, HSC, LEXSCM, MCNNM, MSQRT, PDA, RESCM, RMSI, SBC,
    SCMO, SNN, SparseSC, SPOTSYNTH, TASC, TSSC, VanillaSC,
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
    rows = []
    for i in range(n_units):
        for t in range(T):
            rows.append({"unitid": f"u{i:02d}", "time": t, "y": Y[t, i],
                         "treat": int(i == 0 and t >= T0)})
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
    pytest.param(SPOTSYNTH, {}, id="SPOTSYNTH"),
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
    pytest.param(RESCM, {}, id="RESCM"),
    pytest.param(PDA, {}, id="PDA"),
]


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
