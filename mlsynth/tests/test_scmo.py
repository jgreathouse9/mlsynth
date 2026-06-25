"""Tests for the spec-driven, NumPy-first SCMO estimator and its helpers."""

import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.scmo import SCMO
from mlsynth.config_models import SCMOConfig
from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.scmo_helpers import (
    AVERAGED,
    CONCATENATED,
    MA,
    SEPARATE,
    SCMOInputs,
    SCMOResults,
    build_matching_matrix,
    build_spec,
    conformal_inference,
    derive_treatment,
    fit_scheme,
    model_average,
    prepare_scmo_inputs,
    resolve_schemes,
    simplex_weights,
)
from mlsynth.utils.fast_scm_helpers.structure import IndexSet


@pytest.fixture
def panel() -> pd.DataFrame:
    """Small factor-model panel: unit 0 treated at t>=15, a parallel donor subset."""
    rng = np.random.default_rng(0)
    N, T, T0, K = 8, 20, 15, 3
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)          # 2 common factors
    rows = []
    for i in range(N):
        load = rng.uniform(0.5, 1.5, size=2)
        pop = rng.uniform(5, 20)
        for t in range(T):
            base = 50.0 + f[t] @ load                       # shift positive (valid log/per_capita)
            y = {f"y{k+1}": base * (1 + 0.1 * k) + rng.normal(scale=0.3) for k in range(K)}
            treat = int(i == 0 and t >= T0)
            if treat:
                y["y1"] += 5.0                              # effect on primary outcome
            rows.append({"unit": f"u{i}", "time": 1960 + t,
                         "Population levels": pop, "treat": treat, **y})
    return pd.DataFrame(rows)


SPEC = {"year": 1972, "vars": {
    "a": "y1", "b": "y2", "c_pc": ("y3", "per_capita"), "d_log": ("y1", "log")}}


# --- helpers ---------------------------------------------------------------

def test_indexset_roundtrip():
    idx = IndexSet.from_labels(["a", "b", "c"])
    assert list(idx.get_labels(idx.get_index(["c", "a"]))) == ["c", "a"]


def test_simplex_weights_simplex():
    rng = np.random.default_rng(1)
    Zt, ZJ = rng.normal(size=4), rng.normal(size=(5, 4))
    w = simplex_weights(Zt, ZJ)
    assert w.shape == (5,)
    assert np.all(w >= -1e-9) and abs(w.sum() - 1.0) < 1e-6


def test_build_matching_matrix(panel):
    unit_index = IndexSet.from_labels(list(pd.unique(panel["unit"])))
    Z, labels, col_period = build_matching_matrix(
        panel, unitid="unit", time="time", spec=SPEC, unit_index=unit_index)
    assert Z.shape == (8, len(labels))
    assert set(labels) <= {"a", "b", "c_pc", "d_log"}
    np.testing.assert_allclose(Z.std(axis=0, ddof=1), 1.0, atol=1e-8)
    assert np.all(col_period == 1972)


def test_prepare_inputs(panel):
    inp = prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                              spec=SPEC, treated_unit="u0", intervention_time=1975)
    assert isinstance(inp, SCMOInputs)
    assert inp.treated_label == "u0"
    assert inp.T == 20 and inp.T0 == 15 and inp.n_donors == 7
    assert inp.Y.shape == (8, 20)
    assert inp.y_treated.shape == (20,) and inp.Y_donors.shape == (7, 20)


# --- schemes ---------------------------------------------------------------

@pytest.fixture
def inputs(panel):
    return prepare_scmo_inputs(panel, unitid="unit", time="time", outcome="y1",
                               spec=SPEC, treated_unit="u0", intervention_time=1975)


@pytest.mark.parametrize("scheme", [CONCATENATED, AVERAGED, SEPARATE])
def test_fit_scheme(inputs, scheme):
    fit = fit_scheme(inputs, scheme)
    assert fit.counterfactual.shape == (inputs.T,)
    assert fit.gap.shape == (inputs.T,)
    assert abs(float(fit.weights.sum()) - 1.0) < 1e-6
    assert np.isfinite(fit.att) and fit.pre_rmse >= 0


def test_fit_scheme_demean(inputs):
    fit = fit_scheme(inputs, CONCATENATED, demean=True)
    assert fit.counterfactual.shape == (inputs.T,)
    assert fit.metadata["demean"] is True


def test_model_average(inputs):
    fits = [fit_scheme(inputs, CONCATENATED), fit_scheme(inputs, AVERAGED)]
    ma = model_average(inputs, fits)
    assert ma.name == MA
    lam = ma.metadata["lambdas"]
    assert abs(sum(lam.values()) - 1.0) < 1e-6


# --- inference (CWZ conformal) ---------------------------------------------

def test_conformal_inference(inputs):
    fit = fit_scheme(inputs, CONCATENATED)
    att, p, ci = conformal_inference(inputs.y_treated, fit.counterfactual, inputs.T0, alpha=0.1)
    assert np.isclose(att, fit.att)
    assert 0.0 < p <= 1.0
    assert ci[0] <= att <= ci[1]                       # ATT lies inside its CI


def test_conformal_detects_large_effect(inputs):
    # a big positive shift on the post-period gap should drive the p-value down
    cf_flat = inputs.y_treated.copy().astype(float)
    cf_flat[inputs.T0:] -= 1000.0                       # huge synthetic gap post
    _, p, _ = conformal_inference(inputs.y_treated, cf_flat, inputs.T0)
    assert p < 0.2


# --- orchestration helpers -------------------------------------------------

def test_resolve_schemes():
    assert resolve_schemes(None, "TLP") == [CONCATENATED]
    assert resolve_schemes(None, "SBMF") == [AVERAGED]
    assert resolve_schemes(None, "BOTH") == [CONCATENATED, AVERAGED, MA]
    assert resolve_schemes([SEPARATE], "TLP") == [SEPARATE]


def test_derive_treatment(panel):
    treated, t0, pre = derive_treatment(panel, "unit", "time", "treat")
    assert treated == "u0" and t0 == 1975
    assert pre == sorted(y for y in pd.unique(panel["time"]) if y < 1975)


def test_build_spec_from_addout():
    spec = build_spec(None, "y1", ["y2"], [1960, 1961, 1962])
    assert spec["year"] == [1960, 1961, 1962]
    assert set(spec["vars"]) == {"y1", "y2"}


# --- estimator -------------------------------------------------------------

def test_scmo_fit_spec_path(panel):
    res = SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
                "time": "time", "spec": SPEC, "schemes": [CONCATENATED, AVERAGED, MA],
                "display_graphs": False}).fit()
    assert isinstance(res, SCMOResults)
    assert set(res.fits) == {CONCATENATED, AVERAGED, MA}
    assert res.selected_variant == CONCATENATED
    assert res.att == res.fits[CONCATENATED].att
    assert res.counterfactual.shape == (20,)
    for f in res.fits.values():
        assert 0.0 < f.p_value <= 1.0                  # CWZ conformal p-value
        assert len(f.ci) == 2 and f.ci[0] <= f.ci[1]   # conformal CI


def test_scmo_fit_addout_path(panel):
    res = SCMO({"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
                "time": "time", "addout": ["y2", "y3"], "method": "BOTH",
                "display_graphs": False}).fit()
    assert set(res.fits) == {CONCATENATED, AVERAGED, MA}
    for f in res.fits.values():
        assert 0.0 < f.p_value <= 1.0
        assert f.ci[0] <= f.att <= f.ci[1]


def test_scmo_method_tlp_default(panel):
    res = SCMO(SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                          time="time", spec=SPEC, display_graphs=False)).fit()
    assert list(res.fits) == [CONCATENATED]            # method="TLP" default


def test_scmo_no_treated_raises(panel):
    df = panel.copy(); df["treat"] = 0
    with pytest.raises(MlsynthDataError):
        SCMO({"df": df, "outcome": "y1", "treat": "treat", "unitid": "unit",
              "time": "time", "spec": SPEC, "display_graphs": False}).fit()


# ------------------------------------------------- augment="ridge" (bilevel) ----

def test_fit_scheme_augment_runs(inputs):
    fit = fit_scheme(inputs, CONCATENATED, demean=False, augment="ridge")
    assert np.isfinite(fit.att)
    assert np.all(np.isfinite(fit.counterfactual))
    assert fit.metadata.get("augment") == "ridge"


def test_augment_changes_weights(inputs):
    plain = fit_scheme(inputs, CONCATENATED, demean=False)
    ridge = fit_scheme(inputs, CONCATENATED, demean=False, augment="ridge")
    assert not np.allclose(plain.weights, ridge.weights)
    assert ridge.weights.shape == plain.weights.shape


def test_scmo_augment_end_to_end(panel):
    cfg = {"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
           "time": "time", "addout": ["y2"], "schemes": ["concatenated"],
           "augment": "ridge", "demean": True, "display_graphs": False}
    res = SCMO(cfg).fit()
    assert np.isfinite(res.att)


def test_scmo_augment_fixed_lambda_changes_att(panel):
    common = {"df": panel, "outcome": "y1", "treat": "treat", "unitid": "unit",
              "time": "time", "schemes": ["concatenated"], "display_graphs": False}
    plain = SCMO(common).fit().att
    ridge = SCMO({**common, "augment": "ridge", "ridge_lambda": 5.0}).fit().att
    assert np.isfinite(ridge)
    assert abs(plain - ridge) > 1e-6


def test_ridge_lambda_without_augment_raises(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", ridge_lambda=5.0)


def test_ridge_lambda_nonpositive_raises(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", augment="ridge", ridge_lambda=0.0)
