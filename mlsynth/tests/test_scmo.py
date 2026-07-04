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


# --- weights='pcr' scheme (mRSC-style denoised PCR) ------------------------

def _prepare(panel, addout="y2"):
    """Build SCMOInputs the way SCMO.fit does (concatenated, outcome+addout)."""
    treated_unit, intervention_time, pre_years = derive_treatment(
        panel, unitid="unit", time="time", treat="treat")
    spec = build_spec(None, "y1", addout, pre_years)
    return prepare_scmo_inputs(
        panel, unitid="unit", time="time", outcome="y1", spec=spec,
        treated_unit=treated_unit, intervention_time=intervention_time)


@pytest.fixture
def extrapolation_panel() -> pd.DataFrame:
    """Treated = 1.5*u1 - 0.5*u2 exactly on both outcomes (outside convex hull)."""
    rng = np.random.default_rng(7)
    T, T0 = 8, 6
    donors = {}
    for u in range(1, 5):
        donors[u] = {"y1": rng.uniform(5, 15, T), "y2": rng.uniform(5, 15, T)}
    rows = []
    for u in range(1, 5):
        for t in range(T):
            rows.append(dict(unit=f"u{u}", time=1960 + t, treat=0,
                             y1=donors[u]["y1"][t], y2=donors[u]["y2"][t]))
    for t in range(T):
        y1 = 1.5 * donors[1]["y1"][t] - 0.5 * donors[2]["y1"][t]
        y2 = 1.5 * donors[1]["y2"][t] - 0.5 * donors[2]["y2"][t]
        rows.append(dict(unit="u0", time=1960 + t, treat=int(t >= T0), y1=y1, y2=y2))
    return pd.DataFrame(rows)


def test_pcr_config_defaults_and_accept(panel):
    cfg = SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit", time="time")
    assert cfg.weights == "simplex"                      # default unchanged
    cfg2 = SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                      time="time", weights="pcr", pcr_rank=3, pcr_cumvar=0.9)
    assert cfg2.weights == "pcr" and cfg2.pcr_rank == 3


def test_pcr_rank_requires_pcr(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", pcr_rank=3)             # weights defaults to simplex


def test_pcr_incompatible_with_ridge(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", weights="pcr", augment="ridge")


def test_pcr_rank_positive(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", weights="pcr", pcr_rank=0)


def test_scmo_pcr_runs(panel):
    cfg = SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                     time="time", addout="y2", schemes=["concatenated"],
                     weights="pcr", display_graphs=False)
    res = SCMO(cfg).fit()
    assert np.isfinite(res.att)
    assert np.all(np.isfinite(res.time_series.counterfactual_outcome))


def test_pcr_weights_matches_kernel(panel):
    """SCMO's pcr scheme == direct pcr_weights on the matching matrix."""
    from mlsynth.utils.pcr.core import pcr_weights
    inputs = _prepare(panel)
    fit = fit_scheme(inputs, CONCATENATED, weights="pcr", pcr_rank=4)
    design = inputs.Z[inputs.donor_idx].T                # (P, J)
    target = inputs.Z[inputs.treated_idx]                # (P,)
    w_ref = pcr_weights(design, target, 4)
    np.testing.assert_allclose(fit.weights, w_ref, atol=1e-10)
    cf_ref = w_ref @ inputs.Y[inputs.donor_idx]
    np.testing.assert_allclose(fit.counterfactual, cf_ref, atol=1e-10)


def test_pcr_unconstrained_recovers_extrapolation(extrapolation_panel):
    """PCR (unconstrained) recovers a negative-weight combo simplex cannot."""
    inputs = _prepare(extrapolation_panel)
    pcr = fit_scheme(inputs, CONCATENATED, weights="pcr", pcr_rank=4)
    simplex = fit_scheme(inputs, CONCATENATED, weights="simplex")
    # PCR recovers the exact linear combination -> ~zero pre-treatment error
    assert pcr.pre_rmse < 1e-6
    # a genuine negative weight is present (outside the simplex)
    assert pcr.weights.min() < -0.1
    # the convex solver cannot represent it
    assert simplex.pre_rmse > 1e-3


# --- rolling-origin CV for the PCR metric weights (delta) ------------------

@pytest.fixture
def aux_noise_panel() -> pd.DataFrame:
    """Primary is an exact donor combo; the auxiliary outcome is pure noise.

    Rolling-origin CV should down-weight the uninformative auxiliary metric.
    """
    rng = np.random.default_rng(3)
    T, T0 = 12, 10
    J = 5
    dy1 = {u: rng.uniform(5, 15, T) for u in range(1, J + 1)}
    coef = np.array([1.3, -0.3, 0.0, 0.0, 0.0])
    rows = []
    for u in range(1, J + 1):
        for t in range(T):
            rows.append(dict(unit=f"u{u}", time=1960 + t, treat=0,
                             y1=dy1[u][t], y2=rng.uniform(5, 15)))   # aux = noise
    ty1 = sum(coef[u - 1] * dy1[u] for u in range(1, J + 1))
    for t in range(T):
        rows.append(dict(unit="u0", time=1960 + t, treat=int(t >= T0),
                         y1=ty1[t], y2=rng.uniform(5, 15)))          # aux = noise
    return pd.DataFrame(rows)


def test_pcr_metric_weights_requires_pcr(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", pcr_metric_weights="cv")     # weights=simplex


def test_pcr_cv_horizon_positive(panel):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", weights="pcr", pcr_metric_weights="cv",
                   pcr_cv_horizon=0)


def test_rolling_origin_cv_rejects_noise(aux_noise_panel):
    """Primary is an exact donor combo; the noisy aux only hurts OOS, so CV
    drives its weight to zero and OOS MSE rises monotonically with it."""
    from mlsynth.utils.scmo_helpers import rolling_origin_pcr_cv
    inputs = _prepare(aux_noise_panel)
    grid = [0.0, 0.25, 0.5, 1.0, 2.0]
    best, metric_order, mse_table = rolling_origin_pcr_cv(
        inputs, grid=grid, pcr_rank=5)                   # full rank -> exact recovery
    assert metric_order[0] == "y1"                       # primary anchored first
    assert best[1] == 0.0                                # noise aux fully rejected
    # more weight on the noisy metric strictly worsens out-of-sample MSE
    mses = [mse_table[g] for g in grid]
    assert all(mses[i] < mses[i + 1] for i in range(len(grid) - 1))
    # CV is never worse than equal weighting (1.0 is in the grid)
    assert min(mse_table.values()) <= mse_table[1.0] + 1e-12


def test_rolling_origin_cv_deterministic(aux_noise_panel):
    from mlsynth.utils.scmo_helpers import rolling_origin_pcr_cv
    inputs = _prepare(aux_noise_panel)
    a = rolling_origin_pcr_cv(inputs, grid=[0.0, 0.5, 1.0, 2.0])
    b = rolling_origin_pcr_cv(inputs, grid=[0.0, 0.5, 1.0, 2.0])
    assert a[0] == b[0] and a[2] == b[2]


def test_cv_noop_without_auxiliary(aux_noise_panel):
    """With a single metric there is nothing to tune -> equal weighting."""
    from mlsynth.utils.scmo_helpers import rolling_origin_pcr_cv
    inputs = _prepare(aux_noise_panel, addout=None)      # y1 only
    best, order, mse_table = rolling_origin_pcr_cv(inputs, grid=[0.5, 2.0])  # no 1.0
    assert order == ["y1"] and best == [1.0]


@pytest.mark.parametrize("kwargs", [
    {"pcr_metric_weights": [1.0, -0.5]},                 # negative metric weight
    {"pcr_metric_weights": []},                          # empty metric list
    {"pcr_metric_weights": "cv", "pcr_cv_grid": [-1.0, 1.0]},   # negative grid
    {"pcr_metric_weights": "cv", "pcr_cv_grid": []},     # empty grid
    {"pcr_metric_weights": "cv", "pcr_cv_min_train": 0},  # bad min_train
])
def test_pcr_cv_config_validation(panel, kwargs):
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError):
        SCMOConfig(df=panel, outcome="y1", treat="treat", unitid="unit",
                   time="time", weights="pcr", **kwargs)


def test_pcr_metric_weights_wrong_length_raises(aux_noise_panel):
    """An explicit weight vector must have one entry per metric."""
    from mlsynth.exceptions import MlsynthDataError
    cfg = SCMOConfig(df=aux_noise_panel, outcome="y1", treat="treat",
                     unitid="unit", time="time", addout="y2",
                     schemes=["concatenated"], weights="pcr",
                     pcr_metric_weights=[1.0, 1.0, 1.0], display_graphs=False)
    with pytest.raises(MlsynthDataError):
        SCMO(cfg).fit()


def test_cv_insufficient_preperiods_raises():
    """Too few pre-periods for the requested window -> clear estimation error."""
    from mlsynth.utils.scmo_helpers import rolling_origin_pcr_cv
    from mlsynth.exceptions import MlsynthEstimationError
    rng = np.random.default_rng(5)
    rows = []
    for u in range(4):
        for t in range(3):                               # T=3, treat at t>=2 -> T0=2
            rows.append(dict(unit=f"u{u}", time=t, treat=int(u == 0 and t >= 2),
                             y1=rng.normal(), y2=rng.normal()))
    inputs = _prepare(pd.DataFrame(rows))
    with pytest.raises(MlsynthEstimationError):
        rolling_origin_pcr_cv(inputs, grid=[0.0, 1.0], min_train=2, horizon=1)


def test_scmo_pcr_cv_end_to_end(aux_noise_panel):
    cfg = SCMOConfig(df=aux_noise_panel, outcome="y1", treat="treat",
                     unitid="unit", time="time", addout="y2",
                     schemes=["concatenated"], weights="pcr",
                     pcr_metric_weights="cv", display_graphs=False)
    res = SCMO(cfg).fit()
    assert np.isfinite(res.att)
    # the chosen weights are recorded for inspection
    md = res.method_details.parameters_used
    assert "pcr_metric_weights" in md and "pcr_cv_mse" in md


def test_scmo_explicit_metric_weights_recorded(aux_noise_panel):
    """Explicit per-metric weights run end-to-end and are recorded."""
    cfg = SCMOConfig(df=aux_noise_panel, outcome="y1", treat="treat",
                     unitid="unit", time="time", addout="y2",
                     schemes=["concatenated"], weights="pcr",
                     pcr_metric_weights=[1.0, 0.25], display_graphs=False)
    res = SCMO(cfg).fit()
    assert np.isfinite(res.att)
    recorded = res.method_details.parameters_used["pcr_metric_weights"]
    assert recorded == {"y1": 1.0, "y2": 0.25}


def test_pcr_explicit_zero_aux_matches_primary_only(aux_noise_panel):
    """pcr_metric_weights=[1, 0] drops the aux metric -> primary-only PCR."""
    from mlsynth.utils.pcr.core import pcr_weights
    inputs = _prepare(aux_noise_panel)
    fit = fit_scheme(inputs, CONCATENATED, weights="pcr", pcr_rank=5,
                     metric_weights=[1.0, 0.0])
    # primary-only design: Z columns belonging to y1
    labels = [str(l).split("@")[0] for l in inputs.predictor_labels]
    y1_cols = np.array([i for i, l in enumerate(labels) if l == "y1"])
    design = inputs.Z[np.ix_(inputs.donor_idx, y1_cols)].T
    target = inputs.Z[inputs.treated_idx][y1_cols]
    w_ref = pcr_weights(design, target, 5)
    np.testing.assert_allclose(fit.weights, w_ref, atol=1e-9)
