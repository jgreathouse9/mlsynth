"""Tests for the SMC (Zhu 2023, Synthetic Matching Control) estimator.

Layer 1 -- solver primitive: the exact box QP matches an independent cvxpy
oracle to solver tolerance, certifies its own KKT point, pins the box bounds,
and is warm-start invariant.
Layer 2 -- weight computation: Algorithm 1 recovers an exact donor combination,
is deterministic, and tolerates a rank-deficient (collinear) donor block.
Layer 3 -- data setup: prepare_smc_inputs identifies the treated unit, enforces
a balanced panel with a donor pool and enough pre-periods.
Layer 4 -- estimator integration: SMC.fit() runs end-to-end (with and without
covariates), reproduces the Basque study, and returns an SMCResults respecting
the standardized contract.
Layer 5 -- public API / failure contracts: dict-vs-config equivalence, config
validation, and exception translation (failures are reported, not swallowed).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import SMC
from mlsynth.config_models import SMCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
)
from mlsynth.utils.smc_helpers import (
    SMCFit,
    SMCInputs,
    SMCResults,
    counterfactual,
    prepare_smc_inputs,
    run_smc,
    smc_weights,
    solve_box_qp,
)

BASQUE = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "basque_data.csv")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _toy_panel(n_donors=5, T=15, T0=10, seed=0, effect=-2.0):
    """Factor-model panel with one treated unit ``u0`` ~ donors u1, u2."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.normal(size=T))
    rows = []
    units = ["u0"] + [f"u{j}" for j in range(1, n_donors + 1)]
    for u in units:
        for t in range(T):
            if u == "u0":
                y = 0.6 * factor[t] + rng.normal(scale=0.1)
            elif u in ("u1", "u2"):
                y = factor[t] + rng.normal(scale=0.1)
            else:
                y = rng.normal()
            if u == "u0" and t >= T0:
                y += effect
            rows.append({"unit": u, "t": t, "y": y, "treat": int(u == "u0" and t >= T0)})
    return pd.DataFrame(rows)


@pytest.fixture
def panel():
    return _toy_panel()


@pytest.fixture
def basque():
    df = pd.read_csv(BASQUE)
    df["treat"] = ((df["regionname"] == "Basque Country (Pais Vasco)")
                   & (df["year"] >= 1970)).astype(int)
    return df


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="t",
                display_graphs=False)
    base.update(kw)
    return SMCConfig(**base)


# --------------------------------------------------------------------------- #
# Layer 1 -- the exact box QP
# --------------------------------------------------------------------------- #
def _cvxpy_box_qp(D, d):
    import cvxpy as cp
    n = len(d)
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(D)) - d @ w),
               [w >= 0, w <= 1]).solve(solver=cp.CLARABEL)
    return np.asarray(w.value).ravel()


@pytest.mark.parametrize("seed", range(8))
def test_box_qp_matches_cvxpy(seed):
    rng = np.random.default_rng(seed)
    p = int(rng.integers(2, 25))
    A = rng.standard_normal((p + 3, p))
    D = A.T @ A + 1e-3 * np.eye(p)
    d = rng.standard_normal(p) * 3
    w, info = solve_box_qp(D, d, return_info=True)
    assert info["converged"]
    assert info["kkt"] < 1e-9                      # exact KKT point
    assert w.min() >= -1e-12 and w.max() <= 1 + 1e-12
    assert np.max(np.abs(w - _cvxpy_box_qp(D, d))) < 1e-6


def test_box_qp_warm_start_invariant():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((20, 12))
    D = A.T @ A + 1e-3 * np.eye(12)
    d = rng.standard_normal(12) * 2
    cold = solve_box_qp(D, d)
    warm = solve_box_qp(D, d, warm_start=rng.random(12))
    assert np.max(np.abs(cold - warm)) < 1e-10


def test_box_qp_pins_bounds_exactly():
    # d strongly negative on coord 0 -> w0 exactly at lower bound 0.
    D = np.eye(3) + 0.1
    d = np.array([-5.0, 0.2, 0.3])
    w = solve_box_qp(D, d)
    assert w[0] == 0.0


def test_box_qp_bad_shape_raises():
    with pytest.raises(ValueError):
        solve_box_qp(np.eye(3), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        solve_box_qp(np.eye(2), np.array([1.0, 2.0]), lo=1.0, hi=1.0)


# --------------------------------------------------------------------------- #
# Layer 2 -- the weight computation (Algorithm 1)
# --------------------------------------------------------------------------- #
def test_smc_weights_recovers_exact_combination():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 6))
    true = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    y = X @ true                                    # treated is an exact donor combo
    r = smc_weights(X, y)
    cf = counterfactual(X, r)
    assert np.sqrt(np.mean((y - cf) ** 2)) < 1e-2   # near-perfect in-sample fit
    assert r.w.min() >= -1e-9 and r.w.max() <= 1 + 1e-9


def test_smc_weights_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((25, 8)); y = rng.standard_normal(25)
    a = smc_weights(X, y); b = smc_weights(X, y)
    assert np.array_equal(a.combined, b.combined)
    assert a.bias == b.bias


def test_smc_weights_rank_deficient_collinear():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((8, 4))
    X = np.hstack([base, base + 1e-8 * rng.standard_normal((8, 4))])  # collinear pairs
    y = rng.standard_normal(8)
    r = smc_weights(X, y)                            # ridge keeps it finite
    assert np.all(np.isfinite(r.combined)) and np.isfinite(r.bias)


def test_smc_weights_V_equal_weights_noop():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, 5)); y = rng.standard_normal(20)
    a = smc_weights(X, y)
    b = smc_weights(X, y, V=np.ones(20))            # equal V == deterministic default
    assert np.allclose(a.combined, b.combined)


def test_smc_weights_combined_can_extrapolate():
    # theta unconstrained -> combined coefficient may exceed the box / go negative.
    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 4)); y = 3.0 * X[:, 0] - 2.0 * X[:, 1]
    r = smc_weights(X, y)
    assert r.combined.max() > 1.0 or r.combined.min() < 0.0


# --------------------------------------------------------------------------- #
# Layer 3 -- data setup
# --------------------------------------------------------------------------- #
def test_prepare_inputs_shapes(panel):
    inp = prepare_smc_inputs(panel, outcome="y", treat="treat", unitid="unit", time="t")
    assert inp.treated_label == "u0"
    assert inp.J == 5 and inp.T == 15 and inp.T0 == 10 and inp.T1 == 5
    assert inp.Y_donors.shape == (15, 5)
    assert not inp.has_covariates


def test_prepare_inputs_no_treated_raises(panel):
    bad = panel.copy(); bad["treat"] = 0
    with pytest.raises(MlsynthDataError):
        prepare_smc_inputs(bad, outcome="y", treat="treat", unitid="unit", time="t")


def test_prepare_inputs_empty_donor_pool_raises():
    # single-unit panel: the treated unit exists but there are no donors.
    df = _toy_panel(n_donors=1, T=12, T0=8)
    df = df[df["unit"] == "u0"].copy()
    with pytest.raises(MlsynthDataError):
        prepare_smc_inputs(df, outcome="y", treat="treat", unitid="unit", time="t")


def test_prepare_inputs_multiple_treated_raises(panel):
    bad = panel.copy()
    bad.loc[(bad["unit"] == "u1") & (bad["t"] >= 10), "treat"] = 1   # a 2nd treated unit
    with pytest.raises(MlsynthDataError):
        prepare_smc_inputs(bad, outcome="y", treat="treat", unitid="unit", time="t")


def test_prepare_inputs_too_few_preperiods_raises():
    df = _toy_panel(T=6, T0=1)
    with pytest.raises(MlsynthDataError):
        prepare_smc_inputs(df, outcome="y", treat="treat", unitid="unit", time="t")


def test_prepare_inputs_nan_outcome_raises(panel):
    bad = panel.copy()
    bad.loc[(bad["unit"] == "u1") & (bad["t"] == 0), "y"] = np.nan   # balanced but NaN
    with pytest.raises(MlsynthDataError):
        prepare_smc_inputs(bad, outcome="y", treat="treat", unitid="unit", time="t")


def test_prepare_inputs_missing_covariate_raises(panel):
    df = panel.copy()
    df["x1"] = 1.0
    df.loc[df["unit"] == "u3", "x1"] = np.nan       # covariate absent for one unit
    with pytest.raises(MlsynthDataError):
        prepare_smc_inputs(df, outcome="y", treat="treat", unitid="unit", time="t",
                           covariates=["x1"])


# --------------------------------------------------------------------------- #
# Layer 4 -- estimator integration
# --------------------------------------------------------------------------- #
def test_fit_smoke(panel):
    res = SMC(_cfg(panel)).fit()
    assert isinstance(res, SMCResults)
    T = panel["t"].nunique()
    assert len(res.time_series.counterfactual_outcome) == T
    assert np.isfinite(res.att)
    # standardized sub-models born-populated
    assert res.effects is not None and res.weights is not None
    assert res.fit_diagnostics is not None and res.method_details.method_name == "SMC"


def test_fit_recovers_negative_effect(panel):
    res = SMC(_cfg(panel)).fit()
    assert res.att < -0.5                            # planted effect is -2.0
    assert res.box_weights.min() >= -1e-9 and res.box_weights.max() <= 1 + 1e-9


def test_fit_gap_identity(panel):
    res = SMC(_cfg(panel)).fit()
    ts = res.time_series
    assert np.allclose(ts.estimated_gap,
                       ts.observed_outcome - ts.counterfactual_outcome)


def test_fit_deterministic(panel):
    a = SMC(_cfg(panel)).fit(); b = SMC(_cfg(panel)).fit()
    assert np.allclose(a.weights_vector, b.weights_vector)
    assert a.att == b.att


def test_fit_with_covariates(panel):
    df = panel.copy()
    rng = np.random.default_rng(7)
    df["x1"] = rng.uniform(size=len(df)); df["x2"] = rng.uniform(size=len(df))
    res = SMC(_cfg(df, covariates=["x1", "x2"])).fit()
    assert np.isfinite(res.att)
    assert res.weights.summary_stats["n_covariates"] == 2
    assert res.weights.summary_stats["n_matching_rows"] == 12  # 10 pre-outcomes + 2 covs


def test_basque_reproduction(basque):
    res = SMC(SMCConfig(df=basque, outcome="gdpcap", treat="treat",
                        unitid="regionname", time="year", display_graphs=False)).fit()
    assert res.fit_diagnostics.rmse_pre < 0.1        # tight pre-fit (~0.048)
    assert res.att < 0                               # terrorism depresses GDP
    dw = {k: v for k, v in res.weights.donor_weights.items() if abs(v) > 0.05}
    assert "Madrid (Comunidad De)" in dw and "Murcia (Region de)" in dw


def test_plotting_smoke(panel, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = str(tmp_path / "smc.png")
    SMC(_cfg(panel, display_graphs=True, save=out)).fit()
    assert os.path.exists(out)


def test_plotting_show_branch(panel):
    import matplotlib
    matplotlib.use("Agg")                            # show() is a no-op under Agg
    res = SMC(_cfg(panel, display_graphs=True, save=False)).fit()
    assert res is not None


# --------------------------------------------------------------------------- #
# Layer 5 -- public API / failure contracts
# --------------------------------------------------------------------------- #
def test_dict_config_equivalence(panel):
    cfg = dict(df=panel, outcome="y", treat="treat", unitid="unit", time="t",
               display_graphs=False)
    a = SMC(cfg).fit()
    b = SMC(_cfg(panel)).fit()
    assert np.allclose(a.weights_vector, b.weights_vector)


def test_bad_dict_config_raises_config_error(panel):
    cfg = dict(df=panel, outcome="y", treat="treat", unitid="unit", time="t",
               ridge=0.0, display_graphs=False)               # ridge must be > 0
    with pytest.raises(MlsynthConfigError):
        SMC(cfg)


def test_ridge_must_be_positive(panel):
    with pytest.raises(Exception):
        _cfg(panel, ridge=-1.0)


def test_empty_covariates_list_raises(panel):
    with pytest.raises(MlsynthConfigError):
        _cfg(panel, covariates=[])


def test_extra_field_forbidden(panel):
    with pytest.raises(Exception):
        _cfg(panel, not_a_field=3)


def test_no_treated_raises_data_error(panel):
    bad = panel.copy(); bad["treat"] = 0
    with pytest.raises(MlsynthDataError):
        SMC(_cfg(bad)).fit()


def test_unbalanced_panel_raises_data_error(panel):
    bad = panel.drop(panel[(panel["unit"] == "u3") & (panel["t"] == 4)].index)
    with pytest.raises(MlsynthDataError):
        SMC(_cfg(bad)).fit()
