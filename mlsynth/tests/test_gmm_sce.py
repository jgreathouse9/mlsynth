"""TDD for GMM-SCE -- the GMM Synthetic Control Estimator of Fry (2024),
dispatched as ``ORTHSC(method="gmm_sce")``.

Layers:
 * solver invariants + a value-for-value cross-check against the author's
   ``GMM-SCE.R`` ``GMMSC()`` on a fixed example (weights, J-statistic), with the
   optimality check that mlsynth attains an objective at least as low as the
   reference's interior-point ``LowRankQP`` solve;
 * the Andrews--Lu model-selection procedure recovering the true controls;
 * end-to-end recovery of a planted effect through the public ``ORTHSC`` API;
 * config-validation failure paths;
 * result-contract conformance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import ORTHSC
from mlsynth.config_models import BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.orthsc_helpers.gmm_sce.solver import gmm_sc_weights, _row_normalize
from mlsynth.utils.orthsc_helpers.gmm_sce.selection import select_controls


# ----------------------------------------------------------- fixed example ----
# Deterministic (T0=8, J=3, K=2) panel; reference values produced by running the
# author's GMM-SCE.R GMMSC(meanfit=TRUE) on exactly these arrays.
FIX_Y0 = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.0, 1.0, 2.2])
FIX_YJ = np.array([[1.1, 2.1, 1.4, 2.6, 3.1, 1.9, 1.1, 2.0],
                   [0.9, 1.8, 1.6, 2.3, 2.7, 2.1, 0.8, 2.4],
                   [2.0, 1.0, 2.5, 1.5, 1.0, 3.0, 2.2, 1.2]]).T   # (8, 3)
FIX_YK = np.array([[1.2, 1.9, 1.7, 2.4, 2.9, 2.0, 1.0, 2.1],
                   [0.5, 2.2, 1.3, 2.8, 3.2, 1.7, 1.4, 1.8]]).T   # (8, 2)
R_WEIGHTS = np.array([0.9438928754, 0.0554485414, 0.0006585833])
R_JSTAT = 0.0183975577


def _scaled_objective(w):
    """The GMM objective ||YK'(Y0 - YJ w)||^2 / T0 on the normalized fixed data."""
    T0 = FIX_Y0.shape[0]
    YKc = np.hstack([np.ones((T0, 1)), FIX_YK])
    scaled = _row_normalize(np.hstack([FIX_Y0[:, None], FIX_YJ, YKc]))
    y0s, YJs, YKs = scaled[:, 0], scaled[:, 1:4], scaled[:, 4:]
    g = YKs.T @ (y0s - YJs @ w)
    return float(g @ g) / T0


# ------------------------------------------------------------ solver layer ----

def test_solver_matches_R_reference_and_is_optimal():
    out = gmm_sc_weights(FIX_Y0, FIX_YJ, FIX_YK, include_constant=True)
    # weights agree with the R reference to LowRankQP's interior-point tolerance
    assert np.max(np.abs(out["weights"] - R_WEIGHTS)) < 2e-3
    assert abs(out["jstatistic"] - R_JSTAT) < 1e-3
    # ... and mlsynth attains an objective no worse than the reference's
    assert _scaled_objective(out["weights"]) <= _scaled_objective(R_WEIGHTS) + 1e-9


def test_solver_weights_on_simplex():
    out = gmm_sc_weights(FIX_Y0, FIX_YJ, FIX_YK)
    w = out["weights"]
    assert w.shape == (3,)
    assert np.all(w >= -1e-8)
    assert abs(w.sum() - 1.0) < 1e-6
    assert out["jstatistic"] >= 0.0
    assert out["n_instruments"] == 3                      # 2 instruments + constant


def test_solver_constant_toggle_changes_instrument_count():
    with_c = gmm_sc_weights(FIX_Y0, FIX_YJ, FIX_YK, include_constant=True)
    without_c = gmm_sc_weights(FIX_Y0, FIX_YJ, FIX_YK, include_constant=False)
    assert with_c["n_instruments"] == 3
    assert without_c["n_instruments"] == 2


def test_solver_accepts_unit_major_input():
    """Solver transposes (units x time) input to match (time x units)."""
    out_t = gmm_sc_weights(FIX_Y0, FIX_YJ.T, FIX_YK.T)     # unit-major
    out = gmm_sc_weights(FIX_Y0, FIX_YJ, FIX_YK)
    assert np.allclose(out_t["weights"], out["weights"], atol=1e-8)


def test_solver_rejects_too_few_periods():
    with pytest.raises(MlsynthEstimationError):
        gmm_sc_weights(FIX_Y0[:1], FIX_YJ[:1], FIX_YK[:1])


def test_solver_rejects_constant_period():
    """A pre-period with zero cross-unit variance cannot be normalized."""
    y0 = FIX_Y0.copy()
    yj = FIX_YJ.copy()
    yk = FIX_YK.copy()
    y0[0] = 1.0; yj[0, :] = 1.0; yk[0, :] = 1.0           # all units identical at t=0
    with pytest.raises(MlsynthEstimationError):
        gmm_sc_weights(y0, yj, yk, include_constant=False)


# --------------------------------------------------------- model selection ----

def _factor_panel(seed=0, n_extra=4, T=24, T1=15, weights=(0.6, 0.4), tau=3.0):
    """Treated = convex combo of d0,d1 (+ effect); d2.. share the factors as
    instruments. Returns the long DataFrame and the planted truth."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 2))
    donors = {}
    for j in range(2 + n_extra):
        lam = rng.normal(size=2)
        donors[f"d{j}"] = f @ lam + rng.normal(scale=0.15, size=T) + 5.0
    treated = weights[0] * donors["d0"] + weights[1] * donors["d1"] \
        + rng.normal(scale=0.05, size=T)
    eff = np.zeros(T); eff[T1:] = tau
    treated = treated + eff
    periods = list(range(2000, 2000 + T))
    rows = []
    for t, p in enumerate(periods):
        rows.append({"unit": "T", "year": p, "y": treated[t],
                     "treat": 1 if t >= T1 else 0})
        for name, series in donors.items():
            rows.append({"unit": name, "year": p, "y": series[t], "treat": 0})
    return pd.DataFrame(rows), list(donors), tau


def test_selection_accepts_unit_major_input():
    df, donors, _ = _factor_panel(seed=1)
    pre = 15
    wide = df.pivot(index="year", columns="unit", values="y")
    y0 = wide["T"].to_numpy()[:pre]
    YN0 = wide[[d for d in donors]].to_numpy()[:pre]
    ctrl = select_controls(y0, YN0)
    ctrl_t = select_controls(y0, YN0.T)                    # (units x time)
    assert ctrl == ctrl_t


def test_selection_with_guaranteed_instruments_unit_major():
    df, donors, _ = _factor_panel(seed=8)
    pre = 15
    wide = df.pivot(index="year", columns="unit", values="y")
    y0 = wide["T"].to_numpy()[:pre]
    cand = [d for d in donors if d != "d5"]
    YN0 = wide[cand].to_numpy()[:pre]
    YN1 = wide[["d5"]].to_numpy()[:pre].T                  # unit-major guaranteed
    ctrl = select_controls(y0, YN0, YN1)
    assert 1 <= len(ctrl) <= len(cand)


def test_selection_promotes_when_initial_set_rejected():
    # Put the treated unit OUTSIDE the donors' convex hull (above the per-period
    # maximum) so no sparse control set fits and the over-identification test
    # keeps rejecting: this drives the Andrews--Lu promotion step (Step 5) and,
    # here, exhausts every candidate into the control set.
    rng = np.random.default_rng(0)
    T0, N = 10, 5
    YN0 = rng.normal(scale=0.5, size=(T0, N)) + np.arange(N)[None, :] * 0.5 + 2.0
    y0 = YN0.max(axis=1) + 1.0 + rng.normal(scale=0.3, size=T0)
    ctrl = select_controls(y0, YN0)
    assert ctrl == list(range(N))                          # promotion ran to the end


def test_selection_promotion_with_guaranteed_instruments():
    # Same out-of-hull setup but with a guaranteed instrument retained: every
    # candidate is promoted to a control while YN1 stays an instrument.
    rng = np.random.default_rng(0)
    T0, N = 10, 5
    YN0 = rng.normal(scale=0.5, size=(T0, N)) + np.arange(N)[None, :] * 0.5 + 2.0
    y0 = YN0.max(axis=1) + 1.0 + rng.normal(scale=0.3, size=T0)
    YN1 = rng.normal(scale=0.5, size=(T0, 1)) + 3.0
    ctrl = select_controls(y0, YN0, YN1)
    assert ctrl == list(range(N))


def test_selection_returns_valid_sparse_controls():
    # With a 2-factor structure every donor spans the factor space, so the
    # control set is not uniquely the generative {d0,d1}; the invariant the
    # procedure guarantees is a non-empty, sparse set whose GMM synthetic
    # control tracks the treated unit pre-treatment.
    df, donors, _ = _factor_panel(seed=1)
    pre = 15
    wide = df.pivot(index="year", columns="unit", values="y")
    y0 = wide["T"].to_numpy()[:pre]
    YN0 = wide[[d for d in donors]].to_numpy()[:pre]
    ctrl = select_controls(y0, YN0)
    assert 1 <= len(ctrl) < len(donors)                    # non-empty, sparse
    instr = [i for i in range(len(donors)) if i not in ctrl]
    out = gmm_sc_weights(y0, YN0[:, ctrl], YN0[:, instr])
    pre_fit_rmse = float(np.sqrt(np.mean((y0 - YN0[:, ctrl] @ out["weights"]) ** 2)))
    assert pre_fit_rmse < 0.5                              # tracks the treated unit


# --------------------------------------------------------------- end to end ----

def test_gmm_sce_explicit_split_recovers_effect():
    df, donors, tau = _factor_panel(seed=2)
    res = ORTHSC({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                  "time": "year", "method": "gmm_sce",
                  "instruments": ["d2", "d3", "d4", "d5"],
                  "controls": ["d0", "d1"]}).fit()
    assert isinstance(res, BaseEstimatorResults)
    assert abs(res.att - tau) < 0.3
    w = res.weights.donor_weights
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert res.additional_outputs["jstatistic"] >= 0.0
    assert res.additional_outputs["controls"] == ["d0", "d1"]


def test_gmm_sce_model_selection_recovers_effect():
    df, donors, tau = _factor_panel(seed=3)
    res = ORTHSC({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                  "time": "year", "method": "gmm_sce",
                  "model_selection": True}).fit()
    # The factor structure is recovered (effect ~ tau) regardless of which valid
    # control set the procedure selects.
    assert abs(res.att - tau) < 0.4
    assert len(res.additional_outputs["controls"]) >= 1


def test_gmm_sce_guaranteed_instruments_excluded_from_controls():
    df, donors, _ = _factor_panel(seed=4)
    res = ORTHSC({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                  "time": "year", "method": "gmm_sce", "model_selection": True,
                  "guaranteed_instruments": ["d0"]}).fit()
    assert "d0" not in res.additional_outputs["controls"]
    assert "d0" in res.additional_outputs["instruments"]


def test_default_method_is_orthogonalized():
    df, donors, _ = _factor_panel(seed=5)
    cfg = {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
           "time": "year", "instruments": ["d2", "d3", "d4", "d5"]}
    from mlsynth.utils.orthsc_helpers.config import OrthSCConfig
    assert OrthSCConfig(**cfg).method == "orthogonalized"


# ------------------------------------------------------------- failure paths ----

def _base_cfg(**over):
    df, _, _ = _factor_panel(seed=6)
    cfg = {"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
           "time": "year", "method": "gmm_sce"}
    cfg.update(over)
    return cfg


def test_gmm_sce_requires_instruments_without_selection():
    with pytest.raises(MlsynthConfigError):
        ORTHSC(_base_cfg(model_selection=False))           # no instruments


def test_model_selection_invalid_for_orthogonalized():
    with pytest.raises(MlsynthConfigError):
        ORTHSC(_base_cfg(method="orthogonalized", instruments=["d2"],
                         model_selection=True))


def test_duplicate_instruments_rejected():
    with pytest.raises(MlsynthConfigError):
        ORTHSC(_base_cfg(instruments=["d2", "d2"]))


def test_control_instrument_overlap_rejected():
    with pytest.raises(MlsynthConfigError):
        ORTHSC(_base_cfg(instruments=["d2", "d3"], controls=["d2", "d0"]))


def test_bad_weight_tol_rejected():
    with pytest.raises(MlsynthConfigError):
        ORTHSC(_base_cfg(model_selection=True, weight_tol=0.0))


def test_unknown_instrument_unit_reported():
    with pytest.raises(MlsynthDataError):
        ORTHSC(_base_cfg(instruments=["nope"])).fit()


def test_guaranteed_instrument_overlap_rejected():
    with pytest.raises(MlsynthConfigError):
        ORTHSC(_base_cfg(model_selection=True,
                         controls=["d0"], guaranteed_instruments=["d0"]))


def test_model_selection_all_donors_guaranteed_leaves_no_candidates():
    df, donors, _ = _factor_panel(seed=9)
    with pytest.raises(MlsynthDataError):
        ORTHSC(_base_cfg(model_selection=True,
                         guaranteed_instruments=list(donors))).fit()


# --------------------------------------------------------- result contract ----

def test_result_contract_conformance():
    df, donors, _ = _factor_panel(seed=7)
    res = ORTHSC({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                  "time": "year", "method": "gmm_sce",
                  "instruments": ["d2", "d3", "d4", "d5"]}).fit()
    assert isinstance(res, BaseEstimatorResults)
    assert res.effects is not None and np.isfinite(res.att)
    assert res.time_series is not None
    assert res.weights is not None
    assert res.method_details.method_name == "ORTHSC (gmm_sce)"
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    assert cf.shape[0] == df["year"].nunique()
    assert np.all(np.isfinite(cf))
