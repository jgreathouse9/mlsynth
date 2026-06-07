"""Tests for the VanillaSC estimator and its bilevel engine."""

import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.config_models import VanillaSCConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.vanillasc_helpers import BilevelSCM, BilevelSCMResult


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def _outcome_problem(seed=0, T0=12, J=6):
    rng = np.random.default_rng(seed)
    Y0_pre = rng.normal(5, 1, size=(T0, J))
    w_true = np.zeros(J); w_true[[1, 3]] = [0.6, 0.4]
    y_pre = Y0_pre @ w_true + 0.001 * rng.normal(size=T0)
    return y_pre, Y0_pre, w_true


def test_engine_outcome_only_recovers_convex_combo():
    y, Y0, w_true = _outcome_problem()
    res = BilevelSCM("outcome-only").fit(y, Y0)
    assert isinstance(res, BilevelSCMResult)
    assert res.backend == "outcome-only"
    assert res.W.sum() == pytest.approx(1.0, abs=1e-4)
    assert np.all(res.W >= -1e-9)
    np.testing.assert_allclose(res.W, w_true, atol=5e-2)
    assert res.V is None and res.v_agreement is None


def test_engine_auto_is_outcome_only_without_covariates():
    y, Y0, _ = _outcome_problem(1)
    assert BilevelSCM("auto").fit(y, Y0).backend == "outcome-only"


def test_engine_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        BilevelSCM("bogus")


def test_engine_malo_requires_covariates():
    y, Y0, _ = _outcome_problem(2)
    with pytest.raises(ValueError, match="needs covariates"):
        BilevelSCM("malo").fit(y, Y0)


def _cov_problem(seed=3, T0=12, J=6, P=4):
    rng = np.random.default_rng(seed)
    Y0_pre = rng.normal(5, 1, size=(T0, J))
    X0 = rng.normal(size=(P, J))
    w = np.zeros(J); w[[0, 2]] = [0.7, 0.3]
    y_pre = Y0_pre @ w + 0.01 * rng.normal(size=T0)
    X1 = X0 @ w
    return y_pre, Y0_pre, X1, X0


@pytest.mark.parametrize("backend", ["mscmt", "malo", "penalized"])
def test_engine_covariate_backends_run(backend):
    y, Y0, X1, X0 = _cov_problem()
    kw = dict(maxiter=20, popsize=8, seed=0) if backend == "mscmt" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = BilevelSCM(backend, **kw).fit(y, Y0, X1=X1, X0=X0)
    assert res.W.sum() == pytest.approx(1.0, abs=1e-3)
    assert res.backend == backend
    assert np.isfinite(res.pre_rmspe)


def test_engine_mscmt_reports_v_agreement():
    y, Y0, X1, X0 = _cov_problem(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = BilevelSCM("mscmt", maxiter=20, popsize=8).fit(y, Y0, X1=X1, X0=X0)
    assert res.V is not None
    # v_agreement is reported when the canonicalisation certifies; None otherwise.
    assert res.v_agreement is None or res.v_agreement >= 0.0


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
def _panel(n_units=8, T=20, t0=14, seed=0):
    """Long panel: unit 'u0' treated from period t0, with a planted effect."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(T, 2))
    loads = rng.uniform(0.2, 1.0, size=(n_units, 2))
    rows = []
    for i in range(n_units):
        base = factor @ loads[i] * 5 + 20 + rng.normal(0, 0.1, size=T)
        x1 = loads[i, 0] + rng.normal(0, 0.01, size=T)
        for t in range(T):
            y = base[t]
            treated = 1 if (i == 0 and t >= t0) else 0
            if i == 0 and t >= t0:
                y -= 3.0  # treatment effect
            rows.append((f"u{i}", t, y, treated, x1[t]))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treated", "x1"])


def test_vanillasc_outcome_only_end_to_end():
    df = _panel()
    res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                     "unitid": "unit", "time": "time",
                     "backend": "outcome-only", "display_graphs": False}).fit()
    assert res.effects.att is not None and np.isfinite(res.effects.att)
    assert res.fit_diagnostics.rmse_pre is not None
    w = res.weights.donor_weights
    assert abs(sum(w.values()) - 1.0) < 1e-3
    assert res.inference is not None and 0.0 <= res.inference.p_value <= 1.0
    ts = res.time_series
    assert ts.observed_outcome.shape == ts.counterfactual_outcome.shape


def test_vanillasc_penalized_and_covariates():
    df = _panel(seed=2)
    res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                     "unitid": "unit", "time": "time",
                     "covariates": ["x1"], "backend": "penalized",
                     "inference": False, "display_graphs": False}).fit()
    assert np.isfinite(res.effects.att)
    assert res.method_details.method_name == "VanillaSC[penalized]"
    # predictor weights surfaced for the covariate path
    assert "predictor_weights" in res.weights.summary_stats


def test_vanillasc_covariate_balance_reported():
    df = _panel(seed=6)
    res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                     "unitid": "unit", "time": "time",
                     "covariates": ["x1"], "backend": "penalized",
                     "inference": False, "display_graphs": False}).fit()
    bal = res.additional_outputs["covariate_balance"]
    assert bal is not None
    assert bal["predictors"] == ["x1"]
    for key in ("treated", "synthetic", "donor_average"):
        assert len(bal[key]) == 1 and np.isfinite(bal[key][0])
    gaps = bal["mean_abs_pct_gap"]
    assert np.isfinite(gaps["synthetic"]) and np.isfinite(gaps["donor_average"])
    # synthetic value is the donor-weighted predictor mean (over the pre-period,
    # matching the pipeline's covariate averaging window)
    pre = res.additional_outputs["pre_periods"]
    pre_labels = sorted(df["time"].unique())[:pre]
    w = res.weights.donor_weights
    donors = (df[(df.unit != "u0") & df.time.isin(pre_labels)]
              .groupby("unit")["x1"].mean())
    wv = np.array([w.get(u, 0.0) for u in donors.index])
    expected = float((donors.values * wv).sum() / wv.sum())
    assert bal["synthetic"][0] == pytest.approx(expected, rel=1e-6)


def test_vanillasc_covariate_balance_none_without_covariates():
    df = _panel(seed=6)
    res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                     "unitid": "unit", "time": "time",
                     "backend": "outcome-only", "inference": False,
                     "display_graphs": False}).fit()
    assert res.additional_outputs["covariate_balance"] is None


def test_vanillasc_config_dict_and_validation():
    df = _panel()
    # missing outcome column -> data error from the base config validator
    with pytest.raises((MlsynthConfigError, MlsynthDataError)):
        VanillaSC({"df": df, "outcome": "nope", "treat": "treated",
                   "unitid": "unit", "time": "time"}).fit()


def test_vanillasc_integer_unit_ids_with_covariates():
    # Regression: integer unit ids must not break the covariate groupby lookup.
    df = _panel(seed=4).copy()
    df["unit"] = df["unit"].str.replace("u", "", regex=False).astype(int)
    res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                     "unitid": "unit", "time": "time",
                     "covariates": ["x1"], "backend": "penalized",
                     "inference": False, "display_graphs": False}).fit()
    assert np.isfinite(res.effects.att)
    assert res.weights.summary_stats.get("predictor_weights") is not None


def test_scpi_intervals_module():
    from mlsynth.utils.vanillasc_helpers.scpi import scpi_intervals
    from mlsynth.utils.fscm_helpers.bilevel.simplex import simplex_lstsq
    rng = np.random.default_rng(5)
    T, T0, J = 30, 22, 6
    Y0 = rng.normal(5, 1, size=(T, J))
    w = np.zeros(J); w[[1, 4]] = [0.6, 0.4]
    y = Y0 @ w + 0.05 * rng.normal(size=T)
    y[T0:] -= 1.0                                     # planted effect
    W = simplex_lstsq(Y0[:T0], y[:T0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = scpi_intervals(y, Y0, T0, W, sims=40, u_alpha=0.1, e_alpha=0.1, seed=0)
    assert r.tau.shape == (T - T0,)
    assert np.all(r.lower <= r.tau + 1e-6) and np.all(r.tau - 1e-6 <= r.upper)
    assert np.all(np.isfinite(r.lower)) and np.all(np.isfinite(r.upper))
    assert np.all(r.M1_upper >= r.M1_lower - 1e-9)
    assert np.all(np.asarray(r.M2_upper) >= np.asarray(r.M2_lower) - 1e-9)
    assert np.asarray(r.M2_lower).shape == (T - T0,)
    assert r.metadata["rho"] <= 0.2 + 1e-12        # scpi rho_max cap
    assert "att" in r.metadata and r.metadata["att_lower"] <= r.metadata["att_upper"]


@pytest.mark.skipif(
    importlib.util.find_spec("scpi_pkg") is None,
    reason="reference scpi_pkg not installed",
)
def test_scpi_matches_reference_package():
    """Our from-scratch SCPI reproduces ``scpi_pkg``'s CI to MC error.

    Validates the (MIT) re-derivation against the (GPL) reference on the
    Proposition 99 California panel without importing it at runtime.
    """
    import cvxpy as cp
    from mlsynth.utils.vanillasc_helpers.scpi import scpi_intervals
    base = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")
    csv = os.path.join(base, "augmented_cali_long.csv")
    if not os.path.exists(csv):
        pytest.skip("California base data not available")
    d = pd.read_csv(csv)[["state", "year", "cigsale"]]
    treated = "California"
    donors = [s for s in sorted(d.state.unique()) if s != treated]
    pre = np.arange(1970, 1989); post = np.arange(1989, 2001)
    cig = d.pivot_table(index="year", columns="state", values="cigsale")
    yrs = np.concatenate([pre, post])
    y = cig.loc[yrs, treated].to_numpy()
    Y0 = cig.loc[yrs, donors].to_numpy()
    T0 = len(pre)
    w = cp.Variable(len(donors))
    cp.Problem(cp.Minimize(cp.sum_squares(y[:T0] - Y0[:T0] @ w)),
               [w >= 0, cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    W = np.clip(np.asarray(w.value).ravel(), 0, None); W = W / W.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = scpi_intervals(y, Y0, T0, W, sims=600, u_alpha=0.05,
                              e_alpha=0.05, e_method="gaussian", seed=0)
    from scpi_pkg.scdata import scdata
    from scpi_pkg.scpi import scpi
    prep = scdata(df=d[d.state.isin([treated, *donors])].copy(), id_var="state",
                  time_var="year", outcome_var="cigsale", period_pre=pre,
                  period_post=post, unit_tr=treated, unit_co=donors,
                  features=["cigsale"], constant=False, cointegrated_data=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = scpi(prep, sims=600, w_constr={"name": "simplex"}, u_missp=True,
                   u_order=1, u_lags=0, e_method="gaussian", e_order=1, e_lags=0,
                   u_alpha=0.05, e_alpha=0.05, cores=1, verbose=False)
    ci = np.asarray(ref.CI_all_gaussian, dtype=float)
    obs = y[T0:]
    ref_lo = obs - ci[:, 1]; ref_hi = obs - ci[:, 0]   # effect PI from cf band
    # widths span ~25-45 units; MC error across independent RNG streams is ~1-2
    assert np.max(np.abs(ours.lower - ref_lo)) < 2.5
    assert np.max(np.abs(ours.upper - ref_hi)) < 2.5


def test_vanillasc_scpi_inference_end_to_end():
    df = _panel(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                         "unitid": "unit", "time": "time",
                         "backend": "outcome-only", "inference": "scpi",
                         "alpha": 0.1, "scpi_sims": 40,
                         "display_graphs": False}).fit()
    inf = res.inference
    assert inf is not None
    assert "scpi" in inf.method.lower()
    assert inf.ci_lower is not None and inf.ci_lower <= inf.ci_upper
    det = inf.details
    assert len(det["periods"]) == len(det["pi_lower"]) == len(det["tau"])
    assert np.all(np.asarray(det["pi_lower"]) <= np.asarray(det["pi_upper"]) + 1e-6)


def test_lto_helpers_match_paper():
    """F / c(N, alpha) reproduce Lei-Sudijono (2025) reported values."""
    from mlsynth.utils.vanillasc_helpers.lto import (
        lto_powered_offset, lto_type_i_bound,
    )
    # paper: c(39, 0.02)=0.006, c(39, 0.05)=0.002, c(17, 0.05)=0.0125
    assert abs(lto_powered_offset(39, 0.02) - 0.006) < 5e-4
    assert abs(lto_powered_offset(39, 0.05) - 0.002) < 5e-4
    assert abs(lto_powered_offset(17, 0.05) - 0.0125) < 5e-4
    # discrete Type-I bound is floor(N f)/N and >= alpha by construction
    assert lto_type_i_bound(39, 0.05) >= 0.05
    assert 0.0 < lto_type_i_bound(17, 0.05) < 0.1


def test_lto_placebo_module():
    from mlsynth.utils.vanillasc_helpers.lto import lto_placebo_test
    from mlsynth.utils.vanillasc_helpers import BilevelSCM
    rng = np.random.default_rng(3)
    T, T0, J = 24, 16, 7
    Y0 = rng.normal(5, 1, size=(T, J))
    w = np.zeros(J); w[[0, 2, 5]] = [0.5, 0.3, 0.2]
    y = Y0 @ w + 0.05 * rng.normal(size=T)
    y[T0:] -= 2.0                                    # planted effect
    eng = BilevelSCM("outcome-only", seed=0)
    out = lto_placebo_test(eng, y, Y0, T0, alpha=0.05, seed=0)
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["n_pairs"] == J * (J - 1) // 2        # C(J, 2) pairs
    assert out["N"] == J + 1
    assert out["p_powered"] <= out["p_value"] + 1e-12
    assert out["c"] > 0.0
    assert out["type_i_bound"] >= 0.05
    assert out["reject"] == (out["p_powered"] <= 0.05)


def test_lto_requires_three_donors():
    from mlsynth.utils.vanillasc_helpers.lto import lto_placebo_test
    from mlsynth.utils.vanillasc_helpers import BilevelSCM
    rng = np.random.default_rng(0)
    Y0 = rng.normal(5, 1, size=(12, 2))
    y = Y0 @ np.array([0.6, 0.4]) + 0.05 * rng.normal(size=12)
    with pytest.raises(ValueError):
        lto_placebo_test(BilevelSCM("outcome-only"), y, Y0, 8)


def test_vanillasc_lto_inference_end_to_end():
    df = _panel(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": df, "outcome": "y", "treat": "treated",
                         "unitid": "unit", "time": "time",
                         "backend": "outcome-only", "inference": "lto",
                         "alpha": 0.05, "display_graphs": False}).fit()
    inf = res.inference
    assert inf is not None
    assert "leave-two-out" in inf.method.lower()
    assert 0.0 <= inf.p_value <= 1.0
    det = inf.details
    assert det["n_pairs"] == 7 * 6 // 2              # 7 donors -> C(7,2)
    assert det["n_units"] == 8
    assert det["p_powered"] <= inf.p_value + 1e-12
    assert isinstance(det["reject_at_alpha"], bool)


def test_vanillasc_accepts_config_object():
    df = _panel()
    cfg = VanillaSCConfig(df=df, outcome="y", treat="treated", unitid="unit",
                          time="time", backend="outcome-only",
                          inference=False, display_graphs=False)
    res = VanillaSC(cfg).fit()
    assert np.isfinite(res.effects.att)
