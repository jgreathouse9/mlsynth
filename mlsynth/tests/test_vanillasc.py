"""Tests for the VanillaSC estimator and its bilevel engine."""

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
    assert r.M2_upper > r.M2_lower
    assert "att" in r.metadata and r.metadata["att_lower"] <= r.metadata["att_upper"]


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


def test_vanillasc_accepts_config_object():
    df = _panel()
    cfg = VanillaSCConfig(df=df, outcome="y", treat="treated", unitid="unit",
                          time="time", backend="outcome-only",
                          inference=False, display_graphs=False)
    res = VanillaSC(cfg).fit()
    assert np.isfinite(res.effects.att)
