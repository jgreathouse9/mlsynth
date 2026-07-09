"""Tests for BEAST -- immunized doubly-robust synthetic control (Bléhaut 2021).

Cross-validated against the authors' R on the Proposition 99 panel
(``basedata/augmented_cali_long.csv``): the immunized ATT path matches to ~0.02
packs. The over-rich covariate regime, where the covariate balancing degenerates
(``sum(W) != 1``), is rejected by the balance guard -- the property that
distinguishes BEAST's operating envelope.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import BEAST
from mlsynth.config_models import BaseEstimatorResults, BEASTConfig
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError

_DATA = pathlib.Path(__file__).resolve().parents[2] / "basedata" / "augmented_cali_long.csv"
_COVS = ["loginc", "p_cig", "pct15-24", "pc_beer"]
_LAGS = [1975, 1980, 1988]

pytestmark = pytest.mark.skipif(not _DATA.exists(), reason="Prop 99 covariate panel absent")


def _panel():
    d = pd.read_csv(_DATA)
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    return d


def _fit(covariates=_COVS, lags=_LAGS, **kw):
    cfg = {"df": _panel(), "outcome": "cigsale", "treat": "treated",
           "unitid": "state", "time": "year", "covariates": covariates,
           "outcome_lags": lags, "display_graphs": False}
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BEAST(cfg).fit()


@pytest.fixture(scope="module")
def res():
    return _fit()


# ----------------------------------------------------------------------
# smoke + contract
# ----------------------------------------------------------------------
def test_returns_standard_results(res):
    assert isinstance(res, BaseEstimatorResults)
    assert np.isfinite(res.att)
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    obs = np.asarray(res.time_series.observed_outcome, float)
    assert cf.shape == obs.shape == (31,)               # 1970-2000


# ----------------------------------------------------------------------
# cross-validation vs the authors' R
# ----------------------------------------------------------------------
def test_reproduces_r_att(res):
    # authors' R: mean post-1989 ATT -22.44, -31.51 by 2000
    assert res.att == pytest.approx(-22.44, abs=0.2)
    tau = np.asarray(res.inference.details["tau"], float)
    assert tau[-1] == pytest.approx(-31.51, abs=0.2)


def test_valid_balancing_and_sparse(res):
    det = res.inference.details
    assert det["sum_weights"] == pytest.approx(1.0, abs=0.02)   # valid SC
    assert det["n_selected"] <= 3                                # sparse selection
    # donor weights (dense exponential tilt) sum to ~1, Utah-led classic Prop 99
    top = max(res.weights.donor_weights.items(), key=lambda kv: kv[1])[0]
    assert top == "Utah"


def test_att_interval_excludes_zero(res):
    assert res.inference.ci_upper < 0.0
    assert res.inference.ci_lower <= res.att <= res.inference.ci_upper


# ----------------------------------------------------------------------
# the balance guard: over-rich covariate set is rejected
# ----------------------------------------------------------------------
def test_oversaturated_regime_is_rejected():
    # a large, collinear covariate set (p ~ n) degenerates the balancing;
    # BEAST must refuse rather than return a garbage synthetic control
    d = pd.read_csv(_DATA)
    pre = d[d.year < 1989]
    big = [c for c in d.columns
           if c not in {"state", "year", "treated", "cigsale"}
           and d[c].dtype.kind in "if"
           and pre.groupby("state")[c].mean().notna().all()
           and pre.groupby("state")[c].mean().std(ddof=1) > 0][:30]
    with pytest.raises(MlsynthEstimationError):
        _fit(covariates=big, lags=None)


# ----------------------------------------------------------------------
# config validation
# ----------------------------------------------------------------------
def test_config_requires_covariates():
    with pytest.raises((MlsynthConfigError, ValueError)):
        BEASTConfig(df=_panel(), outcome="cigsale", treat="treated",
                    unitid="state", time="year", covariates=[], display_graphs=False)


def test_bad_outcome_lag_rejected():
    with pytest.raises((MlsynthConfigError, MlsynthEstimationError)):
        _fit(lags=[1999])                                # a post-treatment year


def test_non_immunized_runs(res):
    plug = _fit(immunity=False)
    # the plug-in (mu=0) also gives a strongly negative ATT, near the immunized one
    assert plug.att < -15.0


# ----------------------------------------------------------------------
# plotting smoke
# ----------------------------------------------------------------------
def test_plot_runs(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    r = _fit(display_graphs=True)
    assert np.isfinite(r.att)
