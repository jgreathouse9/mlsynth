"""VanillaSC empirical replication of Andersson (2019)'s Swedish carbon tax.

Path A: under Andersson's own SCM specification (four predictors -- GDP per
capita, motor vehicles per capita, gasoline consumption per capita, urban
population -- averaged 1980-1989, plus lagged CO2 for 1970/1980/1989; 14 OECD
donors; Sweden treated 1990), both VanillaSC predictor-weight backends reproduce
his headline: the average 1990-2005 ATT brackets -0.29 metric tons/capita and the
2005 gap brackets -0.35, with a tight pre-treatment fit.

The paper specification is what makes the two backends agree; in particular the
three lagged-CO2 predictors anchor the pre-treatment fit. The ``pop_density``
column shipped in the data is not one of Andersson's predictors and is excluded.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_DATA = pathlib.Path(__file__).resolve().parents[2] / "basedata" / "carbontax_data.dta"
_PREDICTORS = ["GDP_per_capita", "vehicles_capita", "gas_cons_capita", "urban_pop"]
_LAGS = (1970, 1980, 1989)
PAPER_ATT = -0.29
PAPER_GAP_2005 = -0.35

pytestmark = pytest.mark.skipif(not _DATA.exists(), reason="carbon tax data absent")


def _panel():
    df = pd.read_stata(_DATA)
    for yr in _LAGS:
        lag = df[df.year == yr].set_index("country")["CO2_transport_capita"]
        df[f"co2_{yr}"] = df.country.map(lag)
    df["treat"] = ((df.country == "Sweden") & (df.year >= 1990)).astype(int)
    return df


def _fit(backend, covariates=None):
    covs = covariates if covariates is not None else (
        _PREDICTORS + [f"co2_{yr}" for yr in _LAGS])
    windows = {c: (1980, 1989) for c in covs}
    cfg = {"df": _panel(), "outcome": "CO2_transport_capita", "treat": "treat",
           "unitid": "country", "time": "year", "backend": backend,
           "covariates": covs, "covariate_windows": windows, "display_graphs": False}
    if backend == "mscmt":
        cfg.update(seed=1, mscmt_maxiter=300, mscmt_popsize=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


def _gap_2005(res):
    obs = np.asarray(res.time_series.observed_outcome, float)
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    return float(obs[-1] - cf[-1])


def _pre_rmse(res):
    obs = np.asarray(res.time_series.observed_outcome, float)
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    yr = np.asarray(res.time_series.time_periods)
    pre = yr < 1990
    return float(np.sqrt(np.mean((obs[pre] - cf[pre]) ** 2)))


@pytest.fixture(scope="module")
def malo():
    return _fit("malo")


@pytest.fixture(scope="module")
def mscmt():
    return _fit("mscmt")


# ----------------------------------------------------------------------
# both backends reproduce Andersson's headline under the paper spec
# ----------------------------------------------------------------------
def test_malo_reproduces_paper_att(malo):
    assert malo.att == pytest.approx(PAPER_ATT, abs=0.04)
    assert _gap_2005(malo) == pytest.approx(PAPER_GAP_2005, abs=0.05)


def test_mscmt_reproduces_paper_att(mscmt):
    assert mscmt.att == pytest.approx(PAPER_ATT, abs=0.04)
    assert _gap_2005(mscmt) == pytest.approx(PAPER_GAP_2005, abs=0.06)


def test_tight_pretreatment_fit(malo, mscmt):
    # the lagged-outcome predictors anchor the pre-period fit
    assert _pre_rmse(malo) < 0.06
    assert _pre_rmse(mscmt) < 0.06


def test_denmark_led_synthetic(malo, mscmt):
    for res in (malo, mscmt):
        top = max(res.weights.donor_weights.items(), key=lambda kv: kv[1])[0]
        assert top == "Denmark"


def test_backends_agree_under_paper_spec(malo, mscmt):
    # the whole point of the paper spec: the two V searches land in the same place
    assert abs(malo.att - mscmt.att) < 0.05


def test_lagged_outcomes_anchor_the_fit():
    # dropping Andersson's three lagged-CO2 predictors loosens the pre-fit
    # (they pin the 1970/1980/1989 outcome levels); with them the fit is tight
    no_lags = _fit("mscmt", covariates=_PREDICTORS)
    with_lags = _fit("mscmt")
    assert _pre_rmse(with_lags) < _pre_rmse(no_lags)
