"""VanillaSC empirical replication: Andersson (2019)'s Swedish carbon tax.

Path A (the paper's empirical result on the authors' data). Andersson (2019),
*"Carbon Taxes and CO2 Emissions: Sweden as a Case Study"* (AEJ: Economic Policy
11(4)), finds that Sweden's 1990 carbon tax (and the concurrent VAT on transport
fuel) cut per-capita transport CO2 by an average of 0.29 metric tons over
1990-2005, and by 0.35 tons (12.5 percent) in the final year 2005.

This drives mlsynth's ``VanillaSC`` under the paper's own synthetic-control
specification (Section II.B): outcome ``CO2_transport_capita``, Sweden treated
from 1990, Andersson's 14 OECD donors, and the paper's predictors -- GDP per
capita, motor vehicles per capita, gasoline consumption per capita and urban
population averaged over 1980-1989, plus three lagged outcomes (CO2 in 1970,
1980 and 1989). It checks BOTH predictor-weight backends:

* ``backend="malo"`` -- the Malo et al. (2024) corner search (deterministic);
* ``backend="mscmt"`` -- the global differential-evolution V search (seeded).

Both reproduce Andersson's headline: the average ATT brackets -0.29 and the 2005
gap brackets -0.35, with a tight pre-treatment fit (RMSE ~0.034). The lagged-CO2
predictors matter -- they pin the 1970/1980/1989 outcome levels, tightening the
pre-treatment fit and bringing the two V searches into agreement; the ``mscmt``
differential-evolution search is stable at seed 1 across the default budget.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "carbontax_data.dta")
_REF = load_reference("vanillasc_carbontax")["values"]
_PREDICTORS = ["GDP_per_capita", "vehicles_capita", "gas_cons_capita", "urban_pop"]
_LAGS = (1970, 1980, 1989)


def _panel() -> pd.DataFrame:
    df = pd.read_stata(os.path.abspath(_DATA))
    for yr in _LAGS:                                   # Andersson's lagged-CO2 predictors
        lag = df[df.year == yr].set_index("country")["CO2_transport_capita"]
        df[f"co2_{yr}"] = df.country.map(lag)
    df["treat"] = ((df.country == "Sweden") & (df.year >= 1990)).astype(int)
    return df


def _covs_and_windows():
    covs = _PREDICTORS + [f"co2_{yr}" for yr in _LAGS]
    windows = {c: (1980, 1989) for c in covs}         # paper's 1980-1989 window
    return covs, windows


def _fit(backend: str):
    from mlsynth import VanillaSC

    df = _panel()
    covs, windows = _covs_and_windows()
    cfg = {"df": df, "outcome": "CO2_transport_capita", "treat": "treat",
           "unitid": "country", "time": "year", "backend": backend,
           "covariates": covs, "covariate_windows": windows,
           "display_graphs": False}
    if backend == "mscmt":
        cfg.update(seed=1, mscmt_maxiter=300, mscmt_popsize=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


def _summary(res):
    obs = np.asarray(res.time_series.observed_outcome, float)
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    yr = np.asarray(res.time_series.time_periods)
    pre = yr < 1990
    top = max(res.weights.donor_weights.items(), key=lambda kv: kv[1])[0]
    return {"att": float(res.att),
            "gap_2005": float(obs[-1] - cf[-1]),
            "pre_rmse": float(np.sqrt(np.mean((obs[pre] - cf[pre]) ** 2))),
            "top_donor": top}


def run() -> dict:
    malo = _summary(_fit("malo"))
    mscmt = _summary(_fit("mscmt"))
    paper_att = _REF["paper_att"]
    paper_gap = _REF["paper_gap_2005"]
    return {
        "malo_att": malo["att"],
        "mscmt_att": mscmt["att"],
        "malo_gap_2005": malo["gap_2005"],
        "mscmt_gap_2005": mscmt["gap_2005"],
        "malo_pre_rmse": malo["pre_rmse"],
        "mscmt_pre_rmse": mscmt["pre_rmse"],
        # both backends land within 0.04 tCO2/capita of Andersson's -0.29 ATT
        "both_att_near_paper": float(abs(malo["att"] - paper_att) < 0.04
                                     and abs(mscmt["att"] - paper_att) < 0.04),
        # and near his -0.35 gap in 2005
        "both_gap_near_paper": float(abs(malo["gap_2005"] - paper_gap) < 0.06
                                     and abs(mscmt["gap_2005"] - paper_gap) < 0.06),
        # the familiar Denmark-led Swedish synthetic
        "denmark_top_both": float(malo["top_donor"] == "Denmark"
                                  and mscmt["top_donor"] == "Denmark"),
    }


# Path A (scenario: full repo / paper). mlsynth's VanillaSC reproduces Andersson
# (2019)'s carbon-tax ATT under his own SCM specification with both the malo and
# mscmt predictor-weight backends: average ATT -0.279 (malo) / -0.297 (mscmt) vs
# the paper's -0.29, and a 2005 gap of -0.349 / -0.378 vs the paper's -0.35, with
# pre-treatment RMSE ~0.034. mscmt is seeded (DE, seed 1); tolerances absorb it.
EXPECTED = {
    "malo_att": (-0.2793, 0.03),
    "mscmt_att": (-0.2966, 0.03),
    "malo_gap_2005": (-0.3492, 0.05),
    "mscmt_gap_2005": (-0.3784, 0.06),
    "malo_pre_rmse": (0.0343, 0.02),
    "mscmt_pre_rmse": (0.0347, 0.02),
    "both_att_near_paper": (1.0, 0.0),
    "both_gap_near_paper": (1.0, 0.0),
    "denmark_top_both": (1.0, 0.0),
}


def comparison() -> dict:
    """mlsynth VanillaSC (both backends) vs Andersson (2019)'s reported numbers."""
    malo = _summary(_fit("malo"))
    mscmt = _summary(_fit("mscmt"))
    rows = [
        {"quantity": "ATT (malo)", "mlsynth": round(malo["att"], 4),
         "reference": _REF["paper_att"]},
        {"quantity": "ATT (mscmt)", "mlsynth": round(mscmt["att"], 4),
         "reference": _REF["paper_att"]},
        {"quantity": "gap 2005 (malo)", "mlsynth": round(malo["gap_2005"], 4),
         "reference": _REF["paper_gap_2005"]},
        {"quantity": "gap 2005 (mscmt)", "mlsynth": round(mscmt["gap_2005"], 4),
         "reference": _REF["paper_gap_2005"]},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC",
                         "config": {"backend": "malo | mscmt", "outcome": "CO2_transport_capita",
                                    "covariates": _PREDICTORS + [f"co2_{y}" for y in _LAGS],
                                    "covariate_windows": "1980-1989", "treat": "treat",
                                    "unitid": "country", "time": "year"}},
        "reference": {"impl": "Andersson (2019) AEJ:EP 11(4), Section III reported values",
                      "version": "paper"},
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
