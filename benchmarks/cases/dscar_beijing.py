"""DSCAR Path-A: Beijing air-pollution alerts (Zheng & Chen 2024).

Reproduces the empirical application of Zheng & Chen (2024), *"Dynamic synthetic
control method for evaluating treatment effects in auto-regressive processes"*
(JRSS-B): the effect of Beijing's heavy-pollution **alerts** (mandatory emission
cuts) on station-level PM2.5, with a time-varying AR(1) outcome model and exact
covariate matching via empirical likelihood.

**Orange alert** (17 Nov 2016, 94 stations, 20 treated; 48 h pre / 24 h post)
reproduces the paper **value for value**:

  ===============  ===============  ===============
  Quantity         mlsynth          paper
  ===============  ===============  ===============
  ATT              -33.78           -33.8
  reduction        24.3%            24.3%
  treated mean     105.28           105.3
  counterfactual   139.07           139.0
  ===============  ===============  ===============

**Red alert** (16 Dec 2016, 66 stations): mlsynth recovers a large negative
effect (ATT -55.7, a ~22% reduction), but the magnitude differs from the paper's
-70.4 / 26.2% -- the released code omits a per-unit pressure/humidity de-meaning
step the paper's numbers appear to use (documented on the estimator's page). The
case pins the **qualitative** red-alert finding and mlsynth's value as a
regression guard, not the paper's exact magnitude.

Path A: the orange alert is a faithful reproduction; the red alert is a
qualitative + regression pin. Deterministic.
"""
from __future__ import annotations

import os
import warnings

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "basedata")


def _fit(alert: str):
    import pandas as pd

    from mlsynth import DSCAR

    df = pd.read_csv(os.path.join(_DIR, f"beijing_pm25_{alert}_alert.csv"))
    df["treat_indicator"] = ((df["alert_if"] == 1) & (df["hour_eps"] > 48)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSCAR({
            "df": df, "outcome": "pm25", "treat": "treat_indicator",
            "unitid": "id_eps", "time": "hour_eps",
            "exog_covariates": ["WSPM", "humi", "dewp", "pres"],
            "lagged_outcome": "pm25_lag1", "display_graphs": False,
        }).fit()
    mu0 = float(res.fit.Y0_hat[48:].mean())
    mu1 = float(res.fit.Y_treated_mean[48:].mean())
    return float(res.att), abs(float(res.att_relative)), mu0, mu1


def run() -> dict:
    o_att, o_rel, o_mu0, o_mu1 = _fit("orange")
    r_att, r_rel, _, _ = _fit("red")
    return {
        # orange alert -- faithful Path-A replication
        "orange_att": o_att,
        "orange_reduction": o_rel,
        "orange_counterfactual": o_mu0,
        "orange_treated_mean": o_mu1,
        "orange_vs_paper_att": abs(o_att - (-33.8)),
        # red alert -- qualitative + regression guard
        "red_att": r_att,
        "red_reduction": r_rel,
        "red_large_negative": float(r_att < -30.0),
    }


# Deterministic (empirical-likelihood / closed-form weights, no RNG). The orange
# cells reproduce Zheng & Chen's Table value for value (ATT to 0.05 ug/m^3); the
# red cells pin mlsynth's value as a regression guard and the qualitative finding
# (large negative ATT, ~20% reduction), the paper's exact -70.4 magnitude not
# being reproducible from the released code.
EXPECTED = {
    "orange_att": (-33.78, 0.4),
    "orange_reduction": (0.243, 0.01),
    "orange_counterfactual": (139.07, 1.0),
    "orange_treated_mean": (105.28, 1.0),
    "orange_vs_paper_att": (0.02, 0.4),         # matches paper -33.8
    "red_att": (-55.7, 3.0),
    "red_reduction": (0.219, 0.03),
    "red_large_negative": (1.0, 0.0),
}
