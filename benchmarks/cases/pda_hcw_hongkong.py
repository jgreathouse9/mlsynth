"""HCW Path-A: the original Hsiao-Ching-Wan (2012) Hong Kong best-subset fit.

Path A (empirical, scenario: the authors' canonical study, predictor set, and
data). HCW evaluate the effect of the 1997 change of sovereignty on Hong Kong's
quarterly year-over-year real GDP growth, constructing the counterfactual by
unrestricted OLS on a best-subset-selected set of control economies (chosen by
AICc among ten candidates), then averaging the post-1997:Q3 gap.

This reproduces HCW (2012) Table XVI / XVII value-for-value with mlsynth's
``PDA(method="hcw")`` on the shipped ``basedata/HongKong.csv``: estimation
window 1993:Q1-1997:Q2 (T0 = 18), the ten candidate economies, AICc selection.
The selected model is {Japan, Korea, Taiwan, USA} with OLS weights
(const 0.0263, Japan -0.676, Korea -0.4323, Taiwan 0.7926, USA 0.486),
pre-period R^2 = 0.9314, and a post-period average treatment effect of -3.96%
that HCW find statistically insignificant ("no significant impact ... on Hong
Kong's economic growth"). The fit is deterministic, so the cells below are
exact re-runs and cross-validate against the ``pampe`` R package
(``leaps::regsubsets`` best-subset + AICc + ``lm``).

Provenance: Hsiao, Ching & Wan (2012), *J. Applied Econometrics* 27(5),
Tables XVI-XVII; reference implementation pampe (https://github.com/cran/pampe).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")

# The ten candidate economies HCW consider (region + close associates).
_CANDS = ["China", "Indonesia", "Japan", "Korea", "Malaysia", "Philippines",
          "Singapore", "Taiwan", "Thailand", "United States"]


def run() -> dict:
    from mlsynth import PDA

    d = pd.read_csv(os.path.abspath(_DATA))
    # HCW Table XVI/XVII window (1993:Q1-2003:Q4) and the sovereignty cut at
    # 1997:Q3 (Time 18); restrict the donor pool to the ten candidate economies.
    d = d[d["Country"].isin(["Hong Kong"] + _CANDS) & (d["Time"] <= 43)].copy()
    d["treat"] = ((d["Country"] == "Hong Kong") & (d["Time"] >= 18)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": d, "outcome": "GDP", "treat": "treat",
            "unitid": "Country", "time": "Time", "method": "hcw",
            "display_graphs": False,
        }).fit()

    fit = res.fits["hcw"]
    selected = {str(s) for s in fit.selected_donors}
    weights = {str(k): float(v) for k, v in fit.donor_weights.items()}

    y = d[d["Country"] == "Hong Kong"].sort_values("Time")["GDP"].to_numpy()
    resid = y[:18] - fit.counterfactual[:18]
    r2_pre = 1.0 - float(resid @ resid) / float(np.sum((y[:18] - y[:18].mean()) ** 2))

    return {
        "n_selected": float(len(selected)),
        "selected_is_jkta_usa": float(
            selected == {"Japan", "Korea", "Taiwan", "United States"}),
        "weight_japan": weights.get("Japan", 0.0),
        "weight_taiwan": weights.get("Taiwan", 0.0),
        "intercept": float(fit.intercept),
        "r2_pre": r2_pre,
        "att_pct": float(res.att) * 100.0,        # GDP is decimal growth
    }


# Deterministic (best-subset + AICc + OLS, no RNG) => exact re-runs, matching
# HCW Tables XVI-XVII and the pampe reference.
EXPECTED = {
    "n_selected": (4.0, 0.0),                     # Japan, Korea, Taiwan, USA
    "selected_is_jkta_usa": (1.0, 0.0),
    "weight_japan": (-0.676, 0.01),               # Table XVI
    "weight_taiwan": (0.7926, 0.01),              # Table XVI
    "intercept": (0.0263, 0.005),                 # Table XVI
    "r2_pre": (0.9314, 0.005),                    # Table XVI
    "att_pct": (-3.96, 0.5),                      # Table XVII mean treatment
}
