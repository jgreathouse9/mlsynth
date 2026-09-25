"""HCW Path-A: the CEPA 2004:Q1 result, the paper's headline finding.

Hsiao, Ching & Wan (2012) study two events in the same Hong Kong panel. The
change of sovereignty is the sibling case ``pda_hcw_hongkong``, and they find
no effect from it. This is the other: the Closer Economic Partnership
Arrangement, implemented 2004:Q1, which they find raised Hong Kong's real GDP
growth rate "by more than 4% compared to the growth rate had there been no
CEPA agreement".

It is the harder test of the two. For the sovereignty event HCW restrict the
donor pool to ten regional economies because "there are only 18 observations
between 1993:Q1 and 1997:Q2"; here they write that "since we now have more
degrees of freedom, we can use the model selection strategy discussed in
Section 5", and AICc chooses from all 24 countries in the panel. The
best-subset search therefore runs over a pool more than twice as wide, and
lands on six countries none of which are in the regional set the sibling case
selects from:

  ==================  ====================================================
  quantity            HCW (2012), pp. 725-728
  ==================  ====================================================
  AICc group          Austria, Italy, Korea, Mexico, Norway, Singapore
  average effect      4.03%
  "standard error"    0.016
  t-statistic         2.5134
  pre-period R^2      "above 0.93"
  ==================  ====================================================

mlsynth's ``PDA(method="hcw")`` recovers the selected group exactly and the
average effect to four decimals, and the captured reference run agrees with
both.

A note the numbers force. HCW's "standard error of 0.016" is the standard
deviation of the per-period treatment effects, not the standard error of their
mean: 0.040326 / 0.016045 = 2.5134 reproduces their published t exactly, where
dividing by a standard error of the mean gives about 10. The two are different
statistics and the case pins both, so neither is mistaken for the other.
Reading 0.016 as a standard error of the ATT understates the significance of
this result several-fold.

The standard error mlsynth reports is neither: it is the root of a Bartlett
HAC long-run variance of the effect series over ``T2``, which is the right
object for an average when the per-period effects are serially correlated (and
here they are, at a first-order autocorrelation of 0.27). The case runs with
``lrvar_lag=2``, the fsPDA truncation ``floor(T2 ** 0.25)``, and the reference
computes the same quantity under the same convention -- every autocovariance
divided by ``n``, which is R's ``acf(type = "covariance")`` -- so the two sides
are the same estimator and agree to six decimals. The default prewhitened
Newey-West path is left unpinned because the reference engine does not
implement it.

This panel's shipped ``Integration`` column encodes this event, not the
sovereignty one -- it switches at ``Time`` 44 (2004:Q1) -- so this is what a
reader gets from ``basedata/HongKong.csv`` without overriding anything.

Path: cross-validation against ``leaps::regsubsets`` + AICc + ``lm``, the
engine the CRAN package pampe wraps, captured under
``benchmarks/reference/pda_hcw_cepa/``. pampe is archived on CRAN and was not
installable here, so unlike the sibling bundle this calls the engine directly;
the sovereignty spec run through the same script reproduces the sibling's
numbers. Deterministic (best-subset + AICc + OLS, no RNG).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import reference_value

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")
_T0 = 44          # 1993:Q1-2003:Q4; CEPA implemented 2004:Q1
_HCW_GROUP = {"Austria", "Italy", "Korea", "Mexico", "Norway", "Singapore"}


def _fit():
    from mlsynth import PDA

    d = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return d, PDA({
            "df": d, "outcome": "GDP", "treat": "Integration",
            "unitid": "Country", "time": "Time", "method": "hcw",
            "display_graphs": False,
            # The fsPDA fixed-lag Bartlett branch, so the standard error is a
            # quantity the R reference computes the same way.
            "lrvar_lag": 2,
        }).fit()


def run() -> dict:
    d, res = _fit()
    fit = res.fits["hcw"]
    selected = {str(s) for s in fit.selected_donors}
    weights = {str(k): float(v) for k, v in fit.donor_weights.items()}

    gap = np.asarray(res.time_series.estimated_gap, dtype=float).ravel()
    effect = gap[_T0:]
    sd_effect = float(np.std(effect, ddof=1))

    y = d[d["Country"] == "Hong Kong"].sort_values("Time")["GDP"].to_numpy()
    resid = y[:_T0] - fit.counterfactual[:_T0]
    r2_pre = 1.0 - float(resid @ resid) / float(
        np.sum((y[:_T0] - y[:_T0].mean()) ** 2))

    return {
        "n_selected": float(len(selected)),
        "selected_is_hcw_aicc_group": float(selected == _HCW_GROUP),
        "weight_austria": weights.get("Austria", 0.0),
        "weight_singapore": weights.get("Singapore", 0.0),
        "r2_pre": r2_pre,
        "att_pct": float(res.att) * 100.0,
        # HCW's "standard error", which is sd(effect) -- see the module note.
        "sd_effect": sd_effect,
        "t_hcw": float(res.att) / sd_effect,
        # What mlsynth reports under lrvar_lag=2: the Bartlett HAC root.
        "se_hac": float(res.inference.standard_error),
    }


# Deterministic (best-subset + AICc + OLS, no RNG) => exact re-runs. Reference
# values are pinned from the captured run via reference_value and cannot drift
# from it; that run reproduces HCW pp. 725-728 (4.03%, t = 2.5134, the six-country
# AICc group, R^2 above 0.93).
_c = lambda k: reference_value("pda_hcw_cepa", k)
EXPECTED = {
    "n_selected": (_c("n_selected"), 0.0),
    "selected_is_hcw_aicc_group": (1.0, 0.0),
    "weight_austria": (_c("weight_austria"), 0.001),
    "weight_singapore": (_c("weight_singapore"), 0.001),
    "r2_pre": (_c("r2_pre"), 0.001),
    "att_pct": (_c("att_pct"), 0.01),
    "sd_effect": (_c("sd_effect"), 0.0005),
    "t_hcw": (_c("t_hcw"), 0.01),
    "se_hac": (_c("se_hac"), 1e-05),
}
