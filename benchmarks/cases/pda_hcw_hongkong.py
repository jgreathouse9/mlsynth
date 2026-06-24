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
Kong's economic growth").

The reference side is a live captured run of the ``pampe`` R package (Vega-Bayo;
the canonical CRAN implementation of HCW's panel-data approach -- ``leaps``
best-subset + AICc + ``lm``), captured under
``benchmarks/reference/pda_hcw_hongkong/`` with its provenance pinned (R version,
package versions, data checksum). The case reads its reference numbers from that
bundle via :func:`reference_value`, so the constants in ``EXPECTED`` and the
captured run are the same object and cannot silently drift. The fit is
deterministic (best-subset + AICc + OLS, no RNG), so the captured run reproduces
HCW Tables XVI-XVII to the digit and mlsynth matches it to ~1e-6.

Provenance: Hsiao, Ching & Wan (2012), *J. Applied Econometrics* 27(5),
Tables XVI-XVII; reference implementation pampe (https://github.com/cran/pampe).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import reference_value

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")

# The ten candidate economies HCW consider (region + close associates).
_CANDS = ["China", "Indonesia", "Japan", "Korea", "Malaysia", "Philippines",
          "Singapore", "Taiwan", "Thailand", "United States"]


def _prep() -> pd.DataFrame:
    """The HCW Table XVI/XVII panel: ten candidates + Hong Kong, Time <= 43,
    sovereignty cut at 1997:Q3 (Time 18)."""
    d = pd.read_csv(os.path.abspath(_DATA))
    d = d[d["Country"].isin(["Hong Kong"] + _CANDS) & (d["Time"] <= 43)].copy()
    d["treat"] = ((d["Country"] == "Hong Kong") & (d["Time"] >= 18)).astype(int)
    return d


def _fit(d: pd.DataFrame):
    from mlsynth import PDA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDA({
            "df": d, "outcome": "GDP", "treat": "treat",
            "unitid": "Country", "time": "Time", "method": "hcw",
            "display_graphs": False,
        }).fit()


def run() -> dict:
    d = _prep()
    res = _fit(d)

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


def comparison() -> dict:
    """mlsynth ``PDA(method="hcw")`` vs the pampe R package, quantity by quantity.

    Pairs the mlsynth best-subset HCW fit against ``pampe::pampe()`` on the same
    Hong Kong panel and spec: the selected-control set size, the OLS weights on
    Japan and Taiwan, the intercept, the pre-period R^2, and the post-1997:Q3
    average treatment effect. The reference side is a live ``pampe`` run captured
    in ``benchmarks/reference/pda_hcw_hongkong/`` (the canonical CRAN
    implementation of HCW's panel-data approach), not transcribed from Tables
    XVI/XVII. Returns ``{"rows": [...], "mlsynth_call": {...},
    "reference": {...}}`` with rows ``{quantity, mlsynth, reference}``.
    """
    got = run()
    rows = [
        {"quantity": "n_selected", "mlsynth": round(got["n_selected"], 6),
         "reference": round(reference_value("pda_hcw_hongkong", "n_selected"), 6)},
        {"quantity": "weight[Japan]", "mlsynth": round(got["weight_japan"], 6),
         "reference": round(reference_value("pda_hcw_hongkong", "weight_japan"), 6)},
        {"quantity": "weight[Taiwan]", "mlsynth": round(got["weight_taiwan"], 6),
         "reference": round(reference_value("pda_hcw_hongkong", "weight_taiwan"), 6)},
        {"quantity": "intercept", "mlsynth": round(got["intercept"], 6),
         "reference": round(reference_value("pda_hcw_hongkong", "intercept"), 6)},
        {"quantity": "r2_pre", "mlsynth": round(got["r2_pre"], 6),
         "reference": round(reference_value("pda_hcw_hongkong", "r2_pre"), 6)},
        {"quantity": "att_pct", "mlsynth": round(got["att_pct"], 6),
         "reference": round(reference_value("pda_hcw_hongkong", "att_pct"), 6)},
    ]
    cfg = {"outcome": "GDP", "treat": "treat", "unitid": "Country",
           "time": "Time", "method": "hcw"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PDA", "config": cfg},
        "reference": {"impl": "R package pampe (pampe(), live run, captured)",
                      "version": "pampe 1.1.2 "
                                 "(benchmarks/reference/pda_hcw_hongkong/)"},
    }


# Deterministic (best-subset + AICc + OLS, no RNG) => exact re-runs. The
# reference values are pinned from the live captured pampe run
# (benchmarks/reference/pda_hcw_hongkong/) via reference_value, so they cannot
# drift from the captured output; the run itself reproduces HCW Tables XVI-XVII
# to the digit and mlsynth matches pampe to ~1e-6 (the weight gaps are mlsynth's
# 4-decimal donor_weights rounding).
_hk = lambda k: reference_value("pda_hcw_hongkong", k)
EXPECTED = {
    "n_selected": (_hk("n_selected"), 0.0),       # Japan, Korea, Taiwan, USA
    "selected_is_jkta_usa": (1.0, 0.0),
    "weight_japan": (_hk("weight_japan"), 0.001),  # Table XVI
    "weight_taiwan": (_hk("weight_taiwan"), 0.001),  # Table XVI
    "intercept": (_hk("intercept"), 0.001),       # Table XVI
    "r2_pre": (_hk("r2_pre"), 0.001),             # Table XVI
    "att_pct": (_hk("att_pct"), 0.01),            # Table XVII mean treatment
}
