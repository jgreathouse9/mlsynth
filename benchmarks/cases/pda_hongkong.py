"""PDA Path-A: the Hong Kong / CEPA handover study (Hsiao-Ching-Wan; Shi-Wang).

Path A (empirical, scenario: the authors' canonical application + data). The
panel data approach is validated on the Hsiao, Ching & Wan (2012) Hong Kong
study -- the effect of the 2004 Closer Economic Partnership Arrangement (CEPA)
with mainland China on Hong Kong's quarterly YoY real GDP growth (1 treated unit,
24 controls, T1 = 44). Shi & Wang (L2-relaxation, arXiv) revisit it in their
Appendix E.1: the L2-relaxation panel data approach estimates a CEPA effect of
**+2.65%** with t-statistic **8.35**, decisively rejecting the no-effect null.

This runs all three of mlsynth's PDA methods -- L2-relaxation (``l2``), the
LASSO PDA (``LASSO``) and forward selection (``fs``; Shi & Huang 2023) -- on the
shipped ``basedata/HongKong.csv`` (the HCW panel). All three recover a positive,
highly significant CEPA effect; the L2 estimate (2.48%) lands close to the
paper's 2.65%, the small gap reflecting the L2-relaxation tuning. The fit is
deterministic, so the cells below are exact re-runs.

Provenance: Shi & Wang, *"L2-Relaxation for Economic Prediction,"* Appendix E.1
(Table E.1 and the L2 headline); Hsiao, Ching & Wan (2012) for the study/data.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "HongKong.csv")


def _fit(method):
    from mlsynth import PDA
    import pandas as pd

    d = pd.read_csv(os.path.abspath(_DATA))
    res = PDA({
        "df": d, "outcome": "GDP", "treat": "Integration",
        "unitid": "Country", "time": "Time", "method": method,
        "display_graphs": False,
    }).fit()
    att = float(res.att)
    lo = res.inference.ci_lower
    se = (att - float(lo)) / 1.96 if lo is not None else float("nan")
    return att, (att / se if se else float("nan")), float(res.inference.p_value)


def run() -> dict:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atts = {}
        for method in ("l2", "LASSO", "fs"):
            key = method.lower()
            att, t, p = _fit(method)
            atts[key] = (att, p)
            out[f"{key}_ate_pct"] = att * 100.0         # GDP is decimal growth
            out[f"{key}_pvalue"] = p
        out["l2_abs_tstat"] = abs(_fit("l2")[1])
        # All methods agree the CEPA effect is positive and significant.
        out["n_methods_positive_sig"] = float(sum(
            1 for a, p in atts.values() if a > 0 and p < 0.01))
    return out


# Deterministic (convex L2 / greedy fs / fixed-grid LASSO, no RNG) => exact
# re-runs. All three PDA methods recover a positive, highly significant CEPA
# effect on Hong Kong's GDP growth; the L2-relaxation estimate (2.48%) reproduces
# the paper's +2.65% (t ~ 7.7 vs 8.35) -- the small gap is the L2 tuning. The
# LASSO and forward-selection estimates (3.3%, 3.9%) bracket it.
EXPECTED = {
    "l2_ate_pct": (2.61, 0.5),                 # paper L2: 2.65% (standardised l2)
    "lasso_ate_pct": (3.30, 0.6),
    "fs_ate_pct": (3.95, 0.7),
    "l2_pvalue": (0.0, 0.01),                  # rejects no-effect null
    "l2_abs_tstat": (7.75, 1.5),               # paper L2: 8.35
    "n_methods_positive_sig": (3.0, 0.0),      # all three agree
}
