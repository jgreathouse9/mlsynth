"""ORTHSC empirical replication: Fry (2026) on Andersson (2019)'s carbon tax.

Path A (the paper's empirical result on the authors' data). Fry's Orthogonalized
Synthetic Control, applied to Andersson (2019)'s Swedish carbon-tax panel, finds
the tax cut transport CO2 by an average ATT of -0.29 metric tons/capita/yr, and
-- unlike placebo / conformal / cross-fitting inference -- a t-test that is
strongly significant (p = 0.00018). The control pool is Andersson's 14 donors;
the instruments are the 7 carbon/fuel-tax countries he excluded (Fry's method
uses outcomes of units excluded from the controls as instruments).

This drives mlsynth's public ``ORTHSC`` estimator and pins the ATT, the p-value,
the fixed-smoothing degrees of freedom K, and the confidence interval against
both the paper's reported numbers and the live R reference (which mlsynth's
NumPy/cvxpy port reproduces to the digit). The ATT is delta-invariant by the
orthogonalization, so the match does not depend on bit-matching the reference's
weight solver.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from mlsynth import ORTHSC

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "carbontax_fullsample_data.dta.txt")
_CONTROLS = ["Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
             "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
             "Switzerland", "United States"]
_INSTRS = ["Finland", "Germany", "Ireland", "Italy", "Netherlands", "Norway",
           "United Kingdom"]


def run() -> dict:
    df = pd.read_stata(os.path.abspath(_DATA))
    df = df.rename(columns={"CO2_transport_capita": "Y"})
    df["treat"] = ((df["country"] == "Sweden") & (df["year"] >= 1990)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ORTHSC({
            "df": df, "outcome": "Y", "treat": "treat",
            "unitid": "country", "time": "year",
            "instruments": _INSTRS, "controls": _CONTROLS,
            "display_graphs": False,
        }).fit()
    return {
        "att": float(res.att),
        "pvalue": float(res.inference.p_value),
        "smoothing_K": float(res.method_details.parameters_used["smoothing_K"]),
        "ci_lower": float(res.inference.ci_lower),
        "ci_upper": float(res.inference.ci_upper),
    }


# Targets: Fry's reported carbon-tax result (ATT -0.29, p 0.00018, K 4), matched
# to the live R reference (CI [-0.476, -0.105]). ATT/CI are deterministic; the
# p-value is deterministic at the fixed smoothing.
EXPECTED = {
    "att": (-0.29013, 0.005),
    "pvalue": (0.000183, 0.0005),
    "smoothing_K": (4.0, 0.5),
    "ci_lower": (-0.4757, 0.02),
    "ci_upper": (-0.1045, 0.02),
}
