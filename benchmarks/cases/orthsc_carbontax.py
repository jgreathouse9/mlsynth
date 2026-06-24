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
NumPy/cvxpy port reproduces to the digit). The reference side is a live captured
run of Fry's own R code (github.com/JosephPatrickFry/OrthogonalizedSyntheticControl)
on Andersson's data, recorded in ``benchmarks/reference/orthsc_carbontax/`` and
read here via :func:`reference_value` -- the EXPECTED constants and the captured
run are the same object and cannot silently drift. The ATT is delta-invariant by
the orthogonalization, so the match does not depend on bit-matching the
reference's weight solver.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from benchmarks.reference import reference_value
from mlsynth import ORTHSC

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "carbontax_fullsample_data.dta.txt")
_CONTROLS = ["Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
             "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
             "Switzerland", "United States"]
_INSTRS = ["Finland", "Germany", "Ireland", "Italy", "Netherlands", "Norway",
           "United Kingdom"]


def _panel() -> pd.DataFrame:
    df = pd.read_stata(os.path.abspath(_DATA))
    df = df.rename(columns={"CO2_transport_capita": "Y"})
    df["treat"] = ((df["country"] == "Sweden") & (df["year"] >= 1990)).astype(int)
    return df


def _config() -> dict:
    """ORTHSC config minus the dataframe (shared by run/comparison)."""
    return {"outcome": "Y", "treat": "treat", "unitid": "country", "time": "year",
            "instruments": _INSTRS, "controls": _CONTROLS}


def run() -> dict:
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ORTHSC({**_config(), "df": df, "display_graphs": False}).fit()
    return {
        "att": float(res.att),
        "pvalue": float(res.inference.p_value),
        "smoothing_K": float(res.method_details.parameters_used["smoothing_K"]),
        "ci_lower": float(res.inference.ci_lower),
        "ci_upper": float(res.inference.ci_upper),
    }


# Reference quantities, mapping each comparison label to its key in the captured
# Fry-ORTHSC bundle (benchmarks/reference/orthsc_carbontax/reference.json).
_REF_KEYS = {"ATT": "att", "p_value": "p_value", "smoothing_K": "smoothing_K",
             "CI_lower": "ci_lower", "CI_upper": "ci_upper"}


def _ref(key: str) -> float:
    return reference_value("orthsc_carbontax", key)


def comparison() -> dict:
    """mlsynth ORTHSC vs the Fry R reference, quantity by quantity.

    The mlsynth side is a fresh ``ORTHSC`` fit on Andersson's carbon-tax panel;
    the reference side is a live captured run of Fry's own R code
    (``benchmarks/reference/orthsc_carbontax/``) on the same data -- the
    orthogonalized ATT, the fixed-smoothing t-test p-value, the Sun (2013)
    smoothing K, and the 95% CI -- read via :func:`reference_value`, not
    transcribed. Returns ``{"rows": [...], "mlsynth_call": {...}, "reference": {...}}``.
    """
    df = _panel()
    cfg = _config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ORTHSC({**cfg, "df": df, "display_graphs": False}).fit()
    ml = {
        "ATT": float(res.att),
        "p_value": float(res.inference.p_value),
        "smoothing_K": float(res.method_details.parameters_used["smoothing_K"]),
        "CI_lower": float(res.inference.ci_lower),
        "CI_upper": float(res.inference.ci_upper),
    }
    rows = [{"quantity": q, "mlsynth": round(ml[q], 6),
             "reference": round(_ref(_REF_KEYS[q]), 6)} for q in
            ("ATT", "p_value", "smoothing_K", "CI_lower", "CI_upper")]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "ORTHSC", "config": cfg},
        "reference": {"impl": "Fry OrthogonalizedSyntheticControl (R, live run, captured)",
                      "version": "github.com/JosephPatrickFry @ 3b38684"},
    }


# Targets: a live captured run of Fry's own ORTHSC R code on Andersson's carbon-
# tax data (benchmarks/reference/orthsc_carbontax/), read via reference_value so
# the pins and the captured run are the same object. mlsynth's NumPy/cvxpy port
# reproduces the reference to ~1e-4 (ATT/CI deterministic; the p-value
# deterministic at the fixed smoothing K=4).
EXPECTED = {
    "att": (_ref("att"), 0.005),
    "pvalue": (_ref("p_value"), 0.0005),
    "smoothing_K": (_ref("smoothing_K"), 0.5),
    "ci_lower": (_ref("ci_lower"), 0.02),
    "ci_upper": (_ref("ci_upper"), 0.02),
}
