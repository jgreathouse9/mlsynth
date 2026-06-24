"""SPILLSYNTH (inclusive SCM) cross-validation vs Melnychuk's reference.

Cross-validates mlsynth's ``SPILLSYNTH(method="iscm", iscm_intercept=True)``
against an independent transcription of the inclusive-SCM reference
implementation (``Melnychuk-Andrii/Spillover-SCM``, pinned commit ``282b621``)
on German reunification. The reference's SCM backend is a **demeaned simplex SCM
with an intercept** (``scm_weights``); with ``iscm_intercept=True`` mlsynth uses
the same backend, so the two should agree up to the simplex solver (mlsynth's
FISTA vs the reference's SLSQP/ipop).

mlsynth reproduces the reference's German neighbourhood and inclusive ATT:

  ===============  ===============  ===============
  Quantity         mlsynth          reference port
  ===============  ===============  ===============
  Austria in WG    ~0.46            ~0.45
  WG in Austria    ~0.35            ~0.36
  inclusive ATT    ~-1279           ~-1276
  ===============  ===============  ===============

This is the no-covariates inclusive fit; the residual gap to Di Stefano &
Mellace's *covariate*-based 0.42 weight is the predictor specification, not the
inclusive machinery. Cross-validation (the data + algorithm are the reference's,
transcribed): the case **skips gracefully** when the reference clone is
unavailable.
"""
from __future__ import annotations

import os
import warnings

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "repgermany.dta")


def _fit():
    import pandas as pd

    from mlsynth import SPILLSYNTH

    d = pd.read_stata(os.path.abspath(_DATA))
    d = d[["country", "year", "gdp", "trade", "infrate",
           "industry", "schooling", "invest80"]].copy()
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1990)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year", "method": "iscm",
            "affected_units": ["Austria"], "iscm_intercept": True,
            "display_graphs": False,
        }).fit()


def run() -> dict:
    from benchmarks.reference.clone_iscm import reference_inclusive_german

    ref = reference_inclusive_german()            # skips if reference unavailable
    res = _fit()
    cw = res.iscm.cross_weights
    w_A = float(cw["Austria in West Germany"])
    l_WG = float(cw["West Germany in Austria"])
    return {
        "w_A": w_A,
        "l_WG": l_WG,
        "inclusive_att": float(res.att),
        "naive_att": float(res.att_scm),
        "w_A_vs_ref": abs(w_A - ref["w_A"]),
        "l_WG_vs_ref": abs(l_WG - ref["l_WG"]),
        "inclusive_vs_ref": abs(float(res.att) - ref["inclusive_att"]),
        "naive_vs_ref": abs(float(res.att_scm) - ref["naive_att"]),
    }


def comparison() -> dict:
    """mlsynth's inclusive SCM vs the ported Melnychuk reference, quantity by
    quantity: the two cross-weights (Austria in West Germany, West Germany in
    Austria) and the naive / inclusive ATT, side by side."""
    from benchmarks.reference.clone_iscm import _COMMIT, reference_inclusive_german

    ref = reference_inclusive_german()            # load first: skips if unavailable
    res = _fit()
    cw = res.iscm.cross_weights
    w_A = float(cw["Austria in West Germany"])
    l_WG = float(cw["West Germany in Austria"])
    cfg = {"outcome": "gdp", "treat": "treat", "unitid": "country", "time": "year",
           "method": "iscm", "affected_units": ["Austria"], "iscm_intercept": True}
    rows = [
        {"quantity": "weight[Austria in West Germany]",
         "mlsynth": round(w_A, 6), "reference": round(ref["w_A"], 6)},
        {"quantity": "weight[West Germany in Austria]",
         "mlsynth": round(l_WG, 6), "reference": round(ref["l_WG"], 6)},
        {"quantity": "naive_ATT",
         "mlsynth": round(float(res.att_scm), 6), "reference": round(ref["naive_att"], 6)},
        {"quantity": "inclusive_ATT",
         "mlsynth": round(float(res.att), 6), "reference": round(ref["inclusive_att"], 6)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SPILLSYNTH", "config": cfg},
        "reference": {"impl": "Melnychuk-Andrii/Spillover-SCM inclusive SCM "
                              "(scm_weights/runInclusiveSCM), transcribed to NumPy",
                      "version": f"git {_COMMIT[:7]}"},
    }


# Deterministic (closed-form demeaned simplex SCM + Cramer's rule). The ``*_vs_ref``
# cells pin mlsynth to the ported Melnychuk reference; the residuals are the
# simplex solver (FISTA here vs SLSQP/ipop in the reference), ~2%.
EXPECTED = {
    "w_A": (0.46, 0.03),                 # Austria in synthetic West Germany
    "l_WG": (0.35, 0.03),                # West Germany in synthetic Austria
    "inclusive_att": (-1279.0, 20.0),
    "naive_att": (-1482.0, 20.0),
    "w_A_vs_ref": (0.01, 0.03),          # reproduces the reference weight
    "l_WG_vs_ref": (0.003, 0.03),
    "inclusive_vs_ref": (3.0, 20.0),     # reproduces the reference inclusive ATT
    "naive_vs_ref": (8.0, 20.0),
}
