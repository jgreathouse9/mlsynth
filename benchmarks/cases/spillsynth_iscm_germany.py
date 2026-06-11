"""SPILLSYNTH (inclusive SCM) Path-A: German reunification with Austria spillover.

Exercises the **Inclusive** Synthetic Control Method of Di Stefano & Mellace
(2024), *"The inclusive Synthetic Control Method"* (arXiv 2403.17624) -- mlsynth's
``SPILLSYNTH(method="iscm")`` -- on the 1990 German reunification: it **keeps
Austria, the affected neighbour, in the donor pool** and de-contaminates the
spillover (solving the 2x2 cross-weight system by Cramer's rule), rather than
dropping Austria as a restricted SCM would.

This is distinct from the **Iterative** SCM of Melnychuk (2024) -- a different,
"waterfall" method whose German headline is a ~1,970 USD reduction -- which
mlsynth does not implement. It is also distinct from the *with-covariates*
inclusive specification of Di Stefano & Mellace's Table 2 (which, using Abadie et
al.'s special-predictor weights, assigns Austria a 0.42 weight): that exact spec
is **not** pinned here.

What this case pins is the **outcome-only** inclusive fit on
``basedata/repgermany.dta`` -- the no-covariates regime that Melnychuk's reference
also reports: Austria carries :math:`\\approx 0.33` of synthetic West Germany and
West Germany :math:`\\approx 0.32` of synthetic Austria, the cross-weight system
has :math:`\\det\\Omega \\approx 0.90`, and the inclusive (de-contaminated) ATT is
**more negative** than the naive SCM gap, because the naive synthetic borrows from
a contaminated Austria. A durable regression / machinery guard for the inclusive
correction, not a cross-validation of the with-covariates Table-2 weights.
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
        res = SPILLSYNTH({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "method": "iscm", "affected_units": ["Austria"],
            "display_graphs": False,
        }).fit()
    return res


def run() -> dict:
    res = _fit()
    cw = {k: float(v) for k, v in res.iscm.cross_weights.items()}
    austria_in_wg = next(v for k, v in cw.items() if k.startswith("Austria"))
    wg_in_austria = next(v for k, v in cw.items() if k.startswith("West Germany"))
    return {
        "att_inclusive": float(res.att),
        "att_naive_scm": float(res.att_scm),
        "austria_in_wg_weight": austria_in_wg,
        "wg_in_austria_weight": wg_in_austria,
        "omega_det": float(res.iscm.omega_det),
        "inclusive_more_negative": float(res.att < res.att_scm),
    }


# Deterministic (closed-form bilevel SCM, no RNG). Pins mlsynth's reproduction of
# the Di Stefano-Mellace German-reunification neighbourhood and the
# de-contamination direction (inclusive ATT more negative than the naive gap).
EXPECTED = {
    "att_inclusive": (-1367.76, 5.0),
    "att_naive_scm": (-1305.95, 5.0),
    "austria_in_wg_weight": (0.327, 0.02),    # Austria in synthetic West Germany
    "wg_in_austria_weight": (0.318, 0.02),    # West Germany in synthetic Austria
    "omega_det": (0.896, 0.02),               # 2x2 cross-weight system determinant
    "inclusive_more_negative": (1.0, 0.0),
}
