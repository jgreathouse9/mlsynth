"""SPILLSYNTH (Grossi et al.) Path-A: direct + spillover on German reunification.

Exercises mlsynth's ``SPILLSYNTH(method="grossi")`` -- the direct-and-spillover
potential-outcomes estimator of Grossi, Mariani, Mattei, Lattarulo & Öner (2025),
*"Direct and spillover effects of a new tramway line ..."* (JRSS-A, qnae032) --
on the canonical German-reunification panel (``basedata/repgermany.dta``), the
worked example in the estimator's documentation.

Grossi et al. split the donor pool into a *clean* set and the spillover-exposed
*affected* set: the treated unit's counterfactual is built only from clean
donors (Austria is removed), the **direct** effect on the treated unit is
estimated net of contamination, and a separate **spillover** effect is recovered
for each affected neighbour. On German reunification mlsynth reproduces:

* a **direct** effect on West-German GDP of :math:`\\approx -1605` -- *more*
  negative than the naive SCM gap (:math:`-1306`) and in line with the canonical
  reunification magnitude, because the naive synthetic borrows from a
  contaminated Austria;
* Austria **excluded** from the donor weights but assigned a negative
  **spillover** effect (:math:`\\approx -257`);
* a 15-unit clean donor pool (the full pool minus West Germany and Austria).

The paper's own empirical application (the Florence tramway) relies on a
fine-grained street-level retail panel that is not publicly redistributed, so
this case pins the method's direct/spillover decomposition on the canonical
reunification data as a durable Path-A regression guard.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

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
            "method": "grossi", "affected_units": ["Austria"],
            "display_graphs": False,
        }).fit()
    return res


def _austria_spillover_mean(spillover_att) -> float:
    val = spillover_att["Austria"] if isinstance(spillover_att, dict) else spillover_att
    return float(np.mean(np.asarray(val, dtype=float)))


def run() -> dict:
    res = _fit()
    f = res.grossi
    return {
        "direct_att": float(f.direct_att),
        "att_naive_scm": float(f.att_scm),
        "austria_spillover": _austria_spillover_mean(f.spillover_att),
        "n_clean_donors": float(f.n_clean),
        "austria_excluded": float("Austria" not in f.donor_weights),
        "direct_more_negative_than_naive": float(f.direct_att < f.att_scm),
    }


# Deterministic (closed-form penalised SCM, no RNG). Pins mlsynth's grossi
# direct/spillover decomposition on German reunification: a direct effect more
# negative than the naive gap, Austria removed from the donor pool but assigned a
# negative spillover, on a 15-unit clean pool.
EXPECTED = {
    "direct_att": (-1605.1, 12.0),
    "att_naive_scm": (-1305.95, 8.0),
    "austria_spillover": (-257.2, 25.0),
    "n_clean_donors": (15.0, 0.0),
    "austria_excluded": (1.0, 0.0),
    "direct_more_negative_than_naive": (1.0, 0.0),
}
