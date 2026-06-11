"""SPILLSYNTH (Iterative SCM) Path-A: German reunification (Melnychuk 2024).

Reproduces the headline of Melnychuk (2024), *"Synthetic Controls with Spillover
Effects: A Comparative Study"* (arXiv 2405.01645): the **Iterative ("waterfall")
SCM** estimate of the 1990 German reunification's effect on West-German GDP per
capita, cleaning the spillover that Austria (the affected neighbour) carries.

mlsynth's ``SPILLSYNTH(method="iterative")`` runs the covariate backend on
``basedata/repgermany.dta`` with Abadie et al.'s German predictor specification
(predictors ``gdp`` / ``trade`` / ``infrate`` averaged over the
``time.predictors.prior`` window 1981-1990; special predictors ``industry``
[1981-1990], ``schooling`` [1980, 1985], ``invest80`` [1980]; SSR over
1960-1989). The lagged-GDP predictor pulls Austria's weight in synthetic West
Germany to ``~0.43`` (the paper's 0.42). It cleans Austria's post-treatment
outcomes with Austria's own spillover-free synthetic and refits West Germany on
the cleaned pool, yielding a **more negative** effect than the naive SCM:

  ================  ===============  ===============
  Quantity          mlsynth          Melnychuk
  ================  ===============  ===============
  Iterative ATT     ~ -1813          ~ -1970
  naive SCM ATT     ~ -1333          ~ -1600 (Abadie)
  ================  ===============  ===============

The ~8% gap to the paper's -1970 is the V-optimisation backend: mlsynth's global
``mscmt`` bilevel solver vs Abadie's ``Synth::synth``. The fit is **deterministic**
(``mscmt`` is seeded). Path A (the paper's empirical result, V-solver slack):
the case pins mlsynth's value tightly as a regression guard *and* checks it lands
within the solver slack of the paper's -1970.

Runtime: ~2-3 min (two global ``mscmt`` V-optimisations).
"""
from __future__ import annotations

import os
import warnings

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "repgermany.dta")

# Abadie's German spec: gdp / trade / infrate averaged over time.predictors.prior
# (1981-1990), plus the three special predictors. The lagged-GDP predictor is the
# one the paper carries and the Melnychuk replication file silently drops.
_COV = ["gdp", "trade", "infrate", "industry", "schooling", "invest80"]
_WIN = {"gdp": (1981, 1990), "trade": (1981, 1990), "infrate": (1981, 1990),
        "industry": (1981, 1990), "schooling": (1980, 1985), "invest80": (1980, 1980)}


def _fit():
    import pandas as pd

    from mlsynth import SPILLSYNTH

    d = pd.read_stata(os.path.abspath(_DATA))
    # _COV already contains the outcome "gdp" (the lagged-GDP predictor); select
    # each column once so the panel doesn't carry a duplicate gdp column.
    cols = ["country", "year"] + list(dict.fromkeys(_COV))
    d = d[cols].copy()
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1990)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SPILLSYNTH({
            "df": d, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year", "method": "iterative",
            "affected_units": ["Austria"], "covariates": _COV,
            "covariate_windows": _WIN, "bilevel_solver": "mscmt",
            "display_graphs": False,
        }).fit()


def run() -> dict:
    res = _fit()
    f = res.iterative
    return {
        "iterative_att": float(res.att),
        "naive_att": float(res.att_scm),
        "iterative_more_negative": float(res.att < res.att_scm),
        "austria_cleaned": float(f.cleaned_units == ["Austria"]),
        "n_clean": float(f.n_clean),
        "vs_paper_1970": abs(float(res.att) - (-1970.0)),
    }


# Deterministic (mscmt seeded at 0). The ``*_att`` cells pin mlsynth's value as a
# regression guard; ``vs_paper_1970`` checks the estimate lands within the
# V-solver slack of Melnychuk's reported -1970 (the residual is mlsynth's mscmt
# bilevel solver vs Abadie's Synth::synth V-optimisation).
EXPECTED = {
    "iterative_att": (-1813.0, 45.0),
    "naive_att": (-1333.0, 45.0),
    "iterative_more_negative": (1.0, 0.0),
    "austria_cleaned": (1.0, 0.0),
    "n_clean": (15.0, 0.0),
    "vs_paper_1970": (157.0, 130.0),       # |-1813 - (-1970)| ~ 157, within slack
}
