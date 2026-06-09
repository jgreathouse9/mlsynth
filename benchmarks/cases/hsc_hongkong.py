"""Path A benchmark: HSC 1997 Hong Kong handover (Liu & Xu, Sec. 7).

Reproduces the Harmonic Synthetic Control headline application -- the effect of
the 1997 handover on Hong Kong's real GDP per capita (Hsiao, Ching & Wan 2012
panel). The decisive detail (and the reason a naive default fit does NOT match
the paper): HSC's weight diversification comes from the **SDID-style ridge** plus
a **refined rho grid** -- the CV optimum rho ~ 0.09 falls between the coarse
default grid's 0.0 and 0.2 points. Under the near-unregularized default ridge the
weight instead collapses onto Korea (~0.41); the SDID ridge is what reproduces
the paper's broad mix (Korea 0.18, Germany 0.14, US 0.13, Italy 0.11).

Provenance
----------
* Data: ``basedata/hong_kong_handover.csv`` -- Hong Kong + 11 OECD donors, annual
  real GDP per capita; treatment 1997; authors' post-window 1997-2003.
* Headline: Liu & Xu (Sec. 7) -- rho ~ 0.11, 2003 effect ~ -1900, broad donor
  mix (Korea 0.18 / Germany 0.14 / US 0.13 / Italy 0.11, max < 0.19).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def run() -> dict:
    from mlsynth import HSC

    df = pd.read_csv(_BASE / "hong_kong_handover.csv")
    df = df[df["year"] <= 2003]                      # authors' 1997-2003 window

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = HSC({
            "df": df, "outcome": "gdp", "unitid": "country",
            "time": "year", "treat": "Handover",
            "ridge": "sdid",                          # SDID ridge -> diversified weights
            "rho_grid": list(np.round(np.arange(0.0, 0.98, 0.01), 2)),
            "display_graphs": False,
        }).fit()

    w = res.weights_by_donor or {}
    return {
        "selected_rho": float(res.selected_rho),
        "effect_2003": float(res.treatment_effect[-1]),
        "att": float(res.att),
        "korea_weight": float(w.get("Korea", 0.0)),
        "germany_weight": float(w.get("Germany", 0.0)),
        "max_weight": float(max(w.values())) if w else 0.0,
    }


# Deterministic (rolling-origin CV). The SDID ridge + fine rho grid reproduce the
# paper's interior rho (~0.1), its ~-1900 2003 effect, and the broad donor mix
# (Korea ~0.18, Germany ~0.14, max < 0.19) -- NOT the near-unregularized
# default's Korea ~0.41. Tolerances absorb solver/CV-grid drift.
EXPECTED = {
    "selected_rho": (0.09, 0.06),
    "effect_2003": (-1902.0, 120.0),
    "att": (-1734.0, 120.0),
    "korea_weight": (0.185, 0.05),
    "germany_weight": (0.137, 0.05),
    "max_weight": (0.185, 0.05),     # diversified: no donor dominates (< ~0.24)
}
