"""GEOLIFT CPIC cross-validation vs GeoLiftMarketSelection.

Cross-validation against the reference. GeoLift's market-selection budget layer
computes, per candidate, ``investment = cpic * sum(Y[D==1]) * es`` (cost per
incremental conversion x effect size x summed treated volume over the window).
This pins mlsynth's investment against the values GeoLift's ``GeoLiftMarketSelection``
prints in its ``BestMarkets`` table for the walkthrough call

    GeoLiftMarketSelection(data = GeoTestData_PreTest, treatment_periods = c(10,15),
        N = c(2,3,4,5), effect_size = seq(0, 0.2, 0.05), include_markets = "chicago",
        exclude_markets = "honolulu", cpic = 7.50, budget = 1e5, fixed_effects = TRUE, ...)

run live (augsynth-sourced) on the GeoLift_PreTest panel. The investment is a
deterministic data transform -- no model, no permutation -- so the agreement is
exact to the cent at each candidate's (matching) MDE. The MDE and the augsynth
fit themselves are pinned by ``geolift_augsynth_ref`` / ``geolift_walkthrough``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.geolift_helpers.marketselect.helpers.simulate import simulate_lookback
from mlsynth.utils.geolift_helpers.marketselect.helpers.shaping import donor_matrix

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_market_data.csv")
_CPIC = 7.5
_PRE_TOTAL = 90

# (candidate, duration, MDE) at which GeoLiftMarketSelection reports Investment.
_CASES = [
    (("chicago", "portland"), 15, 0.10, 64563.75),
    (("chicago", "portland"), 10, 0.10, 43646.25),
    (("chicago", "cincinnati", "houston", "portland"), 15, 0.05, 74118.375),
    (("chicago", "cincinnati", "houston", "portland"), 10, 0.10, 99027.75),
]


def run() -> dict:
    Ywide = geoex_dataprep(pd.read_csv(os.path.abspath(_DATA)),
                           "location", "date", "Y")["Ywide"]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cand, dur, es, _gl in _CASES:
            candset = frozenset(cand)
            treated_mean = Ywide[list(cand)].mean(axis=1).to_numpy()
            treated_total = Ywide[list(cand)].sum(axis=1).to_numpy()   # GeoLift sum(Y[D==1])
            Y0 = donor_matrix(Ywide, candset).to_numpy()
            rows = simulate_lookback(
                treated_mean, Y0, _PRE_TOTAL, dur, 1, [es],
                augment="ridge", ns=10, seed=0, fixed_effects=True,
                cpic=_CPIC, treated_total=treated_total)
            key = f"inv_{''.join(c[0] for c in cand)}_{dur}"
            out[key] = float(rows[0]["investment"])
    return out


# Investment is deterministic (cpic x es x volume), so it matches GeoLift exactly.
EXPECTED = {
    "inv_cp_15": (64563.75, 0.01),     # chicago, portland
    "inv_cp_10": (43646.25, 0.01),
    "inv_cchp_15": (74118.375, 0.01),  # chicago, cincinnati, houston, portland
    "inv_cchp_10": (99027.75, 0.01),
}
