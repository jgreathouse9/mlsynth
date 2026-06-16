"""GEOLIFT market-selection cross-validation vs GeoLift's BestMarkets ranking.

Cross-validation against the reference. The GeoLift_Walkthrough runs

    GeoLiftMarketSelection(data = GeoTestData_PreTest, treatment_periods = c(10,15),
        N = c(2,3,4,5), effect_size = seq(0, 0.2, 0.05), include_markets = "chicago",
        exclude_markets = "honolulu", cpic = 7.50, budget = 1e5, fixed_effects = TRUE,
        side_of_test = "two_sided")

and prints a ranked ``BestMarkets`` table whose top five designs are:

    rank  candidate                                  dur  es    investment   abs_lift0
    1     chicago, cincinnati, houston, portland     15   0.05  $74,118.38   0.002
    1     chicago, portland                          15   0.10  $64,563.75   0.001
    3     chicago, cincinnati, houston, portland     10   0.10  $99,027.75   0.004
    3     chicago, portland                          10   0.10  $43,646.25   0.004
    5     chicago, houston, portland                 10   0.10  $75,389.25   0.005

mlsynth's ``GEOLIFT`` design takes a single ``treatment_size``, so this case runs
it for N = 2, 3, 4, 5 (chicago forced in, honolulu out, the same duration/effect
grid) and pools the per-(candidate, duration) MDE rows, then re-applies GeoLift's
composite rank (the mean of the three ``dense_rank``s over |MDE|, power, and
abs_lift_in_zero, ties = min) across the pool -- exactly as
``GeoLiftMarketSelection`` ranks its single ``resultsM`` table. It pins, for each
of the five published designs, the rank, the CPIC investment (a deterministic
data transform, exact to the cent), the MDE, and the rounded abs_lift_in_zero.

This depends on the ``include_markets`` candidate generation being faithful to
GeoLift's generate-then-filter (``pre_test_power.R``): candidates are generated
ignoring the forced units, then filtered to those that already contain them, so
only correlation-natural sets survive. (The per-candidate fit/conformal scoring
that produces the MDE / investment is pinned by ``geolift_walkthrough`` /
``geolift_cpic`` / ``geolift_augsynth_ref``.) The marginal rank-6 design differs
from the vignette (augsynth-version scoring at the tail), so only the stable
top-five are pinned.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from mlsynth import GEOLIFT

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_market_data.csv")  # 90-period PreTest

_CP = frozenset({"chicago", "portland"})
_CCHP = frozenset({"chicago", "cincinnati", "houston", "portland"})
_CHP = frozenset({"chicago", "houston", "portland"})


def _fit_power(N: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(_DATA))
    config = {
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": N, "to_be_treated": ["chicago"],
        "not_to_be_treated": ["honolulu"], "durations": [10, 15],
        "effect_sizes": [0.0, 0.05, 0.10, 0.15, 0.20], "lookback_window": 1,
        "how": "sum", "augment": "ridge", "fixed_effects": True, "alpha": 0.1,
        "power_threshold": 0.8, "cpic": 7.5, "budget": 1e5, "ns": 1000,
        "seed": 0, "conformal_type": "iid", "display_graphs": False, "n_jobs": -1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GEOLIFT(config).fit().power


def run() -> dict:
    pool = pd.concat([_fit_power(N) for N in (2, 3, 4, 5)], ignore_index=True)
    pool["cand"] = pool["candidate"].apply(frozenset)

    # GeoLift composite rank across the pooled MDE table (pre_test_power.R):
    # mean of the three dense_ranks over |MDE|, power, abs_lift_in_zero; ties min.
    pool["rank_mde"] = pool["mde"].abs().rank(method="dense")
    pool["rank_pvalue"] = pool["power"].rank(method="dense")
    pool["rank_abszero"] = pool["abs_lift_in_zero"].rank(method="dense")
    pool["rank"] = pool[["rank_mde", "rank_pvalue", "rank_abszero"]].mean(
        axis=1).rank(method="min")

    def row(cand: frozenset, duration: int) -> pd.Series:
        hit = pool[(pool["cand"] == cand) & (pool["duration"] == duration)]
        return hit.iloc[0]

    cp15, cchp15 = row(_CP, 15), row(_CCHP, 15)
    cp10, cchp10 = row(_CP, 10), row(_CCHP, 10)
    chp10 = row(_CHP, 10)

    # the two rank-1 designs are exactly {chicago, portland}@15 and
    # {chicago, cincinnati, houston, portland}@15
    rank1 = {(r["cand"], int(r["duration"]))
             for _, r in pool[pool["rank"] == 1.0].iterrows()}

    return {
        "cp15_rank": float(cp15["rank"]),
        "cp15_inv": float(cp15["investment"]),
        "cp15_mde": float(cp15["mde"]),
        "cp15_az": float(cp15["abs_lift_in_zero"]),
        "cchp15_rank": float(cchp15["rank"]),
        "cchp15_inv": float(cchp15["investment"]),
        "cchp15_mde": float(cchp15["mde"]),
        "cchp15_az": float(cchp15["abs_lift_in_zero"]),
        "cp10_rank": float(cp10["rank"]),
        "cp10_inv": float(cp10["investment"]),
        "cchp10_rank": float(cchp10["rank"]),
        "cchp10_inv": float(cchp10["investment"]),
        "chp10_rank": float(chp10["rank"]),
        "chp10_inv": float(chp10["investment"]),
        "top1_designs_match": float(
            rank1 == {(_CP, 15), (_CCHP, 15)}),
    }


# Targets are GeoLift's printed BestMarkets top five; investments are exact to
# the cent, ranks/MDEs/abs_lift_in_zero to the published rounding.
EXPECTED = {
    "cp15_rank": (1.0, 0.5),
    "cp15_inv": (64563.75, 1.0),
    "cp15_mde": (0.10, 0.001),
    "cp15_az": (0.001, 0.0015),
    "cchp15_rank": (1.0, 0.5),
    "cchp15_inv": (74118.38, 1.0),
    "cchp15_mde": (0.05, 0.001),
    "cchp15_az": (0.002, 0.0015),
    "cp10_rank": (3.0, 0.5),
    "cp10_inv": (43646.25, 1.0),
    "cchp10_rank": (3.0, 0.5),
    "cchp10_inv": (99027.75, 1.0),
    "chp10_rank": (5.0, 0.5),
    "chp10_inv": (75389.25, 1.0),
    "top1_designs_match": (1.0, 0.5),
}
