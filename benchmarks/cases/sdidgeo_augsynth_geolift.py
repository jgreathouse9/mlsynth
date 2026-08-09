"""SDIDGEO cross-validation: the augsynth engine against GeoLift's BestMarkets.

Cross-validation against the reference implementation. SDIDGEO's scoring
estimator is pluggable, and this case answers what the seam is for: with the
augsynth engine selected, does the design harness reproduce the market selection
GeoLift itself publishes?

The GeoLift_Walkthrough runs

    GeoLiftMarketSelection(data = GeoTestData_PreTest, treatment_periods = c(10,15),
        N = c(2,3,4,5), effect_size = seq(0, 0.2, 0.05), include_markets = "chicago",
        exclude_markets = "honolulu", cpic = 7.50, budget = 1e5, fixed_effects = TRUE,
        side_of_test = "two_sided")

and prints a ranked ``BestMarkets`` table whose top five designs are

    rank  candidate                                  dur  es    investment   abs_lift0
    1     chicago, cincinnati, houston, portland     15   0.05  $74,118.38   0.002
    1     chicago, portland                          15   0.10  $64,563.75   0.001
    3     chicago, cincinnati, houston, portland     10   0.10  $99,027.75   0.004
    3     chicago, portland                          10   0.10  $43,646.25   0.004
    5     chicago, houston, portland                 10   0.10  $75,389.25   0.005

``SDIDGEO(engine="augsynth")`` reaches all five, with the rank, the MDE, the
CPIC investment and the rounded ``abs_lift_in_zero`` matching value for value.
Fourteen quantities are pinned.

Why this is a stronger check than it looks. The engine supplies the fit and the
p-value and nothing else; candidate nomination, the backtest window arithmetic,
effect injection, the power sweep, the MDE rule and GeoLift's composite rank all
sit above it in shared code. Reproducing the published ranking therefore
exercises the whole harness, not the estimator alone -- a divergence anywhere in
that stack would move a rank or an investment. It is also what licenses reading
the SDID engine's numbers as a like-for-like comparison: the same harness,
scored differently.

The scan runs sizes 2 through 5 in one call and ranks them together, which is
what ``GeoLiftMarketSelection`` does with its single ``resultsM`` table. The
marginal rank-6 design differs from the vignette (augsynth-version scoring at the
tail), so only the stable top five are pinned.

Runtime is a few seconds: ``to_be_treated=["chicago"]`` cuts the field to 33
candidates, since GeoLift generates candidates ignoring the forced units and then
keeps those already containing them.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from mlsynth import SDIDGEO
from mlsynth.config_models import SDIDGEOConfig

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_market_data.csv")   # 90-period PreTest

_CP = frozenset({"chicago", "portland"})
_CCHP = frozenset({"chicago", "cincinnati", "houston", "portland"})
_CHP = frozenset({"chicago", "houston", "portland"})


def _shortlist() -> pd.DataFrame:
    """The pooled, ranked design table from one augsynth-engine run."""
    df = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SDIDGEO(SDIDGEOConfig(
            df=df, outcome="Y", unitid="location", time="date",
            treatment_size=[2, 3, 4, 5], to_be_treated=["chicago"],
            not_to_be_treated=["honolulu"], durations=[10, 15],
            effect_sizes=[0.0, 0.05, 0.10, 0.15, 0.20], n_backtests=1,
            how="sum", engine="augsynth", augment="ridge", fixed_effects=True,
            alpha=0.1, power_threshold=0.8, cpic=7.5, budget=1e5, ns=1000,
            seed=0, conformal_type="iid", n_validation_backtests=0,
        )).fit()
    out = res.power.copy()
    out["cand"] = out["candidate"].apply(frozenset)
    return out


def run() -> dict:
    pool = _shortlist()

    def row(cand: frozenset, duration: int) -> pd.Series:
        hit = pool[(pool["cand"] == cand) & (pool["duration"] == duration)]
        if hit.empty:
            raise AssertionError(
                f"design {sorted(cand)} at duration {duration} is absent from "
                "the shortlist; GeoLift ranks it in the top five.")
        return hit.iloc[0]

    cp15, cchp15 = row(_CP, 15), row(_CCHP, 15)
    cp10, cchp10 = row(_CP, 10), row(_CCHP, 10)
    chp10 = row(_CHP, 10)

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
    }


# GeoLift's own published BestMarkets values. The investment is a deterministic
# transform of the panel (cpic x effect_size x summed treated volume), so it is
# pinned to the cent; the MDE sits on the 0.05 effect grid; abs_lift_in_zero is
# reported to three decimals, matching the vignette's rounding.
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
}
