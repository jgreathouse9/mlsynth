"""Arm 2: does a Bayesian engine reproduce GeoLift's market ranking?

``benchmarks/cases/geox_augsynth_geolift.py`` pins GEOX(engine="augsynth") to
the BestMarkets top five the GeoLift_Walkthrough prints. This arm runs the same
data, the same config and the same harness with the engine swapped, so a
divergence is attributable to the estimator.

    python market_selection.py [results/market_selection.json]

Result, mean absolute rank error against GeoLift's published table:

    augsynth   0.00     reproduces it -- it is the reference implementation
    bscm      15.20     all five designs present, ordering scrambled
    mvbbsc    10.60     picks GeoLift's own top design first

The ranking is not reproducible by a different estimator even when that
estimator is calibrated, because the MDE is a functional of the interval the
engine reports. That is the same fragility the criterion arm measures directly.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth import GEOX
from mlsynth.config_models import GEOXConfig

from . import engines as bayes

DATA = "basedata/geolift_market_data.csv"
PUBLISHED = {  # GeoLift_Walkthrough BestMarkets
    ("chicago", "portland"): {15: (1.0, 64563.75), 10: (3.0, 43646.25)},
    ("chicago", "cincinnati", "houston", "portland"): {15: (1.0, 74118.38),
                                                       10: (3.0, 99027.75)},
    ("chicago", "houston", "portland"): {10: (5.0, 75389.25)},
}


def shortlist(engine: str, data_path: str = DATA) -> pd.DataFrame:
    kw = dict(df=pd.read_csv(data_path), outcome="Y", unitid="location",
              time="date", treatment_size=[2, 3, 4, 5], to_be_treated=["chicago"],
              not_to_be_treated=["honolulu"], durations=[10, 15],
              effect_sizes=[0.0, 0.05, 0.10, 0.15, 0.20], n_backtests=1,
              how="sum", engine=engine, alpha=0.1, power_threshold=0.8,
              cpic=7.5, budget=1e5, ns=1000, seed=0, n_validation_backtests=0)
    if engine == "augsynth":
        kw.update(augment="ridge", fixed_effects=True, conformal_type="iid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GEOX(GEOXConfig(**kw)).fit()
    table = res.power.copy()
    table["cand"] = table["candidate"].apply(frozenset)
    return table


def run(which=("augsynth", "bscm", "mvbbsc"), data_path: str = DATA) -> dict:
    bayes.install()
    out = {}
    for engine in which:
        table = shortlist(engine, data_path)
        rows, errs = {}, []
        for markets, by_dur in PUBLISHED.items():
            for dur, (rank, inv) in by_dur.items():
                hit = table[(table["cand"] == frozenset(markets))
                            & (table["duration"] == dur)]
                key = "%s@%d" % ("+".join(sorted(markets)), dur)
                if hit.empty:
                    rows[key] = {"published_rank": rank, "rank": None}
                    continue
                got = hit.iloc[0]
                rows[key] = {"published_rank": rank, "rank": float(got["rank"]),
                             "published_investment": inv,
                             "investment": float(got["investment"])}
                errs.append(abs(float(got["rank"]) - rank))
        best = table.nsmallest(1, "rank").iloc[0]
        out[engine] = {"n_designs": int(len(table)), "designs": rows,
                       "mean_abs_rank_error": float(np.mean(errs)) if errs else None,
                       "own_top": sorted(best["cand"]),
                       "own_top_duration": int(best["duration"])}
        print("%-9s %d designs | mean |rank error| %.2f | own #1 %s (dur %d)"
              % (engine, len(table), np.mean(errs), sorted(best["cand"]),
                 best["duration"]), flush=True)
    return out


if __name__ == "__main__":
    res = run()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=2)
