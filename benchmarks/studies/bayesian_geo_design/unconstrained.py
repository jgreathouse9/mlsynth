"""Arm 4: what the design picks when no market is pinned.

Arm 2 runs GeoLift's walkthrough config, which forces chicago into every
candidate (``to_be_treated=["chicago"]``). That is the analyst choosing the
treated market and the design filling in around it, which is the arrangement
this study exists to examine. This arm drops the constraint and lets nomination
range over all 40 markets.

    python -m benchmarks.studies.bayesian_geo_design.unconstrained \
        results/unconstrained.json

Measured on ``geolift_market_data.csv``, sizes 2..5, honolulu excluded:

    augsynth, unconstrained: 106 scored designs (20 with chicago pinned)
      rank 1  dur 15  mde 0.05  inv 64,524  jacksonville+milwaukee+new orleans
      rank 1  dur 15  mde 0.10  inv 64,564  chicago+portland
      rank 1  dur 15  mde 0.05  inv 74,118  chicago+cincinnati+houston+portland

    mvbbsc, unconstrained: 174 scored designs
      rank 1  dur 15  mde 0.05  inv 32,282  chicago+portland
      rank 1  dur 15  mde 0.05  inv 78,628  detroit+jacksonville+milwaukee+new orleans
      rank 1  dur 15  mde 0.05  inv 62,437  houston+nashville+san diego
      rank 1  dur 15  mde 0.05  inv 69,588  atlanta+las vegas+saint paul

Three results.

Both engines rank chicago+portland first with nothing pinned, so GeoLift's
published design is not an artifact of the constraint. Two estimators choosing
freely over 40 markets agree at the top.

The constraint costs something measurable. augsynth's free search reaches
jacksonville+milwaukee+new orleans at an MDE of 0.05 for 64,524, against
chicago+portland at 0.10 for 64,564 -- one grid step better at the same spend,
from a design the pinned 33-candidate field never contained. The effect grid
steps by 0.05, so the ratio is bracketed and not measured.

The engines agree less when neither is constrained, not more: the top-five
overlap is 1 of 5, and the one is chicago+portland. Under the constraint both
engines at least contained all five published designs.

And the MDE stops discriminating. MVBBSC returns five designs tied at rank 1,
all at the 0.05 grid floor, because its intervals are tighter and a larger field
puts more designs under the smallest effect the grid expresses -- 174 feasible
designs against augsynth's 106. Ranking a free field by MDE is the case
``criterion.py`` measures; power at a fixed effect still separates designs that
the MDE ties.
"""
from __future__ import annotations

import json
import sys
import warnings

import pandas as pd

from mlsynth import GEOX
from mlsynth.config_models import GEOXConfig

from . import engines as bayes

DATA = "basedata/geolift_market_data.csv"
CHICAGO_PORTLAND = frozenset({"chicago", "portland"})


def shortlist(engine: str, forced=None, data_path: str = DATA) -> pd.DataFrame:
    kw = dict(df=pd.read_csv(data_path), outcome="Y", unitid="location",
              time="date", treatment_size=[2, 3, 4, 5],
              not_to_be_treated=["honolulu"], durations=[10, 15],
              effect_sizes=[0.0, 0.05, 0.10, 0.15, 0.20], n_backtests=1,
              how="sum", engine=engine, alpha=0.1, power_threshold=0.8,
              cpic=7.5, budget=1e5, ns=1000, seed=0, n_validation_backtests=0)
    if forced:
        kw["to_be_treated"] = list(forced)
    if engine == "augsynth":
        kw.update(augment="ridge", fixed_effects=True, conformal_type="iid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = GEOX(GEOXConfig(**kw)).fit()
    table = res.power.copy()
    table["cand"] = table["candidate"].apply(frozenset)
    return table


def run(which=("augsynth", "mvbbsc"), data_path: str = DATA) -> dict:
    bayes.install()
    out, tops = {}, {}
    for engine in which:
        table = shortlist(engine, forced=None, data_path=data_path)
        top = table.nsmallest(5, "rank")
        cp = table[table["cand"] == CHICAGO_PORTLAND]
        tops[engine] = {frozenset(r["cand"]) for _, r in top.iterrows()}
        out[engine] = {
            "n_designs": int(len(table)),
            "top5": [{"markets": sorted(r["cand"]), "duration": int(r["duration"]),
                      "rank": float(r["rank"]), "mde": float(r["mde"]),
                      "investment": float(r["investment"])}
                     for _, r in top.iterrows()],
            "chicago_portland_rank": (None if cp.empty else
                                      float(cp.nsmallest(1, "rank").iloc[0]["rank"])),
            "n_tied_at_rank1": int((table["rank"] == table["rank"].min()).sum()),
        }
        print("=== %s, unconstrained: %d designs ===" % (engine, len(table)), flush=True)
        for _, r in top.iterrows():
            print("  rank %-4.0f dur %2d  mde %.2f  inv %8.0f  %s"
                  % (r["rank"], r["duration"], r["mde"], r["investment"],
                     sorted(r["cand"])), flush=True)
        print("  chicago+portland rank %s | designs tied at best rank: %d"
              % (out[engine]["chicago_portland_rank"],
                 out[engine]["n_tied_at_rank1"]), flush=True)
    if len(which) == 2:
        a, b = (tops[w] for w in which)
        out["top5_overlap"] = len(a & b)
        print("\ntop-5 overlap between engines: %d of 5" % len(a & b))
    return out


if __name__ == "__main__":
    res = run()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=2)
