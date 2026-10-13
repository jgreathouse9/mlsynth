"""Cross-validate the paper port against google/matched_markets on its own data.

    MLSYNTH_MATCHED_MARKETS=/path/to/matched_markets \
        python -m benchmarks.studies.tbr_geo.xval

The reference is Apache 2.0 and is not vendored here, so this arm skips unless
that environment variable points at a checkout. ``simulation.py`` needs nothing
outside this package and always runs.

The reference ships ``salesandcost.csv``: 100 geos over 93 daily dates, split
50/50, with cost positive from 2015-02-16 to 2015-03-15. Its panel is
unbalanced -- 75 geo-days are absent, concentrated on the smallest geos, which
is dropped zero-sales days. Summing over the rows that are present, which is
what the reference does, equals summing a zero-filled full grid, so this fills
the grid and hands ``dataprep`` a balanced panel.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.datautils import dataprep

from .tbr import TBR, is_fixed_cost, iroas_fixed_cost

TEST_START = pd.Timestamp("2015-02-16")


def _reference():
    root = os.environ.get("MLSYNTH_MATCHED_MARKETS")
    if not root or not os.path.isdir(root):
        return None
    sys.path.insert(0, root)
    try:
        from matched_markets.examples import salesandcost
        from matched_markets.methodology.tbr import TBR as RefTBR
        from matched_markets.methodology.tbr_iroas import TBRiROAS
    except ImportError:
        return None
    return salesandcost, RefTBR, TBRiROAS, os.path.join(
        root, "matched_markets/csv")


def balanced_frame(salesandcost, csv_dir):
    snc, ga, exdates = salesandcost.example_data(csv_dir)
    frame = salesandcost.format_example_data(snc, ga, exdates).reset_index()
    grid = pd.MultiIndex.from_product(
        [sorted(frame.geo.unique()), sorted(frame.date.unique())],
        names=["geo", "date"])
    frame = (frame.set_index(["geo", "date"]).reindex(grid)
             .assign(sales=lambda d: d.sales.fillna(0.0),
                     cost=lambda d: d.cost.fillna(0.0))
             .reset_index())
    frame["geo.group"] = frame.geo.map(ga.set_index("geo")["geo.group"])
    frame["period"] = salesandcost._get_periods(frame.date, exdates)
    return frame, ga


def group_series(frame, ga, column):
    wide = frame.pivot_table(index="date", columns="geo", values=column,
                             aggfunc="sum")
    group = ga.set_index("geo")["geo.group"]
    treated = [g for g in wide.columns if group[g] == 2]
    control = [g for g in wide.columns if group[g] == 1]
    return wide[treated].sum(axis=1), wide[control].sum(axis=1), wide.index


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ref = _reference()
    if ref is None:
        print("google/matched_markets not found -- set "
              "MLSYNTH_MATCHED_MARKETS to a checkout to run this arm.")
        return 0
    salesandcost, RefTBR, TBRiROAS, csv_dir = ref
    frame, ga = balanced_frame(salesandcost, csv_dir)

    long = frame.rename(columns={"geo": "unit", "date": "time"}).copy()
    long["D"] = ((long["geo.group"] == 2) &
                 (long.time >= TEST_START)).astype(int)
    prep = dataprep(long, "unit", "time", "sales", "D", allow_no_donors=True)
    print(f"dataprep accepts the filled panel: Ywide {prep['Ywide'].shape}\n")

    fits = {}
    for column in ("sales", "cost"):
        y, x, dates = group_series(frame, ga, column)
        pre = dates < TEST_START
        mine = TBR().fit(y[pre].values, x[pre].values)
        dist = mine.cumulative(y[~pre].values, x[~pre].values)
        reference = RefTBR(use_cooldown=True)
        reference.fit(frame, column, key_group="geo.group",
                      key_period="period", key_date="date", key_geo="geo")
        rdist = reference.causal_cumulative_distribution()
        rl = np.atleast_1d(rdist.kwds["loc"]).ravel()
        rs = np.atleast_1d(rdist.kwds["scale"]).ravel()
        ml = np.atleast_1d(dist.kwds["loc"])
        ms = np.atleast_1d(dist.kwds["scale"])
        fits[column] = dict(dist=dist, loc=ml)
        print(f"{column}:  rank-deficient design = {mine.degenerate_}")
        print(f"  alpha, beta   mine={np.round(mine.coef_, 8)}  "
              f"ref={np.round(np.asarray(reference.pre_period_model.params), 8)}")
        print(f"  s^2           mine={mine.s2_:.8f}  "
              f"ref={reference.pre_period_model.scale:.8f}")
        print(f"  max |d loc|   {np.abs(ml - rl).max():.3e}    "
              f"max |d scale| {np.abs(ms - rs).max():.3e}")
        print(f"  Delta(T)      mine={ml[-1]:.6f}  ref={rl[-1]:.6f}\n")

    cost_pre = frame[frame.period == 0]["cost"].values
    cost_test_control = frame[(frame.period == 1) &
                              (frame["geo.group"] == 1)]["cost"].values
    total_cost = fits["cost"]["loc"][-1]
    print(f"fixed-cost scenario (section 3.4): "
          f"{is_fixed_cost(cost_pre, cost_test_control)};  "
          f"total incremental cost {total_cost:.4f}")

    mine_iroas = iroas_fixed_cost(fits["sales"]["dist"], total_cost)
    loc = np.atleast_1d(mine_iroas.kwds["loc"])
    ref_iroas = TBRiROAS(use_cooldown=True)
    ref_iroas.fit(frame, key_group="geo.group", key_period="period",
                  key_date="date", key_geo="geo", key_cost="cost",
                  key_response="sales")
    report = ref_iroas.summary(level=0.9, tails=2)
    print(f"\n  {'quantity':24} {'mine':>16} {'reference':>16} {'diff':>10}")
    pairs = (
        ("iROAS point", loc[-1], float(report["estimate"].iloc[0])),
        ("iROAS 90% lower", float(mine_iroas.ppf(0.05)[-1]),
         float(report["lower"].iloc[0])),
        ("iROAS 90% upper", float(mine_iroas.ppf(0.95)[-1]),
         float(report["upper"].iloc[0])),
        ("incremental cost", total_cost,
         float(report["incremental_cost"].iloc[0])),
        ("incremental response", fits["sales"]["loc"][-1],
         float(report["incremental_response"].iloc[0])),
    )
    for name, mine_v, ref_v in pairs:
        print(f"  {name:24} {mine_v:16.8f} {ref_v:16.8f} "
              f"{abs(mine_v - ref_v):10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
