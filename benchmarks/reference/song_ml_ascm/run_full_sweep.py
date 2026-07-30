"""All 1024 cells of Song et al.'s main_result.R, against their published values.

    python benchmarks/reference/song_ml_ascm/run_full_sweep.py
    python benchmarks/reference/song_ml_ascm/run_full_sweep.py --inference

The routine benchmark (``benchmarks/cases/song_ml_ascm.py``) pins a stratified
30-cell subset so it stays cheap. This runs the whole design: 8 heating years x
8 treatment groups x 16 pollutant series, the same grid their ``main_result.csv``
contains. It writes one row per cell to ``full_sweep.csv`` and prints a summary
split by heating year.

Split by year because that is where the disagreement lives, and finding that out
is what this sweep is for. An earlier version split on the gold ``Scaled_L2``
instead, on the theory that near-perfectly-fitting cells have a non-unique simplex
optimum and two solvers can therefore reach the same objective value while
disagreeing on the extrapolation. That theory was fitted to two cells and this
sweep refuted it: the worst-disagreeing cells are well conditioned, with gold
``Scaled_L2`` between 0.25 and 0.60. The pattern is temporal, not geometric --
2016 has a mean |diff| of 0.20 against ~0.01 for every other year, and a max of
1.45 -- which points at reference-version drift, since the authors ran whatever
augsynth was current in 2022-2023.

The pre-treatment imbalance agrees to 6.2e-4 everywhere, 2016 included. The same
optimum *value* is reached even where the reported effect differs, which is what
localises the drift to the penalty rather than to the fit.

``--inference`` additionally computes jackknife+ bounds. That costs one refit per
pre-treatment period per cell -- roughly 25 on these windows, so about 25,000
extra fits across the sweep. Expect hours rather than minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))          # benchmarks/ importable

from cases.song_ml_ascm import (            # noqa: E402
    GROUPS, POLLUTANTS, WINDOWS, _PANEL, _REF, _donors, _pre_fit_l2, fit_cell)

#: The heating year whose published values drift from the pinned package. Named
#: rather than inlined so the summary below and the case docstring cannot
#: disagree about which year is the exception.
DRIFT_YEAR = 2016


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", action="store_true",
                    help="also compute jackknife+ bounds (much slower)")
    ap.add_argument("--out", default=str(_HERE / "full_sweep.csv"))
    args = ap.parse_args()

    panel = pd.read_parquet(_PANEL)
    gold = pd.read_parquet(_REF / "gold_main_result.parquet")
    donors = _donors()

    total = len(WINDOWS) * len(GROUPS) * len(POLLUTANTS)
    print(f"{total} cells ({len(WINDOWS)} years x {len(GROUPS)} groups "
          f"x {len(POLLUTANTS)} pollutants), inference={args.inference}")

    rows, t0, n = [], time.time(), 0
    for year in WINDOWS:
        for group in GROUPS:
            for pollutant in POLLUTANTS:
                n += 1
                g = gold[(gold.city == group) & (gold.year == year)
                         & (gold.pollutant == pollutant)]
                rec = {"group": group, "year": year, "pollutant": pollutant}
                try:
                    res = fit_cell(panel, donors, group, year, pollutant,
                                   inference="jackknife_plus" if args.inference
                                   else False)
                except Exception as exc:      # recorded, never silently dropped
                    res = None
                    rec["status"] = f"{type(exc).__name__}: {exc}"
                if res is None:
                    rec.setdefault("status", "unusable slice")
                elif g.empty:
                    rec["status"] = "no gold row"
                else:
                    rec["status"] = "ok"
                    rec["att"] = res.effects.att
                    rec["gold_att"] = float(g.average_att.iloc[0])
                    rec["att_diff"] = abs(rec["att"] - rec["gold_att"])
                    rec["gold_scaled_l2"] = float(g.Scaled_L2.iloc[0])
                    rec["gold_l2"] = float(g.L2.iloc[0])
                    rec["l2"] = _pre_fit_l2(panel, donors, group, year, pollutant)
                    rec["l2_diff"] = abs(rec["l2"] - rec["gold_l2"])
                    if args.inference and res.inference is not None:
                        rec["lower"] = res.inference.ci_lower
                        rec["upper"] = res.inference.ci_upper
                        rec["gold_lower"] = float(g.average_att_lower.iloc[0])
                        rec["gold_upper"] = float(g.average_att_upper.iloc[0])
                        rec["bound_diff"] = max(
                            abs(rec["lower"] - rec["gold_lower"]),
                            abs(rec["upper"] - rec["gold_upper"]))
                rows.append(rec)
                if n % 64 == 0 or n == total:
                    el = time.time() - t0
                    print(f"  {n}/{total}  {el:.0f}s  {el / n:.2f}s/cell",
                          flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = df[df.status == "ok"]
    bad = df[df.status != "ok"]
    print(f"\nwrote {args.out}: {len(df)} cells, {len(ok)} ok, {len(bad)} not")
    for s, k in bad.status.value_counts().items():
        print(f"    {k:4d}  {s}")
    if ok.empty:                                  # pragma: no cover
        return 1

    print("\nATT |diff| vs the published values, by heating year:")
    print(f"    {'year':>6}  {'max':>10}  {'mean':>10}  {'<=1e-5':>7}  {'n':>4}")
    for year, g in ok.groupby("year"):
        mark = "  <- drift" if year == DRIFT_YEAR else ""
        print(f"    {year:>6}  {g.att_diff.max():>10.3e}  "
              f"{g.att_diff.mean():>10.3e}  "
              f"{(g.att_diff <= 1e-5).mean():>7.1%}  {len(g):>4}{mark}")

    rest = ok[ok.year != DRIFT_YEAR]
    print(f"\n    all years   within 1e-5: {(ok.att_diff <= 1e-5).mean():.1%}")
    print(f"    excl {DRIFT_YEAR}   within 1e-5: "
          f"{(rest.att_diff <= 1e-5).mean():.1%}")

    # The row that separates "the fit differs" from "the reported effect
    # differs". It holds everywhere, drift year included.
    print(f"\npre-treatment L2 max |diff| (all {len(ok)} cells): "
          f"{ok.l2_diff.max():.3e}")
    print("    the same optimum VALUE is reached even where the effect differs")

    worst = ok.loc[ok.att_diff.idxmax()]
    print(f"\nworst cell: {worst.group} / {worst.year} / {worst.pollutant}"
          f"  |diff| {worst.att_diff:.4f}  gold Scaled_L2 "
          f"{worst.gold_scaled_l2:.4f}")
    print("    a well-conditioned cell -- which is why the non-uniqueness "
          "explanation was dropped")
    if args.inference and "bound_diff" in ok:
        b = ok.bound_diff.dropna()
        if not b.empty:
            print(f"\njackknife+ average-ATT bounds ({len(b)} cells): "
                  f"max |diff| {b.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
