"""All 1024 cells of Song et al.'s main_result.R, against their published values.

    python benchmarks/reference/song_ml_ascm/run_full_sweep.py
    python benchmarks/reference/song_ml_ascm/run_full_sweep.py --inference

The routine benchmark (``benchmarks/cases/song_ml_ascm.py``) pins a stratified
30-cell subset so it stays cheap. This runs the whole design: 8 heating years x
8 treatment groups x 16 pollutant series, the same grid their ``main_result.csv``
contains. It writes one row per cell to ``full_sweep.csv`` and prints a summary
split by whether the pre-treatment fit is well conditioned.

The split matters. Where the treated unit is an almost exact convex combination of
the donors the simplex optimum is not unique, so two solvers can reach the same
objective value and disagree on the post-treatment extrapolation. Pooling those
cells with the rest would report a max disagreement of ~1 and hide the fact that
the well-conditioned cells agree to ~1e-6. Cells are classified by their gold
``Scaled_L2``, i.e. by conditioning, not by whether they happened to disagree.

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
    GROUPS, POLLUTANTS, WINDOWS, _DEGENERATE_SCALED_L2, _PANEL, _REF,
    _donors, _pre_fit_l2, fit_cell)


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

    well = ok[ok.gold_scaled_l2 >= _DEGENERATE_SCALED_L2]
    degen = ok[ok.gold_scaled_l2 < _DEGENERATE_SCALED_L2]
    print(f"\nwell-conditioned cells ({len(well)}):")
    print(f"    ATT   max |diff| {well.att_diff.max():.3e}   "
          f"mean {well.att_diff.mean():.3e}")
    print(f"    preL2 max |diff| {well.l2_diff.max():.3e}")
    print(f"\nnear-degenerate cells ({len(degen)}), gold Scaled_L2 < "
          f"{_DEGENERATE_SCALED_L2}:")
    if not degen.empty:
        print(f"    ATT   max |diff| {degen.att_diff.max():.3e}   "
              f"mean {degen.att_diff.mean():.3e}")
        print(f"    preL2 max |diff| {degen.l2_diff.max():.3e}   "
              f"<- the same optimum VALUE is still reached")
        worst = degen.loc[degen.att_diff.idxmax()]
        print(f"    worst: {worst.group} / {worst.year} / {worst.pollutant}"
              f"  Scaled_L2 {worst.gold_scaled_l2:.4f}")
    if args.inference and "bound_diff" in ok:
        b = ok.bound_diff.dropna()
        if not b.empty:
            print(f"\njackknife+ average-ATT bounds ({len(b)} cells): "
                  f"max |diff| {b.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
