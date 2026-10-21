"""Tables for the detection and RRSC arm.

    python analyze_detection.py results/detection.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main(path: str) -> None:
    d = pd.read_csv(path)
    rms = lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))

    print("=" * 92)
    print("The screen: does it find the contaminated market, and what else does it flag?")
    print("=" * 92)
    s = d.groupby(["dgp", "selection"]).agg(
        reps=("seed", "nunique"), controls=("n_controls", "mean"),
        recall=("screen_found_kstar", "mean"),
        flagged=("n_flagged", "mean"), false_pos=("n_false_pos", "mean"))
    s["flagged_share"] = s.flagged / s.controls
    print(s.round(3).to_string())

    print("\nFalse positives at zero contamination (nothing to find):")
    z = d[d.ratio == 0.0]
    print(z.groupby(["dgp", "selection"]).agg(flagged=("n_flagged", "mean"),
                               false_pos=("n_false_pos", "mean"),
                               recall=("screen_found_kstar", "mean")).round(2).to_string())

    print("\n" + "=" * 92)
    print("What detection error costs: repair told the market, against repair given the screen")
    print("=" * 92)
    rows = []
    for (name, sel, r), g in d.groupby(["dgp", "selection", "ratio"]):
        rows.append(dict(dgp=name, selection=sel, ratio=r,
                         naive=rms(g.naive - g.oracle),
                         known=rms(g.known - g.oracle),
                         detected=rms(g.detected - g.oracle)))
    t = pd.DataFrame(rows)
    print(t.pivot(index=["dgp", "selection"], columns="ratio").round(3).to_string())

    print("\nAveraged over contamination sizes above zero:")
    pos = d[d.ratio > 0]
    agg = []
    for (name, sel), g in pos.groupby(["dgp", "selection"]):
        agg.append(dict(dgp=name, selection=sel, naive=rms(g.naive - g.oracle),
                        known=rms(g.known - g.oracle),
                        detected=rms(g.detected - g.oracle),
                        detected_vs_known=rms(g.detected - g.oracle) - rms(g.known - g.oracle)))
    print(pd.DataFrame(agg).round(3).to_string(index=False))

    print("\n" + "=" * 92)
    print("RRSC, reported only where it passes the clean-panel applicability gate")
    print("=" * 92)
    pos = pos[pos.selection == "S1"]          # RRSC is independent of the screen
    gate = d.drop_duplicates(["dgp", "seed"]).groupby("dgp").rrsc_gate_ok.mean()
    rows = []
    for name, g in pos.groupby("dgp"):
        ok = g[g.rrsc_gate_ok]
        rows.append(dict(dgp=name, gate_pass=float(gate[name]), n_gated=len(ok),
                         rrsc=rms(ok.rrsc - ok.oracle) if len(ok) else np.nan,
                         known=rms(ok.known - ok.oracle) if len(ok) else np.nan,
                         naive=rms(ok.naive - ok.oracle) if len(ok) else np.nan))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nAgainst the true effect instead of the design's own oracle:")
    rows = []
    for name, g in pos.groupby("dgp"):
        ok = g[g.rrsc_gate_ok]
        if not len(ok):
            continue
        rows.append(dict(dgp=name, rrsc=rms(ok.rrsc - ok.tau),
                         known=rms(ok.known - ok.tau), naive=rms(ok.naive - ok.tau),
                         oracle=rms(ok.oracle - ok.tau)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
