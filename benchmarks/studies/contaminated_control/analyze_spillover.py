"""Tables for the inclusive-pool arm.

    python analyze_spillover.py results/spillover_pool.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def main(path: str) -> None:
    d = pd.read_csv(path)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))

    print("=" * 86)
    print("The cross-weight l1: zero by construction in the clean pool, not in the other")
    print("=" * 86)
    u = d.drop_duplicates(["dgp", "seed", "pool"])
    print(u.groupby(["dgp", "pool"]).l1
          .agg(mean="mean", median="median", max="max",
               share_nonzero=lambda s: float((s > 1e-8).mean()))
          .round(3).to_string())

    print("\n" + "=" * 86)
    print("Prediction 1: with no treatment effect the arms agree even when l1 is large")
    print("=" * 86)
    z = d[(d.pool == "inclusive") & (d.tau_mult == 0.0)]
    print(f"  replications with l1 > 0.1: {int((z.drop_duplicates(['dgp','seed']).l1 > 0.1).sum())}")
    print(f"  max |iterative - iscm| at tau = 0: {np.abs(z.iterative - z.iscm).max():.2e}")

    print("\n" + "=" * 86)
    print("Prediction 2: the gap is the leak v_k * l1 * tau")
    print("=" * 86)
    g = d[(d.pool == "inclusive") & (d.tau_mult > 0)].copy()
    g["gap"] = g.iscm - g.iterative
    print(f"  correlation(gap, predicted leak) = {np.corrcoef(g.gap, g.predicted_leak)[0,1]:.4f}")
    fit = np.polyfit(g.predicted_leak, g.gap, 1)
    print(f"  slope = {fit[0]:.4f}   intercept = {fit[1]:.4f}   (slope 1, intercept 0 is exact)")
    print()
    print(g.groupby("tau_mult").apply(
        lambda h: pd.Series({"mean_gap": h.gap.mean(),
                             "mean_predicted_leak": h.predicted_leak.mean()}),
        include_groups=False).round(4).to_string())

    print("\n" + "=" * 86)
    print("What it costs: RMSE against the oracle, inclusive pool")
    print("=" * 86)
    rows = []
    for (name, tm), h in d[d.pool == "inclusive"].groupby(["dgp", "tau_mult"]):
        rows.append(dict(dgp=name, tau_mult=tm, mean_l1=h.l1.mean(),
                         naive=rms(h.naive - h.oracle),
                         iterative=rms(h.iterative - h.oracle),
                         iscm=rms(h.iscm - h.oracle)))
    t = pd.DataFrame(rows)
    t["iscm_better_by"] = t.iterative - t.iscm
    print(t.round(3).to_string(index=False))

    print("\n" + "=" * 86)
    print("Clean pool against inclusive pool: is admitting the treated markets ever right?")
    print("=" * 86)
    rows = []
    for (name, pool), h in d[d.tau_mult == 1.0].groupby(["dgp", "pool"]):
        rows.append(dict(dgp=name, pool=pool, mean_l1=h.l1.mean(),
                         rebuild_thr=h.thr.mean(),
                         iterative=rms(h.iterative - h.oracle),
                         iscm=rms(h.iscm - h.oracle)))
    print(pd.DataFrame(rows).pivot(index="dgp", columns="pool").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
