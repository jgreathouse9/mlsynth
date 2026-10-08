"""Tables from the end-to-end arm.

    python analyze.py results/end_to_end.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ARMS = ("naive", "iterative", "renorm")


def main(path: str) -> None:
    d = pd.read_csv(path)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))

    print(f"replications per DGP: {d.groupby('dgp').seed.nunique().to_dict()}")
    print(f"max |iterative - iscm| = {np.abs(d.iterative - d.iscm).max():.2e}")

    dev = np.abs((d[d.ratio > 0].naive - d[d.ratio > 0].oracle)
                 + d[d.ratio > 0].vk * d[d.ratio > 0].pi).max()
    print(f"max deviation from  naive - oracle = -v_k * pi:  {dev:.2e}\n")

    print("=" * 84)
    print("Threshold calibration, as the end-to-end arm sees it")
    print("=" * 84)
    cal = []
    for name, g in d.groupby("dgp"):
        u = g.drop_duplicates("seed")
        truth = rms(u.e_post)
        cal.append(dict(dgp=name, reps=len(u), mean_vk=u.vk.mean(), truth=truth,
                        thr_in=rms(u.thr_in), in_ratio=rms(u.thr_in) / truth,
                        thr_oos=rms(u.thr_oos), oos_ratio=rms(u.thr_oos) / truth))
    print(pd.DataFrame(cal).sort_values("oos_ratio").round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("Crossover by pi / thr_oos. A calibrated threshold puts the flip at 1.0")
    print("=" * 84)
    for name, g in d.groupby("dgp"):
        t = pd.DataFrame([
            dict(ratio=r, **{a: rms(h[a] - h.oracle) for a in ARMS})
            for r, h in g.groupby("ratio")])
        flip = t[t.iterative < t.naive].ratio.min()
        print(f"\n--- {name}  (flip at ratio {flip})")
        print(t.round(3).to_string(index=False))

    print("\n" + "=" * 84)
    print("Decision accuracy and cost: in-sample rule against out-of-sample rule")
    print("=" * 84)
    acc = []
    for name, g in d[d.ratio > 0].groupby("dgp"):
        err_n = (g.naive - g.oracle).abs()
        err_i = (g.iterative - g.oracle).abs()
        row = dict(dgp=name)
        for label, fires in (("in", g.rule_in), ("oos", g.rule_oos)):
            row[f"acc_{label}"] = float((fires == g.truth).mean())
            row[f"cost_{label}"] = float(np.mean(np.where(fires, err_i, err_n)))
        row["cost_best"] = float(np.minimum(err_n, err_i).mean())
        row["cost_always"] = float(err_i.mean())
        row["cost_never"] = float(err_n.mean())
        acc.append(row)
    print(pd.DataFrame(acc).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
