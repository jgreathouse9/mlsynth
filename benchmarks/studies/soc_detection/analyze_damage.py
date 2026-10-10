"""Paired comparison of the configurations in ``results/damage.csv``.

Every configuration is compared to the baseline on the same program under the
same row and column order, so the ratio removes what the program and the
ordering contribute and leaves what the configuration does. Effort is
summarised by geometric means of those ratios, the right average for factors,
and by how many pairs each configuration wins. The root dual bound is compared
directly: it measures the relaxation each configuration builds, which is the
thing the handler's participation can change.

    python analyze_damage.py results/damage.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

KEY = ["J", "m", "seed", "order"]


def main(path: str) -> None:
    df = pd.read_csv(path)
    assert (df["status"] == "optimal").all(), df.loc[df["status"] != "optimal"]

    spread = df.groupby(KEY)["objective"].agg(lambda s: s.max() - s.min())
    print(f"pairs: {len(spread)}   largest objective spread within a pair: "
          f"{spread.max():.2e}\n")

    base = df[df["config"] == "baseline"].set_index(KEY)
    rows = []
    for config in [c for c in df["config"].unique() if c != "baseline"]:
        other = df[df["config"] == config].set_index(KEY).loc[base.index]
        r = {}
        for col in ("nodes", "lp_iters", "secs", "nonlinear_cuts"):
            ratio = (other[col].clip(lower=1) / base[col].clip(lower=1))
            r[f"{col}_gmean"] = float(np.exp(np.log(ratio).mean()))
            r[f"{col}_wins"] = f"{int((ratio < 1).sum())}/{len(ratio)}"
        diff = other["root_bound"] - base["root_bound"]
        r["root_higher"] = f"{int((diff > 1e-9).sum())}/{len(diff)}"
        r["root_median_diff"] = float(diff.median())
        r["soc"] = f"{int(other['soc'].min())}..{int(other['soc'].max())}"
        rows.append(dict(config=config, **r))
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.float_format",
                           "{:.3f}".format):
        print(out.to_string(index=False))

    print("\nby panel size (geometric mean of lp_iters ratio vs baseline):")
    for J, grp in df.groupby("J"):
        b = grp[grp["config"] == "baseline"].set_index(KEY)
        line = [f"J={J:3d}"]
        for config in ("no_aggregation", "no_expansion", "neither"):
            o = grp[grp["config"] == config].set_index(KEY).loc[b.index]
            g = float(np.exp(np.log(o["lp_iters"].clip(lower=1)
                                    / b["lp_iters"].clip(lower=1)).mean()))
            line.append(f"{config}={g:.3f}")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/damage.csv")
