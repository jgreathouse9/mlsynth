"""Render ``results/timing.csv`` as the table in the README.

Per size, the median over panels of each time, and the geometric mean over
panels of each speedup, the right average for ratios.

    python report.py results/timing.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def gmean(x) -> float:
    return float(np.exp(np.log(np.asarray(x, dtype=float)).mean()))


def table(frame: pd.DataFrame) -> str:
    lines = ["| markets | candidates | search | SCIP, wall | SCIP, solver only "
             "| SCIP nodes | speedup, wall | speedup, solver only | enumeration |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (J, m), g in frame.groupby(["J", "m"]):
        enum = (f"{g['enumerate_secs'].median():.2f}s"
                if "enumerate_secs" in g and g["enumerate_secs"].notna().any()
                else "not run")
        nodes = sorted(int(str(n).strip("[]")) for n in g["scip_nodes"])
        lines.append(
            f"| {J} (m = {m}) | {int(g['candidates'].iloc[0]):,} "
            f"| {g['search_secs'].median():.3f}s "
            f"| {g['scip_wall'].median():.2f}s "
            f"| {g['scip_solving'].median():.2f}s "
            f"| {nodes[0]:,} to {nodes[-1]:,} "
            f"| {gmean(g['speedup_wall']):.0f}x "
            f"| {gmean(g['speedup_solving']):.0f}x | {enum} |")
    return "\n".join(lines)


def main(path: str) -> None:
    frame = pd.read_csv(path)
    print(table(frame))
    print(f"\ninstances: {len(frame)}   all match SCIP: {bool(frame['match'].all())}"
          f"   node counts deterministic: {bool(frame['nodes_deterministic'].all())}"
          f"   load average range: {frame['load_before'].min():.2f}"
          f" to {frame['load_after'].max():.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/timing.csv")
