"""Consolidate the per-seed R outputs into the three CSVs the report reads.

The R scripts write one file per configuration, chain length and seed. The
report reads three tables from `data/`. This merges the former into the latter,
keyed on configuration, chain length and seed, so a freshly computed row
replaces its shipped counterpart and rows this run did not compute survive. A
short run therefore updates what it produced and leaves the long chains intact
instead of deleting them and breaking the render.

    python scripts/collect_results.py
"""
import glob
import os
import sys

import pandas as pd

TARGETS = {
    "gamma1_diagnostic.csv": "gamma1_results_*.csv",
    "effect_basis_diagnostic.csv": "q_results_*.csv",
    "selection_decomposition.csv": "decomp_*.csv",
}
KEY = ["config", "N", "seed"]

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for target, pattern in TARGETS.items():
        out = os.path.join("data", target)
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"{out}: nothing matching {pattern}, keeping the shipped copy")
            continue

        fresh = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
        if os.path.exists(out):
            shipped = pd.read_csv(out)
            key = [c for c in KEY if c in fresh.columns and c in shipped.columns]
            # fresh first, so drop_duplicates keeps the newly computed row
            merged = (pd.concat([fresh, shipped], ignore_index=True)
                        .drop_duplicates(subset=key, keep="first") if key else fresh)
            kept = len(merged) - len(fresh)
        else:
            merged, kept = fresh, 0

        sort_cols = [c for c in KEY if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols).reset_index(drop=True)
        merged.to_csv(out, index=False)

        lengths = sorted(merged["N"].unique()) if "N" in merged else []
        seeds = sorted(merged["seed"].unique()) if "seed" in merged else []
        print(f"wrote {out}: {len(fresh)} row(s) recomputed, {kept} carried over, "
              f"chain lengths {lengths}, seeds {seeds}")

    g = os.path.join("data", "gamma1_diagnostic.csv")
    if os.path.exists(g):
        d = pd.read_csv(g)
        if not ((d.config == "basc") & (d.N == 25000)).any():
            print("\nThe 25000/25000 control is absent and the report reads it for the "
                  "BASC row. Run with --full.", file=sys.stderr)
            sys.exit(1)
