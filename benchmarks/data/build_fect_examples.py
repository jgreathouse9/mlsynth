"""Rebuild the two Liu, Wang & Xu (2022) example panels from their replication package.

    python benchmarks/data/build_fect_examples.py <replication-dir> basedata/

``<replication-dir>`` is the ``replication/`` folder of the AJPS Dataverse
package for Liu, Licheng, Ye Wang and Yiqing Xu (2022), "A Practical Guide to
Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional
Data", DOI 10.7910/DVN/ZVC9W5. Its ``data/`` holds ``hh2015.dta`` and
``fm2015.dta``.

Nothing is rescaled, filtered or imputed: the Stata files are read and written
as Parquet with the integer keys narrowed. The estimation samples the paper
reports are formed downstream by ``complete.cases``, which the study applies
itself so the dropped rows stay visible. See basedata/fect_examples.README.md.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

PANELS = {
    "hh2015.dta": ("hh2015_naturalization.parquet", ["bfs", "year"]),
    "fm2015.dta": ("fm2015_council_grants.parquet", ["councilnumber", "year"]),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="the replication/ directory of the Dataverse package")
    ap.add_argument("out", help="output directory (basedata/)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    for stata_name, (parquet_name, keys) in PANELS.items():
        path = src / "data" / stata_name
        if not path.exists():
            raise SystemExit(f"missing {path}; point src at the replication/ directory")
        d = pd.read_stata(path)
        for k in keys:
            d[k] = d[k].astype("int64")
        target = out / parquet_name
        d.to_parquet(target, index=False, compression="zstd")
        print(f"{stata_name:<14} {len(d):>7,} rows x {d.shape[1]} cols -> {target} "
              f"({os.path.getsize(target) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
