"""Rebuild a CEPR ORG extract from the Stata releases.

The committed `basedata/cepr_org_retail.parquet` is built by this script and
nothing else:

    python benchmarks/data/build_cepr_org.py <src-dir> \\
        basedata/cepr_org_retail.parquet --industry "Retail Trade" \\
        --from 1979 --to 1991

``<src-dir>`` holds one directory per year, each containing the CEPR release
``cepr_org_<year>.dta`` (https://ceprdata.org/cps-uniform-data-extracts/).
33 of the 162 source variables are kept; the occupation recodes occ70 / occ80
are dropped because their Stata value labels collide on reread. Nothing is
rescaled, filtered or imputed beyond the column, industry and year selection.

Drop ``--industry`` to keep every industry (the ``ind_nber`` column is then
retained), and ``--list-industries`` prints the labels available. The full
extract across all industries and 1979-1993 is about 94 MB, which is why only
the slice the California minimum-wage example uses is committed. See
basedata/cepr_org_retail.README.md.
"""
import argparse
import glob
import os

import pandas as pd

KEEP = ["year", "month", "minsamp", "state", "metro", "age", "female", "wbho",
        "educ", "married", "ind_nber", "ind70", "ind80", "empl", "selfemp",
        "pubsect", "unmem", "uncov", "uhours", "hourslw", "paidhre", "weekpay",
        "earnwke", "wage1", "wage2", "wage3", "wage4", "rw", "orgwgt", "fnlwgt",
        "tc", "blsimph", "blsimpw"]


def load(src: str, year_from: int, year_to: int) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(os.path.join(src, "*", "cepr_org_*.dta"))):
        year = int(os.path.basename(p).split("_")[-1][:4])
        if not year_from <= year <= year_to:
            continue
        d = pd.read_stata(p, columns=KEEP, convert_categoricals=True)
        d["year"] = d["year"].astype(float).astype(int)
        for c in d.select_dtypes(["category"]).columns:
            d[c] = d[c].astype("string")
        frames.append(d)
    if not frames:
        raise SystemExit(f"no cepr_org_<year>.dta under {src!r} in "
                         f"{year_from}-{year_to}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="directory of per-year CEPR release folders")
    ap.add_argument("out", nargs="?", help="output .parquet path")
    ap.add_argument("--industry", default=None,
                    help="ind_nber label to keep, e.g. 'Retail Trade'")
    ap.add_argument("--from", dest="year_from", type=int, default=1979)
    ap.add_argument("--to", dest="year_to", type=int, default=1993)
    ap.add_argument("--list-industries", action="store_true")
    args = ap.parse_args()

    a = load(args.src, args.year_from, args.year_to)
    if args.list_industries:
        print(a["ind_nber"].value_counts(dropna=False).to_string())
        return
    if not args.out:
        ap.error("an output path is required unless --list-industries is given")

    if args.industry is not None:
        a = a[a["ind_nber"].astype(str) == args.industry].drop(columns=["ind_nber"])
        if a.empty:
            raise SystemExit(f"no rows for industry {args.industry!r}; "
                             "run --list-industries to see the labels")
    a = a.sort_values(["year", "state"])
    a.to_parquet(args.out, index=False, compression="zstd")
    print(f"{len(a):,} rows x {a.shape[1]} cols, "
          f"{args.year_from}-{args.year_to} -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
