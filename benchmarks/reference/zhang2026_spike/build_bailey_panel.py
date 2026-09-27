"""Assemble the paper's county panel: GFR 1937-1988 for the Bailey (2012) sample.

Zhang (2026) Section 9 builds the pseudo-distance on general fertility rates back
to 1937, which the Bailey replication package does not contain. The history comes
from ICPSR 36603 (Bailey, Clay, Fishback, Haines, Kantor, Severnini & Wentz,
"U.S. County-Level Natality and Mortality Data, 1915-2007", v2, 2018-05-02).

Specified from the ICPSR 36603 codebook; NOT yet executed against the data file,
which was absent from the supplied download. What the codebook fixes:

    file      36603-0001-Data.dta   595,244 rows x 97 vars
              md5 3b28ff5652db68bc912712a642c6355c
    grain     county x year x race, race in {0 all, 1 white, 2 non white};
              race == 0 is 265,468 rows
    year      1915-2007
    cofips    county FIPS, combined 5-digit (max 56,047), matching Bailey's
              `fips` (verified: 1001 to 56045 over 3,037 counties)
    births_res  Births by place of residence      437,609 valid
    popf        Female population 15 to 44 yo     568,403 valid  (1930-2007)
    popf_fixed  same, revised in v2               567,203 valid

GFR is live births per 1000 women aged 15-44, so `births_res / popf * 1000`.

The overlap is the test of that construction: 1959-1988 appears in both sources,
so a constructed series that is right will track Bailey's `gfr_nonint` closely.
The script reports that agreement before anything downstream uses 1937-1958.
Following the paper's footnote 21, the overlap keeps Bailey's own values.

Targets to hit, both stated in the paper: 3,017 counties retained of Bailey's
3,037, and treatment timing 6/43/74, 52/278, 63/75/53/10 (already verified
present in the Bailey file, so the check here is that the merge preserves it).

Usage: build_bailey_panel.py <path-to-36603-0001-Data.dta> <path-to-vs_fo_final.dta>
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

HIST_FIRST, HIST_LAST = 1937, 1958
BAILEY_FIRST, BAILEY_LAST = 1959, 1988
GROUPS = {"1965-1967": (65, 67), "1968-1969": (68, 69), "1970-1973": (70, 73)}


def load_history(path: str) -> pd.DataFrame:
    df = pd.read_stata(path, columns=["cofips", "year", "race", "births_res", "popf"],
                       convert_dates=False)
    df = df[(df.race == 0) & df.year.between(HIST_FIRST, 2007)]
    df = df.dropna(subset=["cofips", "births_res", "popf"])
    df = df[df.popf > 0]
    df["fips"] = df.cofips.astype(int)
    df["gfr"] = df.births_res / df.popf * 1000.0
    return df[["fips", "year", "gfr"]].astype({"year": int})


def load_bailey(path: str) -> pd.DataFrame:
    df = pd.read_stata(path, columns=["fips", "year", "gfr_nonint",
                                      "fp_year_p74_fed", "pop1544_70"],
                       convert_dates=False)
    df["fips"] = df.fips.astype(int)
    # Bailey codes the year as two digits (59..88).
    df["year"] = df.year.astype(int) + 1900
    return df


def main() -> None:
    hist = load_history(sys.argv[1])
    bail = load_bailey(sys.argv[2])

    overlap = hist.merge(
        bail[["fips", "year", "gfr_nonint"]], on=["fips", "year"], how="inner")
    if len(overlap):
        a, b = overlap.gfr.to_numpy(float), overlap.gfr_nonint.to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"overlap {BAILEY_FIRST}-{BAILEY_LAST}: {ok.sum():,} county-years")
        print(f"  correlation      {np.corrcoef(a[ok], b[ok])[0,1]:.4f}")
        print(f"  mean constructed {a[ok].mean():8.2f}   mean Bailey {b[ok].mean():8.2f}")
        print(f"  median |diff|    {np.median(np.abs(a[ok]-b[ok])):8.3f}")
        print("  (a high correlation and a matching scale validate births_res/popf)")
    else:
        print("no overlap rows -- check the FIPS or year coding before proceeding")

    hist = hist[hist.year.between(HIST_FIRST, HIST_LAST)]
    bail_gfr = bail[["fips", "year", "gfr_nonint"]].rename(columns={"gfr_nonint": "gfr"})
    panel = pd.concat([hist, bail_gfr], ignore_index=True).sort_values(["fips", "year"])

    span = HIST_LAST - HIST_FIRST + 1 + BAILEY_LAST - BAILEY_FIRST + 1
    counts = panel.groupby("fips").gfr.apply(lambda s: s.notna().sum())
    complete = counts[counts == span].index
    panel = panel[panel.fips.isin(complete)]

    print()
    print(f"panel {HIST_FIRST}-{BAILEY_LAST} ({span} years)")
    print(f"  counties with a complete history {len(complete):,}   (paper: 3,017)")
    print(f"  of Bailey's {bail.fips.nunique():,}")
    print(f"  rows {len(panel):,}")

    timing = bail.groupby("fips").fp_year_p74_fed.first()
    timing = timing[timing.index.isin(complete)].dropna().astype(int)
    print("  retained treatment timing:")
    for name, (lo, hi) in GROUPS.items():
        sub = timing[(timing >= lo) & (timing <= hi)]
        per = ", ".join(str(int((sub == y).sum())) for y in range(lo, hi + 1))
        print(f"    {name}: {per}   total {len(sub)}")

    out = "bailey_gfr_1937_1988.parquet"
    panel.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
