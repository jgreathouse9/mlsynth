"""Assemble the paper's county panel: GFR 1937-1988 for the Bailey (2012) sample.

Zhang (2026) Section 9 builds the pseudo-distance on general fertility rates back
to 1937, which the Bailey replication package does not contain. The history comes
from ICPSR 36603 (Bailey, Clay, Fishback, Haines, Kantor, Severnini & Wentz,
"U.S. County-Level Natality and Mortality Data, 1915-2007", v2, 2018-05-02),
which carries a redistribution restriction and has to be downloaded directly:

    file  36603-0001-Data.dta   595,244 rows x 97 vars
          md5 3b28ff5652db68bc912712a642c6355c

Specified from the ICPSR 36603 codebook and from the handling in
hollina/duke-replication (Hollingsworth, Karbownik, Thomasson & Wray, "The Gift
of a Lifetime", AER), whose `analysis/scripts/code/2_clean_data.do` reads the
same file. NOT yet executed against the data.

What the codebook fixes:

    grain       county x year x race; race == 0 ("all") is 265,468 rows
    year        1915-2007
    cofips      County FIPS codes, 1990. min 1001, max 56047, 594,836 valid --
                the standard 5-digit key, matching Bailey's `fips` (verified at
                1001 to 56045 over 3,037 counties). NOT v_h_countycode, which is
                the ICPSR/Horan county code (10, 30, 50, ...) on another scheme.
    births_res  Births by place of residence      437,609 valid
    births_occ  Births by place of occurrence     453,535 valid
    popf        Female population 15 to 44 yo     568,403 valid (1930-2007)
    popf_fixed  same, revised in v2               567,203 valid

Three things the AER package shows that the codebook does not:

  1. The distributed .dta ships UPPERCASE names (it does `rename *, lower`), so
     columns are resolved case-insensitively here.
  2. 408 rows have a missing `cofips` and are dropped.
  3. County-year is NOT unique. That package tags duplicates and sums the counts
     across them before keeping one row, so counts are aggregated by (fips, year)
     here and the rate is computed after aggregation -- a ratio of sums, never a
     mean of ratios.

Zhang defines GFR as live births per 1000 women aged 15-44 by the mother's county
of residence, so the numerator is `births_res`. Since residence-based reporting
was phased in and `births_res` has fewer valid cases than `births_occ`, coverage
is reported by year; `--births occ` switches the numerator if the residence
series turns out to be too thin before 1959.

The overlap is the test of the construction: 1959-1988 exists in both sources, so
a correct series tracks Bailey's `gfr_nonint` closely. That is reported before
anything downstream uses 1937-1958. Per the paper's footnote the overlap then
keeps Bailey's own values.

Targets the paper states: 3,017 counties retained of Bailey's 3,037, and
treatment timing 6/43/74, 52/278, 63/75/53/10 (already verified present in the
Bailey file, so the check here is that the merge preserves it).

Usage: build_bailey_panel.py <36603-0001-Data.dta> <vs_fo_final.dta> [--births res|occ]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

HIST_FIRST, HIST_LAST = 1937, 1958
BAILEY_FIRST, BAILEY_LAST = 1959, 1988
GROUPS = {"1965-1967": (65, 67), "1968-1969": (68, 69), "1970-1973": (70, 73)}


def _resolve(path: str, wanted: list[str]) -> dict[str, str]:
    """Map lowercase names to however this file spells them (ICPSR ships UPPER)."""
    head = pd.io.stata.StataReader(path, chunksize=1, convert_dates=False).get_chunk(1)
    actual = {c.lower(): c for c in head.columns}
    missing = [w for w in wanted if w not in actual]
    if missing:
        raise SystemExit(f"columns absent from {path}: {missing}\nfound: {sorted(actual)[:40]}")
    return {w: actual[w] for w in wanted}


def load_history(path: str, births: str = "res") -> pd.DataFrame:
    num = f"births_{births}"
    cols = _resolve(path, ["cofips", "year", "race", num, "popf"])
    df = pd.read_stata(path, columns=list(cols.values()), convert_dates=False)
    df.columns = [c.lower() for c in df.columns]

    df = df[df.race == 0]
    df = df.dropna(subset=["cofips"])                       # 408 rows, per the AER package
    df["fips"] = df.cofips.astype(int)
    df["year"] = df.year.astype(int)

    hist = df[df.year.between(HIST_FIRST, HIST_LAST)]
    print(f"{num} coverage by year, {HIST_FIRST}-{HIST_LAST} (counties with both inputs):")
    cov = hist.groupby("year").apply(
        lambda g: int((g[num].notna() & g.popf.notna() & (g.popf > 0)).sum()),
        include_groups=False)
    for y in range(HIST_FIRST, HIST_LAST + 1, 4):
        window = [f"{yy}:{cov.get(yy, 0)}" for yy in range(y, min(y + 4, HIST_LAST + 1))]
        print("   " + "  ".join(window))
    thin = cov[cov < 2000]
    if len(thin):
        print(f"  WARNING: {len(thin)} years under 2,000 counties: {list(thin.index)}")

    # County-year is not unique: sum the counts, then form the rate.
    agg = (df.groupby(["fips", "year"], as_index=False)[[num, "popf"]].sum(min_count=1))
    agg = agg[(agg.popf > 0) & agg[num].notna()]
    agg["gfr"] = agg[num] / agg.popf * 1000.0
    return agg[["fips", "year", "gfr"]]


def load_bailey(path: str) -> pd.DataFrame:
    cols = _resolve(path, ["fips", "year", "gfr_nonint", "fp_year_p74_fed", "pop1544_70"])
    df = pd.read_stata(path, columns=list(cols.values()), convert_dates=False)
    df.columns = [c.lower() for c in df.columns]
    df["fips"] = df.fips.astype(int)
    df["year"] = df.year.astype(int) + 1900        # Bailey codes the year as 59..88
    return df


def main() -> None:
    births = "occ" if "--births" in sys.argv and "occ" in sys.argv else "res"
    hist_all = load_history(sys.argv[1], births)
    bail = load_bailey(sys.argv[2])

    ov = hist_all.merge(bail[["fips", "year", "gfr_nonint"]], on=["fips", "year"], how="inner")
    print()
    if len(ov):
        a, b = ov.gfr.to_numpy(float), ov.gfr_nonint.to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"overlap {BAILEY_FIRST}-{BAILEY_LAST}: {int(ok.sum()):,} county-years")
        print(f"  correlation      {np.corrcoef(a[ok], b[ok])[0, 1]:.4f}")
        print(f"  mean constructed {a[ok].mean():8.2f}   mean Bailey {b[ok].mean():8.2f}")
        print(f"  median |diff|    {np.median(np.abs(a[ok] - b[ok])):8.3f}")
        print(f"  (high correlation and a matching scale validate births_{births}/popf)")
    else:
        print("no overlap rows -- check the FIPS or year coding before going further")

    hist = hist_all[hist_all.year.between(HIST_FIRST, HIST_LAST)]
    bail_gfr = bail[["fips", "year", "gfr_nonint"]].rename(columns={"gfr_nonint": "gfr"})
    panel = pd.concat([hist, bail_gfr], ignore_index=True).sort_values(["fips", "year"])

    span = (HIST_LAST - HIST_FIRST + 1) + (BAILEY_LAST - BAILEY_FIRST + 1)
    counts = panel.groupby("fips").gfr.apply(lambda s: int(s.notna().sum()))
    complete = counts[counts == span].index
    panel = panel[panel.fips.isin(complete)]

    print()
    print(f"panel {HIST_FIRST}-{BAILEY_LAST} ({span} years)")
    print(f"  complete-history counties {len(complete):,}   (paper: 3,017)")
    print(f"  of Bailey's {bail.fips.nunique():,};  rows {len(panel):,}")

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
