"""Rebuild basedata/vz_minimum_wage.csv from the Vaghul-Zipperer spreadsheets.

    git clone https://github.com/equitablegrowth/VZ_historicalminwage
    python benchmarks/data/build_vz_minimum_wage.py VZ_historicalminwage/rawdata \\
        basedata/vz_minimum_wage.csv

The published releases of that repository are produced by its Stata do-files and
are not checked into it, so this reads the raw change spreadsheets directly and
steps them into an annual panel. Each row of the source is a minimum-wage change
with an effective date; the level in force on 1 July of each year is taken,
1 July being the date California's 1988 increase took effect. See
basedata/cepr_org.README.md.
"""
import os
import sys

import numpy as np
import pandas as pd

YEARS = list(range(1979, 1992))
REF = pd.to_datetime([f"{y}-07-01" for y in YEARS])


def _stamp(d):
    return pd.to_datetime(dict(year=d.year, month=d.month, day=d.day))


def _level_at(df, col, when):
    d = df[df["date"] <= when]
    return float(d.sort_values("date")[col].iloc[-1]) if len(d) else np.nan


def build(rawdata: str) -> pd.DataFrame:
    st = pd.read_excel(os.path.join(rawdata, "VZ_StateMinimumWage_Changes.xlsx"))
    fed = pd.read_excel(os.path.join(rawdata, "VZ_FederalMinimumWage_Changes.xlsx"))
    st["date"], fed["date"] = _stamp(st), _stamp(fed)

    fed_lvl = {y: _level_at(fed, "Fed_mw", r) for y, r in zip(YEARS, REF)}
    rows = []
    for name, g in st.groupby("statename"):
        for y, r in zip(YEARS, REF):
            s = _level_at(g, "VZ_mw", r)
            rows.append({"state": name, "year": y, "state_mw": s,
                         "fed_mw": fed_lvl[y],
                         "eff_mw": np.nanmax([s if s == s else -np.inf, fed_lvl[y]])})
    mw = pd.DataFrame(rows)

    # First year after 1987 in which a state's own minimum rose above its 1987
    # level, on each of the two readings. They disagree wherever a statute rose
    # to a level at or below the federal floor.
    for col, out in (("eff_mw", "first_effective_hike"),
                     ("state_mw", "first_statute_hike")):
        w = mw.pivot(index="year", columns="state", values=col)
        base = w.loc[1987]
        first = {}
        for s in w.columns:
            b = base[s]
            hits = [y for y in range(1988, 1992)
                    if w.loc[y, s] == w.loc[y, s]
                    and (b != b or w.loc[y, s] > b + 1e-9)]
            first[s] = min(hits) if hits else np.nan
        mw[out] = mw["state"].map(first)
    return mw


if __name__ == "__main__":
    df = build(sys.argv[1])
    df.to_csv(sys.argv[2], index=False)
    print(f"{len(df)} state-year rows -> {sys.argv[2]}")
