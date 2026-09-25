"""Section 7: Tables 9, 10 and 11-16, each against the paper's own column.

    python -m benchmarks.studies.hsiao_zhou_counterfactuals.run_empirics

Needs the paper's replication data, which is not shipped here. Point
MLSYNTH_HZ_DATA at the directory holding smoking.csv,
smoking-health-expenditure.csv and turnout.csv; the turnout panel is the one
mlsynth already ships as basedata/xu_edr_turnout.parquet, row for row.
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

from .empirics import build, mab, run_cell

warnings.filterwarnings("ignore")

DATA = pathlib.Path(os.environ.get("MLSYNTH_HZ_DATA", "hz-data"))

# Table 9's MAB row and Table 10's, as published.
T9 = dict(SCM=18.5, PCA=7.46, CCE=9.12, CPDA=9.56, PDA=14.3, PDAX=16.2,
          MA=8.33, MB=7.27)
T10 = dict(SCM=0.128, PCA=0.059, CCE=0.053, CPDA=0.085, PDA=0.061,
           PDAX=0.061, MA=0.062, MB=0.047)
# Tables 11-16, the mean effect implied by each state's MA and MB columns.
TURNOUT = {"ME": (3.22, 2.85), "MN": (4.49, 4.52), "WI": (8.39, 8.59),
           "WY": (7.20, 8.22), "ID": (-0.42, 0.64), "NH": (8.04, 8.28)}
ADOPT = {"ME": 1976, "MN": 1976, "WI": 1976, "WY": 1996, "ID": 1996,
         "NH": 1996}


def table(name, Y, X, T0, paper, r=2, rule="cce"):
    y1 = Y[:, 0]
    paths, beta, nsel = run_cell(Y, X, T0, r, rule)
    print(f"\n{name}   beta = {rule}, r = {r}, beta = {np.round(beta, 4)}")
    print(f"  {'method':6} {'mine':>9} {'paper':>8} {'ratio':>6}")
    for m in ("PCA", "CCE", "CPDA", "PDA", "PDAX", "MA", "MB"):
        v = mab(y1, paths[m], T0)
        print(f"  {m:6} {v:9.3f} {paper[m]:8.3f} {v / paper[m]:6.2f}")
    return paths


def main() -> int:
    if not DATA.is_dir():
        print(f"{DATA} not found; set MLSYNTH_HZ_DATA", file=sys.stderr)
        return 2

    df = pd.read_csv(DATA / "smoking.csv")
    Y, X, years, T0 = build(df, "state", "year", "cigsale",
                            ["lnincome", "EduAttain", "Poverty"], 1, 1989)
    table("Table 9, cigarette consumption", Y, X, T0, T9)

    from mlsynth import VanillaSC
    long = df.rename(columns={"state": "unit", "year": "time",
                              "cigsale": "y"}).copy()
    long["D"] = ((long.unit == 1) & (long.time >= 1989)).astype(int)
    sc = VanillaSC(dict(df=long, outcome="y", treat="D", unitid="unit",
                        time="time", display_graphs=False)).fit()
    cf = np.asarray(sc.time_series.counterfactual_outcome, dtype=float).ravel()
    v = float(np.mean(np.abs(Y[T0:, 0] - cf[T0:])))
    print(f"  {'SCM':6} {v:9.3f} {T9['SCM']:8.3f} {v / T9['SCM']:6.2f}"
          "   (mlsynth VanillaSC, outcome only)")

    h = pd.read_csv(DATA / "smoking-health-expenditure.csv")
    Yh, Xh, _, T0h = build(h, "state", "year", "lnhexpense", ["lnincome"],
                           1, 1989)
    table("Table 10, personal healthcare expenditure", Yh, Xh, T0h, T10)

    t = pd.read_csv(DATA / "turnout.csv")
    donors = sorted(set(t.abb.unique()) - set(t[t.policy_edr == 1].abb.unique()))
    print("\nTables 11-16, turnout: mean effect by state")
    print(f"  {'state':6} {'MA mine':>8} {'paper':>7} {'MB mine':>8} {'paper':>7}")
    waves = {1976: [], 1996: []}
    for st, y0 in ADOPT.items():
        sub = t[t.abb.isin([st] + donors)]
        Yt, Xt, _, T0t = build(sub, "abb", "year", "turnout",
                               ["policy_mail_in", "policy_motor"], st, y0)
        paths, _, _ = run_cell(Yt, Xt, T0t, 2, "cce")
        eff = {m: float(np.mean(Yt[T0t:, 0] - paths[m][T0t:]))
               for m in ("MA", "MB")}
        pa, pb = TURNOUT[st]
        print(f"  {st:6} {eff['MA']:8.2f} {pa:7.2f} {eff['MB']:8.2f} {pb:7.2f}")
        waves[y0].append((eff["MA"], pa))
    for y0, label, xu in ((1976, "first wave", 7.2), (1996, "second wave", 2.17)):
        mine = np.mean([a for a, _ in waves[y0]])
        pap = np.mean([b for _, b in waves[y0]])
        print(f"  {label}: mine {mine:.2f}, paper's own columns {pap:.2f}, "
              f"Xu (2017) {xu}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
