"""Section 7: Tables 9, 10 and 11-16, each against the paper's own column.

    python -m benchmarks.studies.hsiao_zhou_counterfactuals.run_empirics

Runs anywhere out of the box. The turnout arm reads the panel this repository
already ships, `basedata/xu_edr_turnout.parquet`, which is the paper's own
`turnout.csv` row for row. The two smoking panels are not here, so with nothing
configured they are drawn from `standin.py` instead and the published columns
are not printed, because on a drawn panel there is nothing to compare against.

To run the real replication, point MLSYNTH_HZ_DATA at the directory holding
`smoking.csv` and `smoking-health-expenditure.csv`.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

from .empirics import build, mab, run_cell
from .standin import resolve_panels

warnings.filterwarnings("ignore")

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


def table(name, Y, X, T0, paper, r=2, rule="cce", real=True):
    y1 = Y[:, 0]
    paths, beta, nsel = run_cell(Y, X, T0, r, rule)
    print(f"\n{name}   beta = {rule}, r = {r}, beta = {np.round(beta, 4)}")
    if real:
        print(f"  {'method':6} {'mine':>9} {'paper':>8} {'ratio':>6}")
    else:
        print(f"  {'method':6} {'value':>9}   (stand-in panel; no comparison)")
    for m in ("PCA", "CCE", "CPDA", "PDA", "PDAX", "MA", "MB"):
        v = mab(y1, paths[m], T0)
        if real:
            print(f"  {m:6} {v:9.3f} {paper[m]:8.3f} {v / paper[m]:6.2f}")
        else:
            print(f"  {m:6} {v:9.3f}")
    return paths


def main() -> int:
    panels = resolve_panels(os.environ.get("MLSYNTH_HZ_DATA"))
    real = panels["real"]
    # The two smoking panels index by calendar year when real and from zero
    # when drawn, so the treatment period is read off the panel either way.
    df = panels["consumption"]
    cut = 1989 if real else int(df.year.min()) + 19
    print("smoking panels: " + ("the paper's" if real else
          "STAND-IN, drawn by standin.py -- set MLSYNTH_HZ_DATA for the real "
          "replication"))

    Y, X, years, T0 = build(df, "state", "year", "cigsale",
                            ["lnincome", "EduAttain", "Poverty"], 1, cut)
    table("Table 9, cigarette consumption", Y, X, T0, T9, real=real)

    from mlsynth import VanillaSC
    long = df.rename(columns={"state": "unit", "year": "time",
                              "cigsale": "y"}).copy()
    long["D"] = ((long.unit == 1) & (long.time >= cut)).astype(int)
    sc = VanillaSC(dict(df=long, outcome="y", treat="D", unitid="unit",
                        time="time", display_graphs=False)).fit()
    cf = np.asarray(sc.time_series.counterfactual_outcome, dtype=float).ravel()
    v = float(np.mean(np.abs(Y[T0:, 0] - cf[T0:])))
    if real:
        print(f"  {'SCM':6} {v:9.3f} {T9['SCM']:8.3f} {v / T9['SCM']:6.2f}"
              "   (mlsynth VanillaSC, outcome only)")
    else:
        print(f"  {'SCM':6} {v:9.3f}   (mlsynth VanillaSC, outcome only)")

    h = panels["expenditure"]
    cut_h = 1989 if real else int(h.year.min()) + 9
    Yh, Xh, _, T0h = build(h, "state", "year", "lnhexpense", ["lnincome"],
                           1, cut_h)
    table("Table 10, personal healthcare expenditure", Yh, Xh, T0h, T10,
          real=real)

    # The turnout arm always runs on the shipped panel, so it is a replication
    # whichever way the two above resolved.
    t = panels["turnout"]
    donors = sorted(set(t.abb.unique()) - set(t[t.policy_edr == 1].abb.unique()))
    print("\nTables 11-16, turnout (basedata/xu_edr_turnout.parquet): "
          "mean effect by state")
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
