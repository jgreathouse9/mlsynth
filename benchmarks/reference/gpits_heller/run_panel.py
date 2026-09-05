"""All 50 jurisdictions: does D.C. stand out against a nationwide null?

The paper's second substantive claim (Section 6, Figure 4): fitting an
independent GP per jurisdiction should isolate a large D.C. effect and leave
the remaining 49 near zero.
"""
import json

import numpy as np
import pandas as pd
import pyreadr

from gpits_port import gp_its

REPO = "gpits"
START, TREAT, END = "2002-07-01", "2008-07-01", "2008-11-01"

nics = pd.read_csv(f"{REPO}/data/nics/NICS_state_month_11.1998_1.2024.csv")
pops = list(pyreadr.read_r(f"{REPO}/data/census_pop/pops_by_state.RDS").values())[0]
pops["year"] = pops["year"].astype(int)
df = nics[nics.state != "Hawaii"].merge(pops, on=["state", "year"], how="left")
df["handgun_rate"] = df["handgun"] / df["population"] * 100_000
df["longgun_rate"] = df["long_gun"] / df["population"] * 100_000
df["date"] = pd.to_datetime(df["date"] + "-01")
df = df[df.handgun_rate.notna() & df.longgun_rate.notna()]
df = df[(df.date >= START) & (df.date < END)].copy()
df["month_f"] = df["date"].dt.strftime("%m")

rows = []
for state, g in df.groupby("state"):
    g = g.sort_values("date")
    n_pre = int((g.date < TREAT).sum())
    r = gp_its(g.handgun_rate.values, g.month_f.values, n_pre)
    cum, se = r["tau_cum"][-1], r["tau_cum_se"][-1]
    rows.append(dict(state=state, cum=float(cum), se=float(se),
                     lwr=float(cum - 1.959963985 * se),
                     upr=float(cum + 1.959963985 * se),
                     pre_sd=float(np.std(g.handgun_rate.values[:n_pre], ddof=1)),
                     b=float(r["b"]), s2=float(r["s2"])))

res = pd.DataFrame(rows).sort_values("cum", ascending=False).reset_index(drop=True)
res["sig"] = (res.lwr > 0) | (res.upr < 0)
res["std_cum"] = res.cum / res.pre_sd

print("top 5 and bottom 5 by cumulative 4-month effect (per 100k):")
print(res.head(5).to_string(
    index=False, columns=["state", "cum", "lwr", "upr", "sig"],
    float_format=lambda v: f"{v:8.3f}"))
print("...")
print(res.tail(5).to_string(
    index=False, columns=["state", "cum", "lwr", "upr", "sig"],
    float_format=lambda v: f"{v:8.3f}"))

dc = res[res.state == "District of Columbia"].iloc[0]
others = res[res.state != "District of Columbia"]
print(f"\nD.C. rank by raw cumulative effect: {res.index[res.state == 'District of Columbia'][0] + 1} of {len(res)}")
print(f"D.C.   : {dc.cum:8.3f} [{dc.lwr:.3f}, {dc.upr:.3f}]  "
      f"standardised by own pre-SD: {dc.std_cum:.2f}")
print(f"others : median {others.cum.median():8.3f}   "
      f"mean {others.cum.mean():.3f}   sd {others.cum.std():.3f}")
print(f"         range [{others.cum.min():.3f}, {others.cum.max():.3f}]")
print(f"\nsignificant at 95% among the other 49: {int(others.sig.sum())}")
if others.sig.any():
    print(others[others.sig].to_string(
        index=False, columns=["state", "cum", "lwr", "upr"],
        float_format=lambda v: f"{v:8.3f}"))

res.to_json("panel_results.json", orient="records", indent=2)
print("\nwrote panel_results.json")
