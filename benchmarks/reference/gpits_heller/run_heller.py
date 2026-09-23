"""Heller / D.C. Path-A target: cumulative 4-month effect 15.1 [13.0, 17.3]."""
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

print(f"panel: {df.state.nunique()} jurisdictions, "
      f"{df.date.nunique()} months {df.date.min():%Y-%m} to {df.date.max():%Y-%m}")

dc = df[df.state == "District of Columbia"].sort_values("date")
n_pre = int((dc.date < TREAT).sum())
print(f"D.C.: T={len(dc)}  T0={n_pre}  post={len(dc) - n_pre}")
print(f"      pre-period handgun_rate mean={dc.handgun_rate[:n_pre].mean():.4f} "
      f"max={dc.handgun_rate[:n_pre].max():.4f}")
print(f"      post values: {np.round(dc.handgun_rate.values[n_pre:], 3)}")

res = gp_its(dc.handgun_rate.values, dc.month_f.values, n_pre)

print(f"\nhyperparameters: b={res['b']:.4f}  s2={res['s2']:.4f}")
print(f"\n{'m':>2} {'observed':>9} {'counterfac':>11} {'tau_t':>8} "
      f"{'tau_cum':>9} {'cum 95% CI':>20}")
obs = dc.handgun_rate.values[n_pre:]
for i in range(len(obs)):
    print(f"{i+1:>2} {obs[i]:>9.3f} {res['counterfactual'][i]:>11.3f} "
          f"{res['tau_t'][i]:>8.3f} {res['tau_cum'][i]:>9.3f} "
          f"[{res['tau_cum_lwr'][i]:>7.3f},{res['tau_cum_upr'][i]:>7.3f}]")

print(f"\nPAPER  : cumulative 4-month effect 15.1 per 100k, 95% CI [13.0, 17.3]")
print(f"PORT   : {res['tau_cum'][-1]:.4f} per 100k, "
      f"95% CI [{res['tau_cum_lwr'][-1]:.4f}, {res['tau_cum_upr'][-1]:.4f}]")
pop_dc = dc.population.iloc[-1]
print(f"         ~{res['tau_cum'][-1] * pop_dc / 1e5:.1f} additional checks "
      f"(paper: ~90; D.C. population {pop_dc:,.0f})")
