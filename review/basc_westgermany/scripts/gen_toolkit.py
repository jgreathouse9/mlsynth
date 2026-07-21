"""Generate the standard-SC-toolkit comparison data for the West Germany review.

Runs mlsynth's VanillaSC (outcome-only, Malo bilevel, MSCMT nested-DE with the
ADH 2015 predictor spec) and CLUSTERSC (RPCA-SC, PCR/RSC) on the Abadie et al.
(2015) repgermany panel, and writes counterfactual paths, in-sample (pre-1990)
RMSEs, and donor weights to ``../data/``. Reads ``repgermany.dta`` from the
BASC repo (github.com/sll-lee/paper-BASC); set REPGERMANY below.
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from mlsynth import VanillaSC, CLUSTERSC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REPGERMANY = os.environ.get(
    "REPGERMANY",
    "/tmp/claude-0/-home-user-mlsynth/cd2795fa-9dba-53ec-9a57-542cdf057bb0/scratchpad/basc/repo/data/repgermany.dta")

raw = pd.read_stata(REPGERMANY)
covs = ["gdp_avg", "infrate", "trade", "industry", "schooling", "invest80"]
d = raw[["country", "year", "gdp", "infrate", "trade", "industry", "schooling", "invest80"]].copy()
d["gdp_avg"] = d["gdp"]
d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
# ADH 2015 Table 1 averaging windows
windows = {"gdp_avg": (1981, 1990), "infrate": (1981, 1990), "trade": (1981, 1990),
           "industry": (1981, 1990), "schooling": (1980, 1985), "invest80": (1980, 1985)}
years = np.array(sorted(d.year.unique()))
obs = raw[raw.country == "West Germany"].sort_values("year")["gdp"].to_numpy()

paths = {"year": years, "observed": obs}
rmse_rows, weight_cols = [], {}

def record(name, res, label):
    cf = np.asarray(res.time_series.counterfactual_outcome, float)
    paths[name] = cf
    w = res.weights.donor_weights or {}
    weight_cols[label] = {k: round(float(v), 3) for k, v in w.items() if v > 1e-3}
    rmse_rows.append({"method": label,
                      "in_sample_rmse": round(float(res.fit_diagnostics.rmse_pre), 1),
                      "n_donors": int(sum(1 for v in w.values() if v > 0.01))})

base = dict(df=d, outcome="gdp", treat="treat", unitid="country", time="year",
            display_graphs=False)
record("vanillasc_outcome", VanillaSC({**base}).fit(), "VanillaSC (outcome-only)")
record("vanillasc_malo", VanillaSC({**base, "backend": "malo", "covariates": covs,
        "covariate_windows": windows, "fit_window": (1960, 1989), "seed": 0}).fit(),
       "VanillaSC (malo, covariates)")
record("vanillasc_mscmt", VanillaSC({**base, "backend": "mscmt", "covariates": covs,
        "covariate_windows": windows, "fit_window": (1960, 1989), "seed": 0}).fit(),
       "VanillaSC (mscmt, ADH 2015 spec)")

dc = pd.read_csv(os.path.join(HERE, "..", "..", "..", "basedata", "german_reunification.csv"))
dc["treat"] = dc["Reunification"].astype(int)
cbase = dict(df=dc, outcome="gdp", treat="treat", unitid="country", time="year",
             display_graphs=False)
record("clustersc_rpca", CLUSTERSC({**cbase, "method": "rpca", "rpca_method": "PCP"}).fit(),
       "CLUSTERSC (RPCA-SC)")
record("clustersc_pcr", CLUSTERSC({**cbase, "method": "PCR"}).fit(), "CLUSTERSC (PCR/RSC)")

# uniform-weight diagnostic
wide = d.pivot(index="year", columns="country", values="gdp")
pre = wide.index < 1990
order = [c for c in wide.columns if c != "West Germany"]
X = wide[order].to_numpy()
unif_rmse = float(np.sqrt(np.mean((obs[pre] - X[pre] @ np.full(len(order), 1 / len(order))) ** 2)))
rmse_rows.append({"method": "Uniform weights (1/16)", "in_sample_rmse": round(unif_rmse, 1),
                  "n_donors": len(order)})

pd.DataFrame(paths).to_csv(os.path.join(DATA, "toolkit_counterfactuals.csv"), index=False)
pd.DataFrame(rmse_rows).to_csv(os.path.join(DATA, "insample_rmse_toolkit.csv"), index=False)
# ADH 2015 published weights (Comparative Politics, Table)
adh = {"Austria": 0.42, "USA": 0.22, "Japan": 0.16, "Switzerland": 0.11, "Netherlands": 0.09}
weight_cols["ADH 2015 (published)"] = adh
wdf = pd.DataFrame(weight_cols).fillna(0.0).round(3)
wdf.index.name = "donor"
wdf.to_csv(os.path.join(DATA, "weights_comparison.csv"))
print("wrote toolkit_counterfactuals.csv, insample_rmse_toolkit.csv, weights_comparison.csv")
print(pd.DataFrame(rmse_rows).to_string(index=False))
