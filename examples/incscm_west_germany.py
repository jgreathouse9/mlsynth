"""Inclusive SCM (Di Stefano & Mellace 2024) on German reunification.

Runs SPILLSYNTH(method='iscm') for West Germany, treating Austria as the
spillover-affected neighbour. Demonstrates three configurations:

  1. outcome-only matching,
  2. covariate matching with the Malo bilevel backend,
  3. covariate matching with the MSCMT bilevel backend,

so the effect of the solver choice on the synthesis is visible.

Run from the repository root:

    python examples/incscm_west_germany.py
"""

import pandas as pd

from mlsynth import SPILLSYNTH

COVARIATES = ["trade", "infrate", "industry", "schooling", "invest80"]


def load_panel() -> pd.DataFrame:
    d = pd.read_stata("basedata/repgermany.dta")
    d = d[["country", "year", "gdp", *COVARIATES]].copy()
    # Treatment: West Germany from reunification (1990) onward.
    d["treat"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


def run(d: pd.DataFrame, *, covariates=None, bilevel_solver="malo") -> None:
    cfg = {
        "df": d, "outcome": "gdp", "treat": "treat",
        "unitid": "country", "time": "year",
        "method": "iscm", "affected_units": ["Austria"],
        "display_graphs": False,
    }
    if covariates:
        cfg["covariates"] = covariates
        cfg["bilevel_solver"] = bilevel_solver

    res = SPILLSYNTH(cfg).fit()
    f = res.iscm
    tag = f"covariates + {bilevel_solver}" if covariates else "outcome-only"
    print(f"\n=== inclusive SCM: {tag} ===")
    print(f"  ATT (inclusive) = {res.att:+.1f}   |   ATT (naive SCM) = {res.att_scm:+.1f}")
    print("  cross-weights : " + ", ".join(f"{k} = {v:.3f}" for k, v in f.cross_weights.items()))
    print(f"  det(Omega)    = {f.omega_det:.3f}")
    print(f"  pre-RMSPE     = {f.pre_rmspe:.1f} (inclusive)  vs  "
          f"{f.pre_rmspe_restricted:.1f} (Austria excluded)")
    print("  spillover ATT : " + ", ".join(f"{k} = {v:+.1f}" for k, v in f.spillover_att.items()))
    if f.predictor_weights:
        top = sorted(f.predictor_weights.items(), key=lambda x: -x[1])[:3]
        print("  top predictors: " + ", ".join(f"{n} = {v:.3f}" for n, v in top))


if __name__ == "__main__":
    panel = load_panel()
    run(panel)
    run(panel, covariates=COVARIATES, bilevel_solver="malo")
    run(panel, covariates=COVARIATES, bilevel_solver="mscmt")
