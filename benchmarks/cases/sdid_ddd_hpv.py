"""Path A benchmark: SDID synthetic triple difference (SC-DDD) on Virginia's
HPV vaccine mandate (Feldman & Semprini 2026; method = Zhuang 2024).

Cross-validates mlsynth's ``SDID`` SC-DDD mode against the authors' Stata
``sdid`` on their own data (``jsemprini/Virginia_HPVmandate_causal``,
``final-npcr-hpv2024.csv``, here vendored slim as
``basedata/hpv_cervical_ddd.csv``: 39 states x 17 years, cervical-cancer
age-adjusted incidence by 5-year age band, 2003-2019, from public NPCR/SEER).

The design: Virginia's 2008 school-entry HPV mandate; the first exposed cohort
reaches ages 20-24 in 2016 (the target subgroup); older age bands (30-49) are
the non-target controls that the Zhuang transform demeans out. mlsynth demeans
the outcome by the non-target ages within each treatment-group-by-year cell,
then runs SDID on the 20-24 subgroup with Virginia treated from 2016.

The paper reports (cases per 100,000):

  ====================  =================  =============
  Estimator             mlsynth            Stata sdid
  ====================  =================  =============
  SC-DDD (transformed)  +1.559             +1.559
  naive SC-DD (20-24)   +0.252             +0.252
  ====================  =================  =============

The SC-DDD point estimate matches the Stata ``sdid`` output to three decimals
(the transform is exact and mlsynth's SDID engine is itself cross-validated
against the ``synthdid`` R package). The naive SC-DD on the untransformed
outcome reproduces the paper's near-null 0.252, confirming the triple-difference
demeaning is what turns the sign positive and significant.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def run() -> dict:
    from mlsynth import SDID

    df = pd.read_csv(_BASE / "hpv_cervical_ddd.csv")
    df["treated"] = ((df["state"] == "Virginia") & (df["year"] >= 2016)
                     & (df["age"] == "20-24")).astype(int)

    scddd = SDID({
        "df": df, "outcome": "cervix_adj", "treat": "treated",
        "unitid": "state", "time": "year",
        "subgroup": "age", "target_subgroup": "20-24",
        "display_graphs": False,
    }).fit()

    naive = SDID({
        "df": df[df["age"] == "20-24"].copy(), "outcome": "cervix_adj",
        "treat": "treated", "unitid": "state", "time": "year",
        "display_graphs": False,
    }).fit()

    return {
        "n_states": float(df["state"].nunique()),
        "scddd_att": float(scddd.effects.att),
        "scddd_ci_lower_positive": float(scddd.inference.ci_lower > 0),
        "naive_scdd_att": float(naive.effects.att),
    }


# Deterministic point estimates (the transform + SDID weights are a fixed convex
# solve on a fixed panel; only placebo SEs use the RNG). Targets are the paper's
# Stata sdid cells; the SC-DDD point estimate is a cell match.
EXPECTED = {
    "n_states": (39.0, 0.0),
    "scddd_att": (1.559, 0.03),          # Feldman & Semprini SC-DDD = +1.559
    "scddd_ci_lower_positive": (1.0, 0.0),  # placebo CI excludes zero
    "naive_scdd_att": (0.252, 0.03),     # paper SC-DD (untransformed) = +0.252
}
