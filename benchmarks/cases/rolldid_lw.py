"""Path A benchmark: ROLLDID reproduces Lee & Wooldridge (2026) — Prop 99 + castle.

Reproduces both empirical applications of the rolling-transformation DiD paper
("Simple Approaches to Inference with DiD … Small Cross-Sectional Sample Sizes")
to the reported precision, and cross-validates the common-timing point estimates
against the AGPL ``lwdid`` package used **only as a black-box oracle** (skipped
if absent; mlsynth's implementation is clean-room from the paper equations and
shares no code with it).

Provenance
----------
* California Prop 99: ``basedata/smoking_data.csv`` (Abadie et al. 2010 panel,
  39 states x 1970-2000, California treated 1989), outcome = log per-capita
  cigarette sales. Paper Table 3: demean ATT -0.422 (se 0.121), detrend -0.227
  (se 0.094), detrend exact-p 0.021.
* Castle laws: ``basedata/castle.csv`` (Cunningham 2021, 50 states 2000-2010,
  21 staggered-treated / 29 never-treated), outcome = log homicides. Paper
  §7.2: demean aggregate 0.092 (OLS se 0.057), detrend 0.067 (HC3 se 0.055).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def _smoking() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "smoking_data.csv")
    d["logcig"] = np.log(d["cigsale"])
    d["treat"] = d["Proposition 99"].astype(int)
    return d


def _castle() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "castle.csv")
    d["W"] = ((d["effyear"].notna()) & (d["year"] >= d["effyear"])).astype(int)
    return d


def run() -> dict:
    from mlsynth import ROLLDID

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sm = _smoking()
        dm = ROLLDID({"df": sm, "outcome": "logcig", "treat": "treat",
                      "unitid": "state", "time": "year", "rolling": "demean",
                      "inference": "exact", "display_graphs": False}).fit()
        dt = ROLLDID({"df": sm, "outcome": "logcig", "treat": "treat",
                      "unitid": "state", "time": "year", "rolling": "detrend",
                      "inference": "exact", "display_graphs": False}).fit()
        out["prop99_demean_att"] = float(dm.effects.att)
        out["prop99_demean_se"] = float(dm.inference.standard_error)
        out["prop99_detrend_att"] = float(dt.effects.att)
        out["prop99_detrend_se"] = float(dt.inference.standard_error)
        out["prop99_detrend_exact_p"] = float(dt.inference.p_value)

        ca = _castle()
        cdm = ROLLDID({"df": ca, "outcome": "l_homicide", "treat": "W",
                       "unitid": "state", "time": "year", "rolling": "demean",
                       "inference": "exact", "display_graphs": False}).fit()
        cdt = ROLLDID({"df": ca, "outcome": "l_homicide", "treat": "W",
                       "unitid": "state", "time": "year", "rolling": "detrend",
                       "inference": "hc3", "display_graphs": False}).fit()
        out["castle_demean_att"] = float(cdm.effects.att)
        out["castle_demean_se"] = float(cdm.inference.standard_error)
        out["castle_detrend_att"] = float(cdt.effects.att)
        out["castle_detrend_hc3_se"] = float(cdt.inference.standard_error)

    return out


# Paper-reported values (Table 3 and §7.2) with display-rounding tolerances.
# This is a clean-room Path-A reproduction against the *published* numbers; the
# triangulation against the AGPL ``lwdid`` package was done during the
# demonstrate-first step and is deliberately not a committed dependency here.
EXPECTED = {
    "prop99_demean_att": (-0.422, 5e-3),
    "prop99_demean_se": (0.121, 5e-3),
    "prop99_detrend_att": (-0.227, 5e-3),
    "prop99_detrend_se": (0.094, 5e-3),
    "prop99_detrend_exact_p": (0.021, 2e-3),
    "castle_demean_att": (0.092, 3e-3),
    "castle_demean_se": (0.057, 3e-3),
    "castle_detrend_att": (0.067, 3e-3),
    "castle_detrend_hc3_se": (0.055, 3e-3),
}
