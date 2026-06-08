"""Cross-validation benchmark: SpSyDiD vs the authors' reference algorithm.

Path B (Monte Carlo, scenario 3 -- full reference repo). Reproduces the
State-Level Simulation of Serenini & Masek (2024),
``State_Level_Simulations.ipynb`` from https://github.com/serenini/spatial_SDID,
and cross-validates ``mlsynth.SpSyDiD`` **per replication** against the authors'
own estimator (their SDID weight functions + the notebook's spatial WLS,
driven by :mod:`benchmarks.reference.spsydid_ref`).

DGP (matches the notebook)
--------------------------
* Panel: 49 contiguous US states, monthly unemployment 1976-2014
  (``state_unemployment.csv``); queen-contiguity W (``US_no_islands_matrix.gal``),
  row-standardised.
* For each 3-year rolling window (post = the 3rd year, so T0 = 24, T1 = 12),
  Arkansas (FIPS 5) is the directly-treated state. Outcome
  ``UR2 = perc_unem + interaction*ATT + spillover*ATT*rho`` with ATT = 25% of
  the window's mean unemployment and rho = 0.8. Panels are deterministic.

Provenance
----------
* Data: ``basedata/{state_unemployment.csv,US_no_islands_matrix.gal}``
  (vendored from the authors' ``Data/`` directory).
* Reference: serenini/spatial_SDID @ e43427d (pinned), cloned on demand by
  :mod:`benchmarks.reference.clone_spsydid` -- the repo carries no licence, so it
  is not vendored.

Across the windows mlsynth and the reference agree per-rep to ~0.1 pp on the ATT
(correlation > 0.99); both recover the paper's headline finding that the mean ATT
bias of the spatial estimator is essentially zero. The small per-rep residual is
the affected-unit weight convention (mlsynth: 1/N_sp; reference: mean treated-unit
SDID weight) -- both valid downstream of the SDID weight QPs.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_BASE = Path(__file__).resolve().parents[2] / "basedata"
N_WINDOWS = 20         # deterministic windows from 1975; 40 available
RHO = 0.8
TREATED_FIPS = 5       # Arkansas


def _load_state_panel_and_W() -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    try:
        import libpysal
    except ImportError as exc:  # pragma: no cover - optional dep
        raise BenchmarkSkipped("libpysal not installed "
                               "(needed to read the .gal weights)") from exc
    df = pd.read_csv(_BASE / "state_unemployment.csv")
    wq = libpysal.io.open(str(_BASE / "US_no_islands_matrix.gal")).read()
    wq.transform = "r"
    W1, _ = wq.full()
    FIPS = [int(i) for i in wq.id_order]

    df = df[df["FIPS"].isin(FIPS)]
    df = df[df["State"] != "Los Angeles County"]
    df["FIPS"] = pd.Categorical(df["FIPS"], categories=FIPS, ordered=True)
    df = df.sort_values(["year", "month", "FIPS"]).reset_index(drop=True)
    df = df.rename(columns={"FIPS": "ID"})
    df["ID"] = df["ID"].astype(int)
    return df, W1, FIPS


def _build_rep_panel(data0, W1, year, treated_fips, ATT, rho):
    from scipy.linalg import block_diag
    data = data0[(data0["year"] > year) & (data0["year"] < year + 4)].copy()
    n_units = len(data["ID"].unique())
    data["month"] = np.repeat(np.arange(1, 37), n_units)
    data["after_treatment"] = (data["year"] == year + 3)
    data["treatment"] = (data["ID"] == treated_fips)
    data["interaction"] = (data["after_treatment"] & data["treatment"]).astype(int)
    W = block_diag(*[W1] * 36)
    data["spillover"] = W.dot(data["interaction"].values)
    data["UR2"] = (data["perc_unem"] + data["interaction"] * ATT
                   + data["spillover"] * ATT * rho)
    wd_vals = data.loc[data["spillover"] > 0, "spillover"]
    WD = float(wd_vals.mean()) if len(wd_vals) else 0.0
    return data, WD


def run() -> dict:
    from mlsynth import SpSyDiD
    from benchmarks.reference.spsydid_ref import reference_ssdid

    df, W1, FIPS = _load_state_panel_and_W()

    ml_atts, ref_atts, true_atts = [], [], []
    for year in range(1975, 1975 + N_WINDOWS):
        if year + 3 > df["year"].max():
            break
        window = df.loc[(df["year"] > year) & (df["year"] < year + 4), "perc_unem"]
        ATT = float(window.mean()) / 4.0
        if np.isnan(ATT):
            continue
        panel, WD = _build_rep_panel(df, W1, year, TREATED_FIPS, ATT, RHO)
        if panel.empty:
            continue
        panel["treat_indicator"] = (panel["treatment"] & panel["after_treatment"]).astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SpSyDiD({
                "df": panel[["ID", "month", "UR2", "treat_indicator"]],
                "outcome": "UR2", "treat": "treat_indicator",
                "unitid": "ID", "time": "month",
                "spatial_matrix": W1, "unit_order": list(FIPS),
                "row_standardize_spatial": False, "display_graphs": False,
            }).fit()
        ref_att, _, _ = reference_ssdid(panel, WD)

        ml_atts.append(float(res.att))
        ref_atts.append(ref_att)
        true_atts.append(ATT)

    ml = np.asarray(ml_atts)
    ref = np.asarray(ref_atts)
    true = np.asarray(true_atts)
    return {
        "n_reps": float(len(ml)),
        "spsydid_max_abs_att_diff_vs_ref": float(np.max(np.abs(ml - ref))),
        "spsydid_att_corr_vs_ref": float(np.corrcoef(ml, ref)[0, 1]),
        "spsydid_mean_att_bias_mlsynth": float(np.mean(true - ml)),
        "spsydid_mean_att_bias_reference": float(np.mean(true - ref)),
    }


# Deterministic panels => exact re-runs. Tolerances: per-rep ATT agreement with
# the reference is bracketed at 0.2 pp (observed max ~0.09, residual is the
# affected-unit weight convention); the per-rep correlation must exceed 0.98
# (1.0 +/- 0.02); and both estimators' mean ATT bias must be ~0 (|bias| < 0.1
# against an ATT magnitude of ~1.7 pp -- the paper's headline unbiasedness).
EXPECTED = {
    "n_reps": (20.0, 0.0),
    "spsydid_max_abs_att_diff_vs_ref": (0.0, 0.2),
    "spsydid_att_corr_vs_ref": (1.0, 0.02),
    "spsydid_mean_att_bias_mlsynth": (0.0, 0.1),
    "spsydid_mean_att_bias_reference": (0.0, 0.1),
}
