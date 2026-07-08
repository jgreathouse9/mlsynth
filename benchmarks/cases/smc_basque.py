"""SMC cross-validation + Path A: the Basque Country / ETA-terrorism study.

Zhu, Rong J. B. (2023), *"Synthetic Matching Control Method,"* arXiv:2306.02584,
re-examines the Abadie & Gardeazabal (2003) study of the per-capita GDP cost of
ETA terrorism (17 Spanish regions, 1955-1997). SMC matches each donor to the
Basque Country by a univariate regression, then synthesises the matched controls
with box-``[0, 1]`` weights chosen by a Mallows/Cp unbiased-risk criterion.

Primary validation is a **cross-validation against the author's reference R
implementation** (``Code_SMC.R``): on the identical Basque matching matrix, the
mlsynth weight computation matches the reference ``SMCV`` (per-donor OLS + the
Cp box-QP solved by ``solve.QP``) to machine precision -- ``theta``, the box
weights, the combined coefficients, ``bias`` and ``sigma^2`` all agree to
``< 2e-13`` (see ``docs/replications/smc.rst`` for the harness).

This case runs the deterministic Algorithm 1 (outcome-only matching) through the
public estimator on ``basedata/basque_data.csv`` and reports the reproducible
counterfactual summary. The fit is deterministic -- the Cp penalty identifies the
weights, so no seeded V search is involved and the cells are exact re-runs.

  =========================  ==============  ==============================
  Quantity                   mlsynth SMC     note
  =========================  ==============  ==============================
  pre-period RMSE            ~0.048          tight pre-treatment fit
  mean post-1969 ATT         ~-0.858         terrorism depresses GDP/capita
  1997 gap                   ~-0.848
  donors (combined coef)     Murcia 0.63,    box-weighted matched controls
                             Madrid 0.37,
                             Castilla y Leon 0.24
  =========================  ==============  ==============================

Provenance: Zhu (2023) Section 5.2; the reference R ``Code_SMC.R``; Abadie &
Gardeazabal (2003) for the original study.
"""
from __future__ import annotations

import os

import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_data.csv")
_TREATED = "Basque Country (Pais Vasco)"


def run() -> dict:
    from mlsynth import SMC

    df = pd.read_csv(os.path.abspath(_DATA))
    df["treat"] = ((df["regionname"] == _TREATED) & (df["year"] >= 1970)).astype(int)
    res = SMC({
        "df": df, "outcome": "gdpcap", "treat": "treat",
        "unitid": "regionname", "time": "year", "display_graphs": False,
    }).fit()

    w = {str(k): float(v) for k, v in res.donor_weights.items()}
    gap = res.time_series.observed_outcome - res.time_series.counterfactual_outcome
    return {
        "att": float(res.att),
        "pre_rmse": float(res.fit_diagnostics.rmse_pre),
        "gap_1997": float(gap[-1]),
        "murcia_coef": w.get("Murcia (Region de)", 0.0),
        "madrid_coef": w.get("Madrid (Comunidad De)", 0.0),
    }


# Deterministic (Cp-identified weights; no V search). Exact re-runs.
EXPECTED = {
    "att": (-0.8575, 0.02),
    "pre_rmse": (0.0481, 0.01),
    "gap_1997": (-0.8484, 0.02),
    "murcia_coef": (0.625, 0.05),
    "madrid_coef": (0.370, 0.05),
}
