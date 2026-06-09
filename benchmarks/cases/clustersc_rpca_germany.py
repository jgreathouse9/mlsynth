"""Path A benchmark: CLUSTERSC RPCA-SC on West German reunification (Bayani 2021).

The CLUSTERSC PCR-SC / RSC path is already pinned by ``clustersc_subgroups``,
``rsc_synth_error`` and ``rsc_shen_coverage``; this case closes the gap for the
estimator's *other* family -- **RPCA-SC** (robust low-rank L+S donor denoising,
Bayani 2021, *Robust PCA Synthetic Control*) -- which had no durable check.

Reproduces the canonical German-reunification application: with the PCP (Candes,
Li, Ma & Wright 2011) robust-PCA denoiser, RPCA-SC builds West Germany's
synthetic control from a sparse donor set dominated by Norway and France, with a
close pre-period fit -- matching Bayani's reference figures and the classical
Abadie-Diamond-Hainmueller result that reunification depressed West German
per-capita GDP.

Provenance
----------
* Data: ``basedata/german_reunification.csv`` -- annual GDP per capita for 17
  countries, 1960-2003; West Germany treated from reunification (1990).
* Headline: Bayani (2021) RPCA-SC weights -- Norway ~0.48, France ~0.35,
  New Zealand ~0.30 -- with pre-fit RMSE ~90; the estimated effect is negative.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"


def run() -> dict:
    from mlsynth import CLUSTERSC

    df = pd.read_csv(_BASE / "german_reunification.csv")
    df["treat"] = df["Reunification"].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC({
            "df": df, "outcome": "gdp", "treat": "treat",
            "unitid": "country", "time": "year",
            "method": "rpca", "rpca_method": "PCP", "display_graphs": False,
        }).fit()

    w = res.donor_weights or {}
    return {
        "norway_weight": float(w.get("Norway", 0.0)),
        "france_weight": float(w.get("France", 0.0)),
        "newzealand_weight": float(w.get("New Zealand", 0.0)),
        "pre_rmse": float(res.pre_rmse),
        # 1.0 iff reunification's estimated effect on West German GDP is negative
        # (the classical ADH finding).
        "att_negative": float(res.att < 0.0),
    }


# Deterministic (PCP is a fixed-point ADMM, no RNG). RPCA-SC must recover the
# Bayani reference design -- a Norway+France-dominated sparse synthetic West
# Germany with close pre-fit -- and a negative reunification effect. Tolerances
# absorb solver/library drift around the published weights and pre-RMSE.
EXPECTED = {
    "norway_weight": (0.485, 0.06),       # Bayani ~0.48
    "france_weight": (0.354, 0.06),       # Bayani ~0.35
    "newzealand_weight": (0.296, 0.08),   # Bayani ~0.30
    "pre_rmse": (88.6, 10.0),             # docs ~90
    "att_negative": (1.0, 0.0),
}
