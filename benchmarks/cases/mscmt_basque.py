"""MSCMT cross-validation: Abadie-Gardeazabal's Basque study, the MSCMT spec.

Cross-validation (the reference is the R package MSCMT, Becker & Klossner 2018,
"Computing Generalized Synthetic Controls with the R package MSCMT"). The MSCMT
vignette replicates Abadie & Gardeazabal's (2003) study of the per-capita GDP
cost of ETA terrorism in the Basque Country, fitting ``gdpcap`` over the
optimisation window 1960-1969 -- even though the panel begins in 1955 -- with
their thirteen-predictor specification (schooling, investment, sector shares,
population density, and lagged GDP, each averaged over its own window).

This reproduces the MSCMT result value-for-value through ``VanillaSC.fit()``,
using ``backend="mscmt"`` (the nested predictor-weight ``V`` search) with
``fit_window=(1960, 1969)`` (MSCMT's ``times.dep``) and the AG predictor windows:

    =======================  ===============  =========================
    Quantity                 mlsynth VanillaSC  MSCMT (vignette)
    =======================  ===============  =========================
    Cataluna                 0.6328           0.63279
    Baleares (Islas)         0.2193           0.21931
    Madrid (Comunidad De)    0.1479           0.14790
    avg post gap 1970-1990   -0.7709          -0.77096
    =======================  ===============  =========================

The donor weights match to four decimals; the post-period gap (averaged over
1970-1990, MSCMT's ``did`` range) matches MSCMT's ``average.post`` of -0.770963.
The fit window is essential: without it the outcome SSR would run over the full
1955-1969 pre-period and the weights would drift off the MSCMT solution.

The reference numbers were produced by installing MSCMT (and Synth) from source
and running its vignette specification with ``outer.optim="DEoptim"`` and
``seed=42``; see ``benchmarks/R/mscmt_basque.R``. Data ship as
``basedata/basque_mscmt.csv`` (the MSCMT-transformed ``basque`` panel: schooling
rescaled to per-unit percentage shares, ``school.higher = school.high +
school.post.high``, Spain excluded from the donor pool downstream).

Provenance: Becker & Klossner (2018), MSCMT vignette "Working with package
MSCMT"; Abadie & Gardeazabal (2003), AER 93(1).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_mscmt.csv")

_W16 = (1964, 1969)   # schooling + investment
_W19 = (1961, 1969)   # sector shares
_COVS = ["school.illit", "school.prim", "school.med", "school.higher", "invest",
         "gdpcap", "sec.agriculture", "sec.energy", "sec.industry",
         "sec.construction", "sec.services.venta", "sec.services.nonventa",
         "popdens"]
_WINDOWS = {
    "school.illit": _W16, "school.prim": _W16, "school.med": _W16,
    "school.higher": _W16, "invest": _W16, "gdpcap": (1960, 1969),
    "sec.agriculture": _W19, "sec.energy": _W19, "sec.industry": _W19,
    "sec.construction": _W19, "sec.services.venta": _W19,
    "sec.services.nonventa": _W19, "popdens": (1969, 1969),
}


def run() -> dict:
    from mlsynth import VanillaSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= 1970)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": d, "outcome": "gdpcap", "treat": "treat",
            "unitid": "regionname", "time": "year",
            "backend": "mscmt", "canonical_v": "min.loss.w",
            "covariates": _COVS, "covariate_windows": _WINDOWS,
            "fit_window": (1960, 1969),       # MSCMT's times.dep
            "mscmt_maxiter": 400, "mscmt_popsize": 20, "seed": 42,
            "display_graphs": False,
        }).fit()

    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    years = np.array(sorted(d["year"].unique()))
    gap = np.asarray(res.time_series.estimated_gap, dtype=float)
    post = (years >= 1970) & (years <= 1990)
    n_pos = sum(1 for v in w.values() if v > 1e-3)

    return {
        "cataluna": w.get("Cataluna", 0.0),
        "baleares": w.get("Baleares (Islas)", 0.0),
        "madrid": w.get("Madrid (Comunidad De)", 0.0),
        "three_donor_mass": (w.get("Cataluna", 0.0)
                             + w.get("Baleares (Islas)", 0.0)
                             + w.get("Madrid (Comunidad De)", 0.0)),
        "n_positive_donors": float(n_pos),
        "avg_post_gap_70_90": float(gap[post].mean()),
    }


# Deterministic (DE search is seeded). VanillaSC's mscmt backend, fit over the
# MSCMT optimisation window with the AG predictor spec, reproduces the MSCMT
# vignette value-for-value: Cataluna ~0.633, Baleares ~0.219, Madrid ~0.148 carry
# all the weight, and the 1970-1990 gap averages ~-0.771 (MSCMT did -0.770963).
EXPECTED = {
    "cataluna": (0.63279, 0.006),
    "baleares": (0.21931, 0.006),
    "madrid": (0.14790, 0.006),
    "three_donor_mass": (1.0, 0.01),
    "n_positive_donors": (3.0, 1.0),
    "avg_post_gap_70_90": (-0.77096, 0.02),
}
