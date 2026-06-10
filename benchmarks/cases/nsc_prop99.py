"""NSC cross-validation: California Proposition 99 vs Tian's reference code.

Cross-validation (scenario: authors' reference implementation + canonical data).
Tian (2023), *"The Synthetic Control Method with Nonlinear Outcomes,"* revisits
Abadie-Diamond-Hainmueller's Proposition 99 study (Section 5.1, Table 2) with
the nonlinear synthetic-control estimator. The author's R implementation
(``benchmarks/R/nsc_tian2023_reference.R``) and the ADH smoking panel are public,
so we validate mlsynth's NSC against them directly.

The author's cross-validation of the penalty ``(a, b)`` is *stochastic* (it draws
a random held-in donor each fold, hence ``set.seed(123)`` in the application
script), so it does not port to Python. But the application is deterministic
given the *selected* penalty ``a* = 0.3, b* = 0.7`` reported in Table 2, so we
fix those and match:

  * the per-donor NSC weights against the author's Table 2 (all 38 donors,
    including the negative weights the method permits), and
  * the estimated effect path against the paper's reported figures -- the
    nonlinear SC reduces per-capita sales by ~9.5 packs in 1990, ~24.5 in 1995
    and ~28.7 in 2000, an average post-period effect near -19.

mlsynth's NSC is a faithful port of the reference QP (eigenvalue-scaled penalty,
the ``rbind(Z, -Z)`` negativity trick, distance-weighted L1), so the weights
match to a correlation of ~0.99; residual per-weight differences (<~0.025) come
from the standardisation convention and Table 2's 3-decimal rounding.

Provenance: Tian (2023), arXiv:2306.01967v1, Section 5.1, Table 2; the author's
NSC.R / App_Cal.R and the ADH ``smoking_data.csv`` panel.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "smoking_data.csv")

# Author's Table 2 "NSC Weight" column (a* = 0.3, b* = 0.7), all 38 donors.
_REF_WEIGHTS = {
    "Alabama": -0.015, "Arkansas": -0.057, "Colorado": 0.119, "Connecticut": 0.112,
    "Delaware": 0.0, "Georgia": 0.0, "Idaho": 0.183, "Illinois": 0.02,
    "Indiana": 0.0, "Iowa": 0.039, "Kansas": 0.0, "Kentucky": 0.0,
    "Louisiana": 0.0, "Maine": 0.0, "Minnesota": 0.027, "Mississippi": -0.007,
    "Missouri": 0.0, "Montana": 0.176, "Nebraska": 0.094, "Nevada": 0.091,
    "New Hampshire": 0.0, "New Mexico": 0.103, "North Carolina": 0.0,
    "North Dakota": 0.0, "Ohio": 0.0, "Oklahoma": 0.0, "Pennsylvania": 0.0,
    "Rhode Island": 0.0, "South Carolina": -0.003, "South Dakota": 0.0,
    "Tennessee": -0.071, "Texas": 0.0, "Utah": 0.045, "Vermont": 0.0,
    "Virginia": 0.0, "West Virginia": 0.083, "Wisconsin": 0.06, "Wyoming": 0.0,
}
# Paper's reported NSC effect path (per-capita packs).
_REF_ITE = {1990: -9.5, 1995: -24.5, 2000: -28.7}


def run() -> dict:
    from mlsynth import NSC

    d = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = NSC({
            "df": d, "outcome": "cigsale", "treat": "Proposition 99",
            "unitid": "state", "time": "year",
            "a": 0.3, "b": 0.7,                  # Table-2 selected penalty
            "run_inference": True, "display_graphs": False,
        }).fit()

    w = {str(k): float(v) for k, v in res.design.donor_weights.items()}
    mv = np.array([w.get(s, 0.0) for s in _REF_WEIGHTS])
    rv = np.array(list(_REF_WEIGHTS.values()))

    years = sorted(d["year"].unique())
    gap = np.asarray(res.inference_detail.gap)
    ite = {yr: float(gap[years.index(yr)]) for yr in _REF_ITE}

    return {
        "n_donors": float(len(_REF_WEIGHTS)),
        "weight_max_abs_dev": float(np.max(np.abs(mv - rv))),
        "weight_mean_abs_dev": float(np.mean(np.abs(mv - rv))),
        "weight_correlation": float(np.corrcoef(mv, rv)[0, 1]),
        "att_mean_post": float(res.att),
        "ite_1990": ite[1990],
        "ite_1995": ite[1995],
        "ite_2000": ite[2000],
    }


# Deterministic (penalty fixed at the Table-2 values; no stochastic CV) => exact
# re-runs. Tolerances absorb the standardisation convention and Table 2's
# 3-decimal rounding while pinning the cross-validation: mlsynth's NSC weights
# track the author's to correlation ~0.99 with no per-weight gap above ~0.025,
# recover the same (signed) donor pool, and reproduce the paper's effect path.
EXPECTED = {
    "n_donors": (38.0, 0.0),
    "weight_max_abs_dev": (0.024, 0.020),     # largest single-donor gap < ~0.044
    "weight_mean_abs_dev": (0.006, 0.008),
    "weight_correlation": (0.989, 0.02),      # fails below ~0.97
    "att_mean_post": (-19.13, 1.5),
    "ite_1990": (-9.05, 2.0),                 # paper -9.5
    "ite_1995": (-22.62, 3.0),                # paper -24.5
    "ite_2000": (-27.01, 3.0),                # paper -28.7
}
