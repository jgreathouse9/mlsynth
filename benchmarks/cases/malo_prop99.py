"""Malo et al. (2024) bilevel optimum: the California Prop 99 application.

Path A (the authors' Table 1). Malo, Eskelinen, Zhou & Kuosmanen (2024,
*Computing Synthetic Controls Using Bilevel Optimization*, Computational
Economics 64(2):1113-1136) show that the seminal ADH (2010) Prop 99 synthetic
control is a bilevel optimisation problem whose global optimum is a corner
solution: the predictor weights V collapse onto a single predictor (cigarette
sales per capita in 1980) and the donor weights are the outcome-fit simplex.
Their Table 1 reports this "Optimum", which the Synth and MSCMT packages both
miss.

mlsynth's ``backend="malo"`` is the staged corner search of that paper. Through
``VanillaSC.fit()`` it reaches the Table 1 optimum:

    Donor          mlsynth malo   Malo Table 1 "Optimum"
    Utah           0.3977         0.3939
    Montana        0.2270         0.2318
    Nevada         0.2039         0.2049
    Connecticut    0.1093         0.1091
    New Hampshire  0.0470         0.0454
    Colorado       0.0151         0.0148

mlsynth lands on the exact cigsale-1980-matched corner (L_W = 0); the paper
reports the outcome-fit lower bound to which it rounds (L_V = 2.74366,
R^2 = 0.97878). The two agree to two decimals on every donor weight and on the
upper-level objective. The fix that makes the optimum reachable: the bilevel
stages solve the simplex least-squares with the exact active-set QP rather than
the FISTA primitive, which under-converges on the long (1970-1988) pre-period.

Provenance: Malo et al. (2024), Table 1. Data ship as
``basedata/augmented_cali_long.csv`` (ADH 2010 Prop 99 panel + predictors).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "augmented_cali_long.csv")

# Malo et al. (2024) Table 1, "Optimum" column.
_OPT = {"Utah": 0.3939, "Montana": 0.2318, "Nevada": 0.2049,
        "Connecticut": 0.1091, "New Hampshire": 0.0454, "Colorado": 0.0148}
_COVS = ["loginc", "p_cig", "pct15-24", "pc_beer", "cig1975", "cig1980", "cig1988"]
_WINDOWS = {"loginc": (1980, 1988), "p_cig": (1980, 1988), "pct15-24": (1980, 1988),
            "pc_beer": (1984, 1988), "cig1975": (1975, 1975),
            "cig1980": (1980, 1980), "cig1988": (1988, 1988)}


def run() -> dict:
    from mlsynth import VanillaSC

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    for L in (1975, 1980, 1988):
        m = d[d.year == L].set_index("state")["cigsale"]
        d[f"cig{L}"] = d["state"].map(m)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treated",
            "unitid": "state", "time": "year",
            "backend": "malo", "covariates": _COVS, "covariate_windows": _WINDOWS,
            "seed": 0, "display_graphs": False,
        }).fit()

    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    return {
        "weight_max_abs_dev": float(max(abs(w.get(k, 0.0) - v)
                                        for k, v in _OPT.items())),
        "weight_utah": w.get("Utah", 0.0),
        "weight_montana": w.get("Montana", 0.0),
        "weight_connecticut": w.get("Connecticut", 0.0),
        "L_V": float(res.fit_diagnostics.rmse_pre ** 2),
        "n_positive_donors": float(sum(1 for v in w.values() if v > 1e-3)),
    }


# Deterministic (exact QP, fixed seed). The malo corner reproduces Malo Table 1
# to two decimals on every donor weight (max deviation ~0.005) and on the
# upper-level objective L_V (2.744 vs 2.74366), over the same six-state pool.
EXPECTED = {
    "weight_max_abs_dev": (0.005, 0.008),      # vs Table 1; all donors within ~0.005
    "weight_utah": (0.3939, 0.01),             # Malo 0.3939
    "weight_montana": (0.2318, 0.01),          # Malo 0.2318
    "weight_connecticut": (0.1091, 0.01),      # Malo 0.1091
    "L_V": (2.74366, 0.01),                    # Malo Table 1 L_V
    "n_positive_donors": (6.0, 1.0),           # Malo's six-state corner
}
