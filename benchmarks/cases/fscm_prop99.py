"""FSCM Path-A: Forward-Selected Synthetic Control on Proposition 99.

Path A (empirical, scenario: the canonical ADH panel). Cerulli (2024)'s
forward-selected synthetic control greedily grows the donor set, choosing its
size by cross-validated RMSPE. On California's Proposition 99 tobacco panel it
selects a small donor pool whose pre-period fit beats the full-pool synthetic
control. This makes that result durable (it previously lived only in the
catalogue / test suite).

Outcome-only matching on the shipped ``basedata/smoking_data.csv``; the fit is
deterministic, so the cells below are exact re-runs.

Provenance: Cerulli (2024), forward-selected SC; the ADH Proposition 99 panel.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "smoking_data.csv")


def run() -> dict:
    from mlsynth import FSCM

    d = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = FSCM({
            "df": d, "outcome": "cigsale", "treat": "Proposition 99",
            "unitid": "state", "time": "year", "display_graphs": False,
        }).fit()

    diag = res.diagnostics
    import numpy as np
    return {
        "att": float(res.att),
        "pre_r_squared": float(diag["pre_r_squared"]),
        "cv_rmspe_at_optimum": float(diag["cv_rmspe_at_optimum"]),
        "cv_rmspe_full_pool": float(diag["cv_rmspe_full_pool"]),
        "n_donors_selected": float((np.asarray(res.weights_vector) > 1e-6).sum()),
    }


# Deterministic (greedy selection + CV, no RNG) => exact re-runs. FSCM selects a
# three-donor pool whose cross-validated RMSPE (1.61) roughly halves the
# full-pool baseline (2.92), with a strong pre-period fit (R^2 ~ 0.97) and the
# canonical Proposition 99 effect (ATT -20.15).
EXPECTED = {
    "att": (-20.15, 0.6),
    "pre_r_squared": (0.970, 0.02),
    "cv_rmspe_at_optimum": (1.605, 0.15),
    "cv_rmspe_full_pool": (2.916, 0.25),
    "n_donors_selected": (3.0, 1.0),
}
