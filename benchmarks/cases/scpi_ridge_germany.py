"""scpi ridge-constraint cross-validation via CLUSTERSC: German reunification.

scpi (Cattaneo, Feng, Palomba & Titiunik 2025, JSS) Table 3 assigns the ridge
weight constraint to Amjad, Kim, Shah & Shen (2018) Robust Synthetic Control,
which CLUSTERSC's PCR / RSC path implements. CLUSTERSC now routes that fit's
prediction intervals through VanillaSC's generalized ``scpi_intervals`` under the
ridge constraint (``compute_scpi_pi=True, scpi_constraint="ridge"``).

This cross-validates the ridge constraint machinery value-for-value against a
live ``scpi_pkg`` run on the West Germany reunification panel
(``basedata/scpi_germany.csv``, GDP per capita, 1960-2003, 16 donors). Run at
full PCR rank so the denoised donor design equals the raw donors, the ridge
budget ``Q``, penalty ``lambda`` and effective degrees of freedom ``df`` that
CLUSTERSC surfaces via ``.fit()`` (``res.cluster_inference.scpi``) reproduce
``scest`` / ``df_EST`` exactly -- these depend on the panel only (the OLS
shrinkage rule-of-thumb and the SVD effective-dof), not on the fitted weights.

The ``scpi`` reference is recorded in ``benchmarks/reference/scpi_ridge_germany/``
(``scpi`` is GPL, mlsynth MIT). scpi's own ECOS band simulation is
numpy-2.x-incompatible, so the (weight-independent) constraint machinery is
cross-validated here; the band itself rides the shared, simplex-validated engine
(see ``scpi_germany_pi``).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "scpi_germany.csv")
_REF = load_reference("scpi_ridge_germany")["values"]
_SIMS = 600
_SEED = 8894


def _panel() -> pd.DataFrame:
    d = pd.read_csv(os.path.abspath(_DATA))[["country", "year", "gdp"]].dropna()
    d["status"] = ((d.country == "West Germany") & (d.year >= 1991)).astype(int)
    return d


def _fit():
    from mlsynth import CLUSTERSC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CLUSTERSC({
            "df": _panel(), "outcome": "gdp", "treat": "status",
            "unitid": "country", "time": "year", "display_graphs": False,
            "method": "pcr", "pcr_objective": "OLS", "clustering": False,
            # full rank -> HSVT is the identity, denoised donors == raw donors,
            # so the ridge Q/lambda/df match scest's raw-donor values.
            "rank": int(_REF["J"]), "rank_method": "fixed",
            "standardize_for_rank": False, "compute_shen_ci": False,
            "compute_scpi_pi": True, "scpi_constraint": "ridge",
            "scpi_sims": _SIMS, "random_state": _SEED,
        }).fit()


def run() -> dict:
    res = _fit()
    sc = res.cluster_inference.scpi
    lo, hi = res.att_ci
    return {
        # value-for-value: the ridge constraint machinery matches scpi_pkg
        "ridge_Q_abs_diff": float(abs(sc.Q - _REF["Q"])),
        "ridge_lambda_abs_diff": float(abs(sc.lambda_ - _REF["lambda"])),
        "ridge_df_abs_diff": float(abs(sc.df - _REF["df"])),
        "constraint_is_ridge": float(sc.constraint == "ridge"),
        # the routed inference is coherent (finite, ordered ATT interval)
        "att_ci_ordered": float(np.isfinite(lo) and np.isfinite(hi) and hi >= lo),
        "simultaneous_ge_pointwise": float(
            np.mean(np.asarray(sc.cf_upper_simul) - np.asarray(sc.cf_lower_simul))
            >= np.mean(np.asarray(sc.cf_upper) - np.asarray(sc.cf_lower)) - 1e-6),
    }


# Deterministic constraint machinery (Q, lambda, df are panel-only, so they match
# scpi_pkg exactly, independent of the simulation seed). The routed prediction
# interval is coherent and its simultaneous band is never tighter than pointwise.
EXPECTED = {
    "ridge_Q_abs_diff": (0.0, 1e-6),
    "ridge_lambda_abs_diff": (0.0, 1e-6),
    "ridge_df_abs_diff": (0.0, 1e-5),
    "constraint_is_ridge": (1.0, 0.0),
    "att_ci_ordered": (1.0, 0.0),
    "simultaneous_ge_pointwise": (1.0, 0.0),
}


def comparison() -> dict:
    """CLUSTERSC scpi ridge constraint machinery vs the scpi_pkg reference."""
    res = _fit()
    sc = res.cluster_inference.scpi
    rows = [
        {"quantity": "ridge_Q", "mlsynth": round(float(sc.Q), 8),
         "reference": round(float(_REF["Q"]), 8)},
        {"quantity": "ridge_lambda", "mlsynth": round(float(sc.lambda_), 8),
         "reference": round(float(_REF["lambda"]), 8)},
        {"quantity": "ridge_df", "mlsynth": round(float(sc.df), 8),
         "reference": round(float(_REF["df"]), 8)},
    ]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "CLUSTERSC",
                         "config": {"method": "pcr", "pcr_objective": "OLS",
                                    "compute_scpi_pi": True,
                                    "scpi_constraint": "ridge",
                                    "rank": int(_REF["J"]), "rank_method": "fixed",
                                    "outcome": "gdp", "treat": "status",
                                    "unitid": "country", "time": "year"}},
        "reference": {"impl": "scpi_pkg scest(w_constr={'name':'ridge'}) + df_EST",
                      "version": "scpi_pkg 4.0.0 (PyPI)"},
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
