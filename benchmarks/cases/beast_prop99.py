"""BEAST cross-validation: California Proposition 99 vs the authors' R.

Cross-validates mlsynth's ``BEAST`` -- the immunized doubly-robust synthetic
control of Bléhaut, D'Haultfœuille, L'Hour & Tsybakov (2021, arXiv 2005.12225) --
against a live run of the authors' own R code
(``jeremylhour/alternative-synthetic-control-sparsity``:
``CalibrationLasso.R`` / ``OrthogonalityReg.R`` / ``LassoFISTA.R`` /
``ImmunizedATT.R``) on the Abadie, Diamond & Hainmueller (2010) Proposition 99
panel. California is treated from 1989; the balancing design is four economic
predictors (``loginc``, ``p_cig``, ``pct15-24``, ``pc_beer``) plus lagged cigarette
sales (1975/1980/1988), the paper's ``basic'' informative-covariate regime.

The R reference (run on mlsynth's exact design matrix, so any gap is purely the
R OWL-QN vs mlsynth's proximal-gradient calibration) is recorded in
``benchmarks/reference/beast_prop99/``. mlsynth reproduces it: the immunized ATT
mean matches to ~0.02 packs (per-year path within ~0.15) and the standard errors
to ~0.05.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "augmented_cali_long.csv")
_REF = load_reference("beast_prop99")["values"]
_COVS = ["loginc", "p_cig", "pct15-24", "pc_beer"]
_LAGS = [1975, 1980, 1988]
_R_ATT = np.array(_REF["att_path"])
_R_SE = np.array(_REF["se_path"])


def _fit():
    from mlsynth import BEAST

    d = pd.read_csv(os.path.abspath(_DATA))
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BEAST({"df": d, "outcome": "cigsale", "treat": "treated",
                      "unitid": "state", "time": "year", "covariates": _COVS,
                      "outcome_lags": _LAGS, "display_graphs": False}).fit()


def run() -> dict:
    res = _fit()
    det = res.inference.details
    tau = np.asarray(det["tau"], float)
    se = np.asarray(det["se"], float)
    n = min(len(tau), len(_R_ATT))
    return {
        "att_path_max_abs_diff_vs_R": float(np.max(np.abs(tau[:n] - _R_ATT[:n]))),
        "se_path_max_abs_diff_vs_R": float(np.max(np.abs(se[:n] - _R_SE[:n]))),
        "att_mean_vs_R": float(abs(res.att - _REF["att_mean"])),
        "att_2000_vs_R": float(abs(tau[-1] - _REF["att_2000"])),
        # the balancing is a valid synthetic control (weights sum to one)
        "sum_weights": float(det["sum_weights"]),
        # sparse selection in the informative regime
        "n_selected": float(det["n_selected"]),
    }


# Cross-validation (deterministic). mlsynth's BEAST reproduces the authors' R
# immunized ATT (mean -22.44 to ~0.02 packs, -31.51 by 2000; per-year path within
# ~0.15) and the SEs to ~0.05, with a valid balancing (sum of weights = 1) and the sparse single-
# covariate selection of the informative Prop 99 regime.
EXPECTED = {
    "att_path_max_abs_diff_vs_R": (0.0, 0.15),
    "se_path_max_abs_diff_vs_R": (0.0, 0.07),
    "att_mean_vs_R": (0.0, 0.1),
    "att_2000_vs_R": (0.0, 0.1),
    "sum_weights": (1.0, 0.02),
    "n_selected": (1.0, 0.0),
}


def comparison() -> dict:
    """mlsynth BEAST vs the authors' R immunized ATT path, per post year."""
    res = _fit()
    tau = np.asarray(res.inference.details["tau"], float)
    rows = [{"quantity": f"ATT[{yr}]", "mlsynth": round(float(tau[i]), 4),
             "reference": round(float(_R_ATT[i]), 4)}
            for i, yr in enumerate(_REF["years"])]
    rows.append({"quantity": "ATT[mean]", "mlsynth": round(float(res.att), 4),
                 "reference": round(float(_REF["att_mean"]), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "BEAST",
                         "config": {"outcome": "cigsale", "treat": "treated",
                                    "unitid": "state", "time": "year",
                                    "covariates": _COVS, "outcome_lags": _LAGS}},
        "reference": {"impl": "jeremylhour/alternative-synthetic-control-sparsity R "
                              "(CalibrationLasso/OrthogonalityReg/ImmunizedATT)",
                      "version": "live run on mlsynth's design matrix"},
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
