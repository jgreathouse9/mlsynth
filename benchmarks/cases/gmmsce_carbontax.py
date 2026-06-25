"""GMM-SCE cross-validation: Fry (2024) GMM-SCE.R on Andersson (2019)'s carbon tax.

Cross-validation against the genuine author implementation. Fry's GMM Synthetic
Control Estimator -- ``ORTHSC(method="gmm_sce")`` -- is run on Andersson's
Swedish carbon-tax panel with the same control pool (14 donors) and instruments
(the 7 carbon/fuel-tax countries he excluded) as the ``orthsc_carbontax`` case,
and is held to a live captured run of Fry's own ``GMM-SCE.R`` ``GMMSC()``
(``benchmarks/reference/gmmsce_carbontax/``), read via :func:`reference_value`.

Because the donors fit Sweden's pre-treatment path almost exactly (the
over-identification J-statistic is ~1e-5), the GMM program is close to having a
flat optimum: with fourteen control weights and only eight moment conditions the
individual weights are not point-identified (the L-infinity situation), so this
case pins the quantities that *are* identified -- the J-statistic -- and verifies
that mlsynth attains a GMM objective no worse than the reference's interior-point
``LowRankQP`` solve. The discriminating weight-level cross-check against the R is
the over-identified, imperfect-fit example in ``mlsynth/tests/test_gmm_sce.py``
(``test_solver_matches_R_reference_and_is_optimal``), where the optimum is unique
and the weights agree to ``LowRankQP``'s tolerance.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference, reference_value
from mlsynth import ORTHSC

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "carbontax_fullsample_data.dta.txt")
_CONTROLS = ["Australia", "Belgium", "Canada", "Denmark", "France", "Greece",
             "Iceland", "Japan", "New Zealand", "Poland", "Portugal", "Spain",
             "Switzerland", "United States"]
_INSTRS = ["Finland", "Germany", "Ireland", "Italy", "Netherlands", "Norway",
           "United Kingdom"]


def _panel() -> pd.DataFrame:
    df = pd.read_stata(os.path.abspath(_DATA)).rename(
        columns={"CO2_transport_capita": "Y"})
    df["treat"] = ((df["country"] == "Sweden") & (df["year"] >= 1990)).astype(int)
    return df


def _config() -> dict:
    return {"outcome": "Y", "treat": "treat", "unitid": "country", "time": "year",
            "method": "gmm_sce", "controls": _CONTROLS, "instruments": _INSTRS}


def _fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ORTHSC({**_config(), "df": _panel(), "display_graphs": False}).fit()


def run() -> dict:
    res = _fit()
    py_j = float(res.additional_outputs["jstatistic"])
    ref_j = reference_value("gmmsce_carbontax", "jstatistic")
    return {
        "jstatistic": py_j,
        # mlsynth's GMM objective is no worse than the reference's (1.0 == True).
        "objective_no_worse": 1.0 if py_j <= ref_j + 1e-9 else 0.0,
    }


def comparison() -> dict:
    """mlsynth GMM-SCE vs Fry's GMM-SCE.R, quantity by quantity (weights shown
    for the record; only the identified J-statistic is pinned)."""
    res = _fit()
    w = res.weights.donor_weights
    ref = load_reference("gmmsce_carbontax")
    ref_w = ref["weights"]
    rows = [{"quantity": "J-statistic",
             "mlsynth": round(float(res.additional_outputs["jstatistic"]), 8),
             "reference": round(float(ref["values"]["jstatistic"]), 8)}]
    for c in _CONTROLS:
        rows.append({"quantity": f"weight[{c}]",
                     "mlsynth": round(float(w[c]), 6),
                     "reference": round(float(ref_w[c]), 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "ORTHSC", "config": _config()},
        "reference": {"impl": "Fry GMM-SCE.R GMMSC() (R, live run, captured)",
                      "version": "github.com/JosephPatrickFry @ 3b38684"},
    }


# Targets: a live captured run of Fry's own GMM-SCE.R on Andersson's carbon-tax
# data (benchmarks/reference/gmmsce_carbontax/), read via reference_value so the
# pin and the captured run are the same object. The J-statistic is the identified
# over-identification objective; mlsynth attains it no worse than the reference.
EXPECTED = {
    "jstatistic": (reference_value("gmmsce_carbontax", "jstatistic"), 5e-6),
    "objective_no_worse": (1.0, 0.0),
}
