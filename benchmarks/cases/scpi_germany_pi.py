"""SCPI prediction intervals (single unit) cross-validation: German reunification.

Cross-validates mlsynth's ``VanillaSC(inference="scpi")`` -- an MIT re-derivation
of the Cattaneo, Feng & Titiunik (2021, JASA) synthetic-control prediction
intervals -- against a live ``scpi_pkg`` run on the classic single-treated-unit
West Germany reunification panel (``basedata/scpi_germany.csv``, GDP per capita,
1960-2003, West Germany treated from 1991, 16 donor countries). This is the setup
of the Mendez ``python_scpi`` tutorial.

Both the point estimate and the prediction bands are checked:

* the simplex synthetic control is identical -- mlsynth and ``scpi`` pick the same
  donors and weights (Austria, USA, Italy, Netherlands, Switzerland, France) to
  four decimals;
* the synthetic-prediction band (``scpi``'s ``CI_all_gaussian``) reproduces to
  Monte-Carlo error in BOTH specifications:

  - ``scpi_cointegrated=False`` (levels) vs ``scdata(cointegrated_data=False)``;
  - ``scpi_cointegrated=True``  (first differences, the tutorial's setting) vs
    ``scdata(cointegrated_data=True)`` -- the in-sample ``E[u]`` and out-of-sample
    ``E[e]`` models fit on differenced donors, with the pre->post bridge
    ``dP[0] = P[0] - B[T0-1]``.

The ``scpi`` reference (``e_method="gaussian"``, seed 8894, sims 2000) is recorded
in ``benchmarks/reference/scpi_germany_pi/`` rather than computed live: ``scpi`` is
GPL and mlsynth is MIT, so the benchmark carries no run-time dependency on it.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "scpi_germany.csv")
_REF = load_reference("scpi_germany_pi")["values"]
_SIMS = 2000
_SEED = 8894

_COINT_W = np.array(_REF["coint_hi"]) - np.array(_REF["coint_lo"])
_LEVELS_W = np.array(_REF["levels_hi"]) - np.array(_REF["levels_lo"])


def _panel() -> pd.DataFrame:
    d = pd.read_csv(os.path.abspath(_DATA))[["country", "year", "gdp"]].dropna()
    d["status"] = ((d.country == "West Germany") & (d.year >= 1991)).astype(int)
    return d


def _fit(cointegrated: bool):
    from mlsynth import VanillaSC

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({"df": _panel(), "outcome": "gdp", "treat": "status",
                          "unitid": "country", "time": "year",
                          "display_graphs": False, "inference": "scpi",
                          "scpi_sims": _SIMS, "seed": _SEED,
                          "scpi_e_method": "gaussian",
                          "scpi_cointegrated": cointegrated}).fit()


def _band(res):
    det = res.inference.details
    return (np.asarray(det["counterfactual_lower"], float),
            np.asarray(det["counterfactual_upper"], float))


def run() -> dict:
    coint = _fit(True)
    levels = _fit(False)

    # point estimate agreement with scpi (weights)
    wd = coint.weights.donor_weights
    w_ref = _REF["weights"]
    w_max_abs = max(abs(wd.get(k, 0.0) - v) for k, v in w_ref.items())

    lo_c, hi_c = _band(coint)
    lo_l, hi_l = _band(levels)
    wc, wl = hi_c - lo_c, hi_l - lo_l

    coint_lo_ref = np.array(_REF["coint_lo"]); coint_hi_ref = np.array(_REF["coint_hi"])
    levels_lo_ref = np.array(_REF["levels_lo"]); levels_hi_ref = np.array(_REF["levels_hi"])

    return {
        "weights_max_abs_diff_vs_scpi": float(w_max_abs),
        # cointegrated band reproduces scpi's cointegrated_data=True band
        "coint_width_max_abs_diff": float(np.max(np.abs(wc - _COINT_W))),
        "coint_endpoint_max_abs_diff": float(max(
            np.max(np.abs(lo_c - coint_lo_ref)), np.max(np.abs(hi_c - coint_hi_ref)))),
        # levels default reproduces scpi's cointegrated_data=False band
        "levels_width_max_abs_diff": float(np.max(np.abs(wl - _LEVELS_W))),
        "levels_endpoint_max_abs_diff": float(max(
            np.max(np.abs(lo_l - levels_lo_ref)), np.max(np.abs(hi_l - levels_hi_ref)))),
        # cointegration is a real, distinct spec (early-year band much narrower)
        "coint_matches_coint_better_than_levels": float(
            np.mean(np.abs(wc - _COINT_W)) < np.mean(np.abs(wc - _LEVELS_W))),
        "cointegration_narrows_1992_band": float(wc[1] < wl[1] - 0.3),
    }


# Deterministic (seeded MCMC-free simulation, seed 8894, sims 2000). mlsynth's
# SCPI prediction intervals reproduce scpi_pkg's CI_all_gaussian to Monte-Carlo
# error in both the levels (cointegrated_data=False) and first-difference
# (cointegrated_data=True) specifications, and the simplex weights match to 4 dp.
EXPECTED = {
    "weights_max_abs_diff_vs_scpi": (0.0, 1e-3),
    "coint_width_max_abs_diff": (0.11, 0.30),       # ~0.11 at sims=2000
    "coint_endpoint_max_abs_diff": (0.10, 0.30),
    "levels_width_max_abs_diff": (0.15, 0.30),
    "levels_endpoint_max_abs_diff": (0.15, 0.30),
    "coint_matches_coint_better_than_levels": (1.0, 0.0),
    "cointegration_narrows_1992_band": (1.0, 0.0),
}


def comparison() -> dict:
    """mlsynth SCPI band vs the scpi_pkg reference, per post year (cointegrated)."""
    coint = _fit(True)
    lo_c, hi_c = _band(coint)
    years = _REF["years"]
    rows = []
    for i, yr in enumerate(years):
        rows.append({"quantity": f"coint_width[{yr}]",
                     "mlsynth": round(float(hi_c[i] - lo_c[i]), 4),
                     "reference": round(float(_COINT_W[i]), 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC",
                         "config": {"inference": "scpi", "scpi_cointegrated": True,
                                    "scpi_e_method": "gaussian", "outcome": "gdp",
                                    "treat": "status", "unitid": "country",
                                    "time": "year"}},
        "reference": {"impl": "scpi_pkg scdata(cointegrated_data=True)+scpi CI_all_gaussian",
                      "version": "scpi_pkg (PyPI), seed 8894, sims 2000"},
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(run(), indent=2))
