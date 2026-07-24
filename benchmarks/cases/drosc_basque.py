"""Cross-validation: DROSC vs the authors' R ``DRoSC`` (Koo & Guo 2026) on the
Basque Country study, the deterministic worst-case point estimand across the
robustness-radius sweep.

Path A / cross-validation. mlsynth's :class:`mlsynth.DROSC` reproduces the
authors' ``helpers.R`` ``DRoSC`` (``limSolve::lsei``) value-for-value on the
Basque terrorism panel (T0 = 15, N = 16 donors, T1 = 28): as the robustness
radius ``robustness_lambda`` grows the effect shrinks from the classical-SC
neighbourhood toward zero (tau: -0.742 -> -0.256 -> 0 at lambda = 0 / 0.03 /
0.06), and the lambda = 0 donor weights match by name (Madrid 0.388, Baleares
0.274, Cataluna 0.203, Asturias 0.135). The perturbation union CI is stochastic
(seed-dependent) and is not pinned here; the deterministic estimand is.

Provenance
----------
* Data: ``basedata/basque_jasa.csv`` (the Abadie-Gardeazabal Basque panel; the
  same file ``masc_basque`` uses), treatment at the 16th period.
* Reference: the authors' R ``DRoSC`` (helpers.R, ``limSolve::lsei``) via
  ``benchmarks/R/drosc_basque.R``, captured in
  ``benchmarks/reference/drosc_basque/``. R does not install from CRAN in CI;
  the reference is baked and read here.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "basque_jasa.csv")
_REF = load_reference("drosc_basque")
_LAMBDAS = (0.0, 0.015, 0.03, 0.045, 0.06)


def _basque_df():
    d = pd.read_csv(os.path.abspath(_DATA))
    d = d[d.regionname != "Spain (Espana)"].copy()
    treat_year = sorted(d.year.unique())[15]
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= treat_year)).astype(int)
    return d


def _fit(lam):
    from mlsynth import DROSC
    d = _basque_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DROSC({
            "df": d, "outcome": "gdpcap", "treat": "treat",
            "unitid": "regionname", "time": "year",
            "robustness_lambda": lam, "display_graphs": False,
        }).fit()


def run() -> dict:
    v = _REF["values"]
    res = {lam: _fit(lam) for lam in _LAMBDAS}
    tau = {lam: float(res[lam].effects.att) for lam in _LAMBDAS}
    w0 = res[0.0].weights.donor_weights
    by = lambda frag: next(val for name, val in w0.items() if frag in name)

    tau_dev = max(abs(tau[lam] - v[f"tau_lam{lam:.2f}" if lam in (0.0, 0.03, 0.06)
                                   else f"tau_lam{lam:g}"]) for lam in _LAMBDAS)
    return {
        # --- cross-validation of the worst-case estimand (value-for-value) ---
        "tau_max_abs_dev_vs_R": tau_dev,
        "classical_sc_att_dev_vs_R": abs(
            float(res[0.0].additional_outputs["classical_sc_att"]) - v["tau_SC"]),
        "w_madrid_dev_vs_R": abs(by("Madrid") - v["w_Madrid_lam0"]),
        # --- deterministic descriptors / regression guards ---
        "tau_lam0": tau[0.0],
        "tau_lam06": tau[0.06],
        "shrinks_to_zero": float(abs(tau[0.0]) > abs(tau[0.03]) > abs(tau[0.06])
                                 and abs(tau[0.06]) < 1e-2),
        "weights_sum_to_one": float(sum(w0.values())),
    }


def comparison() -> dict:
    """mlsynth DROSC vs the authors' R DRoSC, quantity by quantity."""
    v = _REF["values"]
    rows = []
    for lam in _LAMBDAS:
        key = f"tau_lam{lam:.2f}" if lam in (0.0, 0.03, 0.06) else f"tau_lam{lam:g}"
        rows.append({"quantity": f"tau (lambda={lam:g})",
                     "mlsynth": round(float(_fit(lam).effects.att), 4),
                     "reference": round(v[key], 4)})
    res0 = _fit(0.0)
    w0 = res0.weights.donor_weights
    rows.append({"quantity": "classical SC ATT",
                 "mlsynth": round(float(res0.additional_outputs["classical_sc_att"]), 4),
                 "reference": round(v["tau_SC"], 4)})
    for frag, key in (("Madrid", "w_Madrid_lam0"), ("Baleares", "w_Baleares_lam0"),
                      ("Cataluna", "w_Cataluna_lam0"), ("Asturias", "w_Asturias_lam0")):
        rows.append({"quantity": f"w[{frag}] (lambda=0)",
                     "mlsynth": round(next(val for n, val in w0.items() if frag in n), 4),
                     "reference": round(v[key], 4)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "DROSC", "config": {"backend": "lsei-band"}},
        "reference": {"impl": "R DRoSC (Koo & Guo 2026 helpers.R, limSolve::lsei)",
                      "version": "arXiv:2511.02632"},
    }


# The worst-case estimand is deterministic and matches R's solve; tolerances
# cover cvxpy-vs-lsei solver noise on the shared optimum.
EXPECTED = {
    "tau_max_abs_dev_vs_R": (0.0, 0.02),
    "classical_sc_att_dev_vs_R": (0.0, 0.01),
    "w_madrid_dev_vs_R": (0.0, 0.03),
    "tau_lam0": (-0.7424, 0.02),
    "tau_lam06": (0.0, 0.02),
    "shrinks_to_zero": (1.0, 0.0),
    "weights_sum_to_one": (1.0, 1e-4),
}
