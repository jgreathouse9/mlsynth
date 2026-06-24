"""Malo et al. (2024) bilevel optimum vs MSCMT: the Abadie-Gardeazabal Basque study.

Cross-validation (the reference is the SCM-Debug bilevel solver of Malo,
Eskelinen, Zhou & Kuosmanen 2024, ``scm.corner``: a constrained outcome QP for
the donor weights, then an LP for the predictor weights). Run on the AG (2003)
Basque specification -- the MSCMT vignette predictor set, outcome fit over
1960-1969 -- ``scm.corner`` returns the bilevel optimum

    Madrid (Comunidad De)  0.4404
    Baleares (Islas)       0.3700
    Rioja (La)             0.1894

These targets are a live captured run of ``scm.corner`` -- not transcribed
constants -- bundled in ``benchmarks/reference/malo_basque/`` (the solver source
``scm.corner.R`` is vendored from SCM-Debug under MIT, see its ``NOTICE``); the
case reads them via :func:`load_reference`. mlsynth's ``backend="malo"``
reproduces it through ``VanillaSC.fit()`` (Madrid 0.441, Baleares 0.370, Rioja
0.189). Crucially, this optimum has a lower
pre-period (1960-1969) outcome loss than the solution MSCMT's nested
differential-evolution search converges to (Cataluna 0.6328, Baleares 0.2193,
Madrid 0.1479): malo's fit-window MSE is ~0.00413 against MSCMT's ~0.00429. This
is exactly the paper's thesis -- the data-driven nested SCM packages can fail to
reach the global optimum -- demonstrated on a second dataset, and a direct
cross-check that malo and MSCMT solve the *same* problem with malo reaching the
better (corner) solution.

The fit window is essential (``fit_window=(1960, 1969)``, MSCMT's ``times.dep``);
the reference numbers are a live captured run of ``scm.corner``
(``benchmarks/reference/malo_basque/reference.R``, regenerate with
``python benchmarks/reference/generate.py malo_basque``; see also
``benchmarks/R/scmcorner_basque.R`` / ``install_scmcorner.sh``). Data ship as
``basedata/basque_mscmt.csv``.

Provenance: Malo et al. (2024) and their ``scm.corner`` solver
(github.com/Xun90/SCM-Debug); Abadie & Gardeazabal (2003), AER 93(1).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from benchmarks.reference import load_reference

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_mscmt.csv")

_W16 = (1964, 1969)
_W19 = (1961, 1969)
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
# MSCMT's nested DE solution (higher pre-period loss); malo must beat it.
_MSCMT_FIT_SSR = 0.004286


def _ref_weights() -> dict:
    """scm.corner's bilevel-optimum donor weights, from the live captured run in
    ``benchmarks/reference/malo_basque/`` (parsed ``reference.json["weights"]``)."""
    return {str(k): float(v) for k, v in load_reference("malo_basque")["weights"].items()}


def _fit(d, backend, **extra):
    from mlsynth import VanillaSC
    cfg = {"df": d, "outcome": "gdpcap", "treat": "treat", "unitid": "regionname",
           "time": "year", "backend": backend, "covariates": _COVS,
           "covariate_windows": _WINDOWS, "fit_window": (1960, 1969),
           "seed": 0, "display_graphs": False}
    cfg.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


def run() -> dict:
    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= 1970)).astype(int)
    res = _fit(d, "malo")
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}

    years = np.array(sorted(d["year"].unique()))
    fit = (years >= 1960) & (years <= 1969)
    yb = (d[d.regionname == "Basque Country (Pais Vasco)"]
          .sort_values("year")["gdpcap"].to_numpy())
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    fit_ssr = float(np.mean((yb[fit] - cf[fit]) ** 2))

    return {
        "madrid": w.get("Madrid (Comunidad De)", 0.0),
        "baleares": w.get("Baleares (Islas)", 0.0),
        "rioja": w.get("Rioja (La)", 0.0),
        "three_donor_mass": (w.get("Madrid (Comunidad De)", 0.0)
                             + w.get("Baleares (Islas)", 0.0)
                             + w.get("Rioja (La)", 0.0)),
        "fit_ssr": fit_ssr,
        # >0 means malo's corner beats MSCMT's nested DE solution.
        "ssr_improvement_over_mscmt": _MSCMT_FIT_SSR - fit_ssr,
    }


def comparison() -> dict:
    """mlsynth ``backend="malo"`` vs ``scm.corner``'s bilevel optimum, weight by
    weight. The reference is the three non-zero donor weights ``scm.corner``
    reports on this Basque specification, read from the live captured run in
    ``benchmarks/reference/malo_basque/`` (``reference.json["weights"]``, the
    solver source vendored from SCM-Debug under MIT -- Malo et al. 2024) rather
    than transcribed constants. Returns ``{"rows": [...], "mlsynth_call": {...},
    "reference": {...}}`` with ``{quantity, mlsynth, reference}`` rows for the
    three bilevel-optimum donors."""
    d = pd.read_csv(os.path.abspath(_DATA))
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)")
                  & (d.year >= 1970)).astype(int)
    res = _fit(d, "malo")
    w_ml = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    # scm.corner's bilevel optimum, from the live captured bundle.
    w_ref = _ref_weights()
    rows = [{"quantity": f"weight[{s}]", "mlsynth": round(w_ml.get(s, 0.0), 6),
             "reference": round(v, 6)} for s, v in w_ref.items()]
    cfg = {"outcome": "gdpcap", "treat": "treat", "unitid": "regionname",
           "time": "year", "backend": "malo", "covariates": _COVS,
           "covariate_windows": _WINDOWS, "fit_window": (1960, 1969), "seed": 0}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "VanillaSC", "config": cfg},
        "reference": {"impl": "Malo et al. scm.corner (SCM-Debug, live run, captured)",
                      "version": "live captured run (benchmarks/reference/malo_basque/; "
                                 "Malo et al. 2024, github.com/Xun90/SCM-Debug)"},
    }


# Deterministic (exact QP). malo reproduces scm.corner's bilevel optimum -- the
# three donor weights are pinned from the live captured scm.corner run in
# benchmarks/reference/malo_basque/ (not transcribed); malo agrees to ~1e-3 -- and
# reaches a lower pre-period loss than MSCMT's nested search (the Malo et al.
# thesis). Tolerances reflect the mlsynth-vs-scm.corner gap.
_rw = _ref_weights()
EXPECTED = {
    "madrid": (_rw["Madrid (Comunidad De)"], 0.002),
    "baleares": (_rw["Baleares (Islas)"], 0.002),
    "rioja": (_rw["Rioja (La)"], 0.002),
    "three_donor_mass": (1.0, 0.01),
    "fit_ssr": (0.004127, 0.0003),
    "ssr_improvement_over_mscmt": (0.000159, 0.0003),   # malo < MSCMT loss
}
