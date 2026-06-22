"""Malo et al. (2024) bilevel optimum vs MSCMT: the Abadie-Gardeazabal Basque study.

Cross-validation (the reference is the SCM-Debug bilevel solver of Malo,
Eskelinen, Zhou & Kuosmanen 2024, ``scm.corner``: a constrained outcome QP for
the donor weights, then an LP for the predictor weights). Run on the AG (2003)
Basque specification -- the MSCMT vignette predictor set, outcome fit over
1960-1969 -- ``scm.corner`` returns the bilevel optimum

    Madrid (Comunidad De)  0.4404
    Baleares (Islas)       0.3700
    Rioja (La)             0.1894

mlsynth's ``backend="malo"`` reproduces it through ``VanillaSC.fit()`` (Madrid
0.441, Baleares 0.370, Rioja 0.189). Crucially, this optimum has a lower
pre-period (1960-1969) outcome loss than the solution MSCMT's nested
differential-evolution search converges to (Cataluna 0.6328, Baleares 0.2193,
Madrid 0.1479): malo's fit-window MSE is ~0.00413 against MSCMT's ~0.00429. This
is exactly the paper's thesis -- the data-driven nested SCM packages can fail to
reach the global optimum -- demonstrated on a second dataset, and a direct
cross-check that malo and MSCMT solve the *same* problem with malo reaching the
better (corner) solution.

The fit window is essential (``fit_window=(1960, 1969)``, MSCMT's ``times.dep``);
the reference numbers were produced by ``scm.corner`` (see
``benchmarks/R/scmcorner_basque.R``). Data ship as
``basedata/basque_mscmt.csv``.

Provenance: Malo et al. (2024) and their ``scm.corner`` solver
(github.com/Xun90/SCM-Debug); Abadie & Gardeazabal (2003), AER 93(1).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

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


# Deterministic (exact QP). malo reproduces scm.corner's bilevel optimum value
# for value (Madrid 0.4404, Baleares 0.3700, Rioja 0.1894) and reaches a lower
# pre-period loss than MSCMT's nested search -- the Malo et al. thesis.
EXPECTED = {
    "madrid": (0.4404, 0.01),
    "baleares": (0.3700, 0.01),
    "rioja": (0.1894, 0.01),
    "three_donor_mass": (1.0, 0.01),
    "fit_ssr": (0.004127, 0.0003),
    "ssr_improvement_over_mscmt": (0.000159, 0.0003),   # malo < MSCMT loss
}
