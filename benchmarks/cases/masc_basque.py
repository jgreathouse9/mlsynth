"""MASC Path-A: the Basque Country / ETA-terrorism study (Kellogg et al. 2020).

Path A (empirical, scenario: the authors' canonical application + predictor set).
Kellogg, Mogstad, Pouliot & Torgovitsky (2020), *"Combining Matching and
Synthetic Control to Trade off Biases from Extrapolation and Interpolation,"*
re-examine the Abadie & Gardeazabal (2003) study of the per-capita GDP cost of
ETA terrorism in the Basque Country (KMPT Section 5: 17 Spanish regions,
1955-1997, with the JASA predictor specification). MASC model-averages a
synthetic control with nearest-neighbour matching via a rolling-origin
cross-validated weight ``phi``.

This reproduces KMPT's qualitative finding on ``basedata/basque_jasa.csv``: a
large negative GDP gap built from a **Cataluna-dominated** donor pool with a
tight pre-period fit. The point estimate differs from the paper's because the SC
predictor-weight (``V``) optimisation is non-unique -- KMPT use the ``synth``
package's quasi-Newton search, mlsynth the Malo et al. bilevel solver -- so the
two converge to different ``V`` (and hence ``W`` and CV-selected ``m, phi``).
The agreement is therefore on the robust quantities: the dominant donors, the
pre-period RMSE, and the sign/order of magnitude of the effect.

  =====================  ===============  =========================
  Quantity               mlsynth MASC     KMPT (Section 5)
  =====================  ===============  =========================
  pre-period RMSE        ~$89/capita      ~$94/capita
  top donor              Cataluna         Cataluna (0.85)
  Cataluna + Madrid      ~0.74            1.00 (0.85 + 0.15)
  ATT                    ~-$816/cap/yr    ~-$580/cap/yr
  =====================  ===============  =========================

The fit is deterministic (no RNG in the CV or the V-solver), so the cells below
are exact re-runs.

Provenance: Kellogg, Mogstad, Pouliot & Torgovitsky (2020), Section 5;
Abadie & Gardeazabal (2003) for the original study and predictor windows.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "basque_jasa.csv")

_COVS = ["school.illit", "school.prim", "school.med", "school.high", "invest",
         "sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
         "sec.services.venta", "sec.services.nonventa", "popdens"]
# Abadie & Gardeazabal (2003) Table 1 averaging windows.
_WINDOWS = {
    "sec.agriculture": (1961, 1969), "sec.energy": (1961, 1969),
    "sec.industry": (1961, 1969), "sec.construction": (1961, 1969),
    "sec.services.venta": (1961, 1969), "sec.services.nonventa": (1961, 1969),
    "popdens": (1969, 1969), "invest": (1964, 1969),
    "school.illit": (1964, 1969), "school.prim": (1964, 1969),
    "school.med": (1964, 1969), "school.high": (1964, 1969),
    "gdpcap": (1960, 1969),
}


def run() -> dict:
    from mlsynth import MASC

    df = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MASC({
            "df": df, "outcome": "gdpcap", "treat": "terrorism",
            "unitid": "regionname", "time": "year",
            "m_grid": list(range(1, 11)), "min_preperiods": 5,
            "covariates": _COVS, "covariate_windows": _WINDOWS,
            "display_graphs": False,
        }).fit()

    w = {str(k): float(v) for k, v in res.donor_weights.items()}
    top_donor = max(w, key=w.get)
    cm = w.get("Cataluna", 0.0) + w.get("Madrid (Comunidad De)", 0.0)

    return {
        "att": float(res.att),
        "phi_hat": float(res.phi_hat),
        "pre_rmse": float(res.fit.pre_rmse),
        "top_donor_is_cataluna": 1.0 if top_donor == "Cataluna" else 0.0,
        "cataluna_madrid_mass": float(cm),
    }


# Deterministic (no RNG). Tolerances pin the mlsynth result as a regression guard
# while the prose documents the KMPT comparison: the pre-period RMSE (~$89 vs the
# paper's ~$94) and the Cataluna-dominated pool agree closely; the ATT (-$816)
# shares the paper's sign and ~$600-800 magnitude but differs in level because the
# V-optimiser is non-unique (see the estimator docs). MASC blends SC with a small
# amount of matching (phi ~ 0.3) rather than the paper's pure SC (phi = 0).
EXPECTED = {
    "att": (-0.816, 0.12),
    "phi_hat": (0.308, 0.15),                  # small-to-moderate matching
    "pre_rmse": (0.0892, 0.015),               # ~$89/capita; KMPT ~$94
    "top_donor_is_cataluna": (1.0, 0.0),
    "cataluna_madrid_mass": (0.744, 0.15),     # KMPT: 1.00
}
