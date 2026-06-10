"""MASC Path-A: the Basque Country / ETA-terrorism study (Kellogg et al. 2020).

Path A (empirical, scenario: the authors' canonical application + predictor set).
Kellogg, Mogstad, Pouliot & Torgovitsky (2020), *"Combining Matching and
Synthetic Control to Trade off Biases from Extrapolation and Interpolation,"*
re-examine the Abadie & Gardeazabal (2003) study of the per-capita GDP cost of
ETA terrorism in the Basque Country (KMPT Section 5: 17 Spanish regions,
1955-1997, with the JASA predictor specification). MASC model-averages a
synthetic control with nearest-neighbour matching via a rolling-origin
cross-validated weight ``phi``.

This reproduces KMPT's result on ``basedata/basque_jasa.csv`` **value for value**,
running MASC exactly as their ``SC_application.R`` does: the MSCMT/``synth``
V-optimisation for the SC step (``sc_backend="mscmt"``, the default) blended with
**covariate** nearest-neighbour matching (``match_on="covariates"``, their
``solve.covmatch``).

  =====================  ===============  =========================
  Quantity               mlsynth MASC     KMPT (Section 5)
  =====================  ===============  =========================
  CV-selected ``phi``    0.00             0.00 (MASC = SC)
  pre-period RMSE        ~$89/capita      ~$94/capita
  Cataluna + Madrid      1.00             1.00 (0.85 + 0.15)
  ATT                    -$585/cap/yr     -$580/cap/yr
  =====================  ===============  =========================

The two predictor-weight optimisers and the two matching bases are exposed as
``sc_backend`` and ``match_on``; the historical defaults (``"bilevel"`` /
``"outcomes"``) instead give -$816 / -$769, off the paper because they are not
the authors' configuration. The fit is deterministic, so the cells are exact
re-runs.

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
            # The KMPT Basque application's exact estimator: MSCMT/synth SC
            # (default) blended with covariate matching (their solve.covmatch).
            "sc_backend": "mscmt", "match_on": "covariates",
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


# Deterministic (MSCMT differential-evolution is seeded). Configured exactly as
# the KMPT Basque application (MSCMT/synth SC + covariate matching), MASC
# reproduces their result value-for-value: the CV selects pure SC (phi = 0), the
# Cataluna + Madrid pool carries all the weight (1.00, matching 0.85 + 0.15), the
# pre-period RMSE is ~$89 (paper ~$94), and the ATT is -$585 (paper -$580).
EXPECTED = {
    "att": (-0.5847, 0.04),                    # KMPT -0.580
    "phi_hat": (0.0, 0.05),                    # pure SC, as in KMPT
    "pre_rmse": (0.0888, 0.012),               # ~$89/capita; KMPT ~$94
    "top_donor_is_cataluna": (1.0, 0.0),
    "cataluna_madrid_mass": (1.0, 0.05),       # KMPT: 1.00 (0.85 + 0.15)
}
