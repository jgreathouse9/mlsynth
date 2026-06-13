"""Cross-validation benchmark: SNN vs ``deshen24/syntheticNN`` (Prop 99).

Cross-validation against the reference implementation. ``SNN`` is mlsynth's port
of Synthetic Nearest Neighbors (Agarwal, Dahleh, Shah & Shen 2021, *Causal
Matrix Completion*), whose canonical implementation is
`deshen24/syntheticNN <https://github.com/deshen24/syntheticNN>`_. On block
missingness -- the synthetic-control setting, where the treated unit's
post-period is the only missing block -- the reference's NetworkX maximum-
biclique anchor search and mlsynth's dependency-free greedy search both return
the full *control x pre-period* block. With the same Donoho-Gavish (2014) rank
and principal-component regression, the two implementations therefore impute the
**same** counterfactual.

Provenance
----------
* Data: ``basedata/smoking_data.csv`` -- the Abadie, Diamond & Hainmueller (2010)
  Prop 99 panel (39 states, 1970-2000; California treated from 1989). Outcome
  ``cigsale`` (per-capita cigarette packs).
* ``REF_CF`` is California's imputed untreated counterfactual from a live run of
  ``deshen24/syntheticNN`` (``SyntheticNearestNeighbors(n_neighbors=1)``,
  universal/Donoho-Gavish rank), with California's 1989-2000 entries set to
  ``NaN``. mlsynth reproduces it to machine precision (``< 1e-6`` here).

This case also guards the Donoho-Gavish ``omega`` coefficients: an earlier
mlsynth bug had ``1.43`` and ``1.82`` swapped, which mis-selected the rank and
shifted the ATT by ~1 pack/capita. Matched to the reference formula, the gap
vanishes.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# California untreated counterfactual (cigsale) from deshen24/syntheticNN.
REF_CF = {
    1989: 89.236371, 1990: 82.673847, 1991: 79.485312, 1992: 78.230661,
    1993: 78.142375, 1994: 77.473927, 1995: 78.417803, 1996: 77.388595,
    1997: 77.785353, 1998: 77.896060, 1999: 77.752842, 2000: 70.925823,
}
REF_ATT = -18.434081


def run() -> dict:
    from mlsynth import SNN

    df = pd.read_csv(_BASE / "smoking_data.csv")
    obs = df.pivot(index="state", columns="year", values="cigsale").loc["California"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SNN({"df": df, "outcome": "cigsale", "treat": "Proposition 99",
                   "unitid": "state", "time": "year", "display_graphs": False}).fit()

    years = sorted(REF_CF)
    cf_mls = np.array([float(obs[y]) - res.att_by_period[y] for y in years])
    cf_ref = np.array([REF_CF[y] for y in years])

    return {
        "snn_att": float(res.att),
        "snn_counterfactual_max_abs_diff": float(np.max(np.abs(cf_mls - cf_ref))),
        "snn_gap_2000": float(res.att_by_period[2000]),
        "n_states": int(df.state.nunique()),
        "n_pre_periods": int((df[df.state == "California"].year < 1989).sum()),
    }


# mlsynth reproduces the reference to ~1e-6 (residual is the printed-digit
# rounding of REF_CF); the structural agreement is exact.
EXPECTED = {
    "snn_att": (REF_ATT, 1e-3),
    "snn_counterfactual_max_abs_diff": (0.0, 1e-5),
    "snn_gap_2000": (-29.3258, 1e-2),                # reference gap by 2000
    "n_states": (39, 0),
    "n_pre_periods": (19, 0),
}
