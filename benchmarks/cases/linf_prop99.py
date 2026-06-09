"""Path A: the L-infinity SC's dense weighting on California Proposition 99.

Wang, Xing & Ye (2025), Section 6.1, apply the L-infinity SC to the canonical
Abadie-Diamond-Hainmueller tobacco panel (California treated, Proposition 99
from 1989, 38 donor states, 1970-2000). Their headline contrast (Figure 4) is
**qualitative and graphical**: classic SC concentrates on ~6 donor states while
the L-infinity / L1+L-infinity methods "allocate weights more evenly" -- a dense
weighting -- and SC "appears to overestimate the effect" relative to the denser
estimators. The paper reports no numeric ATT, RMSPE, or weight table for Prop
99, so this case pins the reproducible qualitative facts rather than cell values.

What we check
-------------
* ``sc_nnz`` -- classic SC is sparse (~6 donors).
* ``linf_nnz`` -- the faithful Wang-Xing-Ye LINF is dense (most donors active).
* ``linf_n_negative`` -- LINF uses negative weights (it is not on the simplex).
* ``linf_max_weight`` < ``sc_max_weight`` -- weight is spread, not concentrated.
* ``sc_att`` < ``linf_att`` < 0 -- both negative, and SC's effect is the more
  negative (the paper's "SC overestimates the effect").

Provenance / scenario
---------------------
* Paper only for numbers (scenario 1) -> qualitative Path A.
* Data: ``basedata/smoking_data.csv`` (the ADH Prop 99 panel shipped with
  mlsynth); no external dependency, always runs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "smoking_data.csv"


def _fit():
    import pandas as pd

    from mlsynth import RESCM

    df = pd.read_csv(_DATA)
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RESCM({
            "df": df, "outcome": "cigsale", "treat": "treat", "unitid": "state",
            "time": "year", "methods": ["SC", "LINF"], "display_graphs": False,
        }).fit()


def run() -> dict:
    from benchmarks.compare import BenchmarkSkipped

    if not _DATA.exists():  # pragma: no cover - data ships with the repo
        raise BenchmarkSkipped(f"missing Prop 99 data at {_DATA}")

    res = _fit()
    sc, linf = res.fits["SC"], res.fits["LINF"]
    sc_w = np.array(list(sc.donor_weights.values()), dtype=float)
    linf_w = np.array(list(linf.donor_weights.values()), dtype=float)

    return {
        "sc_nnz": float(np.sum(np.abs(sc_w) > 1e-3)),
        "linf_nnz": float(np.sum(np.abs(linf_w) > 1e-3)),
        "linf_n_negative": float(np.sum(linf_w < -1e-3)),
        "spread": float(np.abs(linf_w).max() < np.abs(sc_w).max()),
        "sc_more_negative": float(sc.att < linf.att < 0),
    }


# Qualitative reproduction of the paper's Figure 4/5 narrative (no numeric
# targets exist). Tolerances bracket the dense-vs-sparse contrast robustly.
EXPECTED = {
    "sc_nnz": (6.0, 4.0),          # classic SC: a handful of donors
    "linf_nnz": (38.0, 10.0),      # LINF: dense across the donor pool
    "linf_n_negative": (12.0, 11.0),  # LINF leaves the simplex (negative weights)
    "spread": (1.0, 0.0),          # LINF spreads weight (lower peak than SC)
    "sc_more_negative": (1.0, 0.0),   # SC overestimates the effect vs LINF
}
