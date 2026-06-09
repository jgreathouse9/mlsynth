"""Path A benchmark: TSSC Brooklyn-showroom empirical (Li & Shankar 2024).

Reproduces the published Two-Step Synthetic Control result on the authors' own
``Data.csv`` panel (the Brooklyn-showroom illustration from the Management
Science replication package MS-MKG-20-01498): 110 weeks, one treated unit + 10
donor markets, treatment at week 76. mlsynth's Step-1 restriction tests pick the
MSC(b) variant the paper flags, and its recovered ATT / pre-fit match the
published numbers to the third decimal.

Provenance
----------
* Data: ``basedata/Data.csv`` -- the authors' public Brooklyn-showroom panel
  (vendored from the MNSC replication package's ``Data.csv``).
* Headline: Li & Shankar (2024) report the showroom MSC(b) effect at
  ATT = 1131.97 with pre-RMSE 434.43; Step 1 selects MSC(b).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parents[2] / "basedata"
T1 = 76


def run() -> dict:
    from mlsynth import TSSC

    raw = pd.read_csv(_BASE / "Data.csv")
    T = len(raw)
    rows = [{"unit": "Brooklyn", "time": t, "y": float(raw.iloc[t, 0]),
             "treat": int(t >= T1)} for t in range(T)]
    for j in range(1, raw.shape[1]):
        rows += [{"unit": f"Donor{j}", "time": t, "y": float(raw.iloc[t, j]),
                  "treat": 0} for t in range(T)]
    df = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TSSC({"df": df, "outcome": "y", "treat": "treat",
                    "unitid": "unit", "time": "time", "seed": 0,
                    "display_graphs": False}).fit()

    mscb = res.variants["MSCb"]
    return {
        # 1.0 iff Step 1's restriction tests select MSC(b) (the paper's path).
        "recommends_mscb": float(res.selection.recommended == "MSCb"),
        "mscb_att": float(mscb.att),
        "mscb_pre_rmse": float(mscb.rmse_pre),
    }


# Deterministic (seeded). The recommended variant must be MSC(b) and its ATT /
# pre-RMSE must match the paper's published 1131.97 / 434.43 to display precision.
EXPECTED = {
    "recommends_mscb": (1.0, 0.0),
    "mscb_att": (1131.97, 1.0),
    "mscb_pre_rmse": (434.43, 1.0),
}
