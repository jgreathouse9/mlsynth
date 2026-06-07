"""Path A benchmark: Li (2024) Forward DiD, Hong Kong GDP empirical result.

Li's headline application uses a confidential retailer panel, but the author
released a **public** companion replication on the Hsiao, Ching & Wan (2012)
Hong Kong GDP panel (the political/economic integration of Hong Kong with
mainland China). This case reproduces that released result cell by cell:
mlsynth's :class:`~mlsynth.FDID` on ``basedata/HongKong.csv`` against the
ATT / %ATT / pre-period R^2 / selected-control-count printed by the author's
own MATLAB and R code (see ``ForwardDID_Readme.txt``).

Pure Python (no R); deterministic (forward selection has no randomness), so
tolerances only absorb the estimator's 3-4 dp display rounding.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlsynth import FDID

# basedata/HongKong.csv lives at the repo root.
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "HongKong.csv"


def run() -> dict:
    df = pd.read_csv(_DATA)
    res = FDID(
        {
            "df": df,
            "outcome": "GDP",
            "treat": "Integration",
            "unitid": "Country",
            "time": "Time",
            "display_graphs": False,
            "verbose": False,
        }
    ).fit()
    f, d = res.fdid, res.did
    return {
        "fdid_att": float(f.att),
        "fdid_att_pct": float(f.att_percent),
        "fdid_r2_pre": float(f.r_squared),
        "fdid_n_controls": float(len(f.selected_names)),
        "did_att": float(d.att),
        "did_att_pct": float(d.att_percent),
        "did_r2_pre": float(d.r_squared),
    }


# Author's released MATLAB/R output (ForwardDID_Readme.txt); tol absorbs the
# estimator's display rounding only -- the result is otherwise deterministic.
EXPECTED = {
    "fdid_att": (0.025405, 5e-4),
    "fdid_att_pct": (53.843, 0.1),
    "fdid_r2_pre": (0.84278, 2e-3),
    "fdid_n_controls": (9.0, 0.0),
    "did_att": (0.031721, 5e-4),
    "did_att_pct": (77.62, 0.1),
    "did_r2_pre": (0.50465, 2e-3),
}
