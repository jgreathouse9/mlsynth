"""The panel every script in this study measures on.

GeoLift's 40 markets as one MAREX cluster, the 90 pre-periods of
``basedata/geolift_market_data.csv`` as the fit window, no covariates and no
standardisation, four treated markets. ``geolift`` returns what MAREX's design
hands :func:`~mlsynth.utils.marex_helpers.exact.search_cluster` on that
configuration: the markets as the columns of ``B`` and their mean as the target
``A``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "basedata" / "geolift_market_data.csv"
RESULTS = Path(__file__).resolve().parent / "results"

T0 = 90
M = 4
BETAS = (1.0, 100.0)


def geolift(T0: int = T0):
    """``B`` (T0, 40), one column per market, and ``A`` (T0,), their mean."""
    d = pd.read_csv(DATA)
    Y = d.pivot(index="date", columns="location", values="Y").sort_index()
    X = Y.to_numpy(float)[:T0].T
    return X.T.copy(), X.mean(axis=0)


def optimum(beta: float) -> float:
    """The optimal objective at ``beta``, as ``search.py`` recorded it."""
    path = RESULTS / "search.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run search.py first.")
    return float(json.loads(path.read_text())["runs"][str(beta)]["shipped"]["value"])
