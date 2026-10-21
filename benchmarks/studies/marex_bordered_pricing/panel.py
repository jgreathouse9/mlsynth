"""The two panels this study prices candidates on.

``geolift`` is GeoLift's 40 markets as one MAREX cluster, the 90 pre-periods of
``basedata/geolift_market_data.csv`` as the fit window, no covariates and no
standardisation: what MAREX's design hands
:func:`~mlsynth.utils.marex_helpers.exact.search_cluster`, the markets as the
columns of ``B`` and their mean as the target ``A``. ``synthetic`` is an
80-market panel of the same length, two random-walk factors with loadings and
noise that vary by market, for the larger ``n`` the issue asks about.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "basedata" / "geolift_market_data.csv"
RESULTS = Path(__file__).resolve().parent / "results"

T0 = 90


def geolift(T0: int = T0):
    """``B`` (T0, 40), one column per market, and ``A`` (T0,), their mean."""
    d = pd.read_csv(DATA)
    Y = d.pivot(index="date", columns="location", values="Y").sort_index()
    X = Y.to_numpy(float)[:T0].T
    return X.T.copy(), X.mean(axis=0)


def synthetic(n: int = 80, T: int = T0, seed: int = 7):
    """``B`` (T, n) and its column mean ``A``."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    load = rng.uniform(0.4, 1.6, size=(2, n))
    noise = rng.normal(size=(T, n)) * rng.uniform(0.05, 4.0, n)
    Y = f @ load + noise + 20.0
    return Y, Y.mean(axis=1)
