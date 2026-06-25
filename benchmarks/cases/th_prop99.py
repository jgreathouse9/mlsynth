"""Truncated History left-TH replication: Spoelstra et al. (2025) Table 1.

Path A (the paper's empirical result on the authors' data). The Truncated History
framework of Spoelstra, Stolp, Golsteyn, Cornelisz & van Klaveren (2025,
*Economics Letters* 257, 112701) re-estimates the synthetic-difference-in-
differences effect of California's Proposition 99 on truncated pre-treatment
windows (left-TH: drop the earliest pretreatment years). Their Table 1 reports
the SDID ATE growing modestly as the window shrinks -- a stable profile that
supports the causal reading.

This drives mlsynth's :func:`mlsynth.truncated_history` over ``SDID`` and pins the
full-sample ATE and the 1971/1972/1974 left-truncated ATEs against the paper's
reported Table 1 SDID column, which mlsynth reproduces to the decimal.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from mlsynth import truncated_history, SDID

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata", "P99data.csv")


def _panel() -> pd.DataFrame:
    df = pd.read_csv(os.path.abspath(_DATA)).rename(columns={"cigsale": "y"})
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    return df


def _config() -> dict:
    return {"outcome": "y", "treat": "treat", "unitid": "state", "time": "year"}


def run() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = truncated_history(SDID, {**_config(), "df": _panel()}, mode="left")
    by = {w.label: float(w.att) for w in res.profile}
    return {
        "ate_full": float(res.att_full),
        "ate_1971": by["1971-1988"],
        "ate_1972": by["1972-1988"],
        "ate_1974": by["1974-1988"],
        "stable": 1.0 if res.stable else 0.0,
    }


# Targets: Spoelstra et al. (2025) Table 1, SDID column (full sample 1970-1988 and
# left-truncated starts). Published paper numbers, reproduced by the synthdid path.
EXPECTED = {
    "ate_full": (-15.6, 0.3),
    "ate_1971": (-16.3, 0.3),
    "ate_1972": (-16.7, 0.4),
    "ate_1974": (-17.2, 0.4),
    "stable": (1.0, 0.0),
}
