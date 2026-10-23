"""The panels the scale study measures on (#842).

Three kinds, as the issue asks:

- ``bls``: a structured real panel. GeoLift has 40 markets and Walmart 45
  stores, so neither reaches 60. The BLS county unemployment panel of the
  spillover benchmarks has 193 counties of four states (Alabama, Oregon,
  Pennsylvania, Wyoming) over 36 months, with the common cycle, seasonality and
  state structure a market panel has. Counties are taken in one fixed shuffle
  of FIPS order, so each size holds the smaller ones and all four states.
- ``noise``: no structure beyond a common shock (``common_shock`` of
  ``benchmarks/studies/syndes_global_bnb/quality.py``).
- ``flat``: that study's four-factor generator (``factor``), on which the
  bounds prune least.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.studies.syndes_global_bnb.quality import common_shock, factor

ROOT = Path(__file__).resolve().parents[3]
BLS = ROOT / "basedata" / "spsydid_bls_county_subset.csv"
KINDS = ("bls", "noise", "flat")


@lru_cache(maxsize=1)
def _bls_wide():
    d = pd.read_csv(BLS)
    wide = d.pivot_table(index=["year", "month"], columns="FIPS",
                         values="Unemployment_Rate").sort_index()
    wide = wide[sorted(wide.columns)]
    order = np.random.default_rng(0).permutation(wide.shape[1])
    return wide.to_numpy(float)[:, order], np.asarray(wide.columns)[order]


def bls_fips(N: int) -> np.ndarray:
    """The FIPS codes of the first ``N`` counties in the study's order."""
    return _bls_wide()[1][:N]


def bls(N: int) -> np.ndarray:
    """Months by the first ``N`` counties in the study's order."""
    Y, fips = _bls_wide()
    if N > len(fips):
        raise ValueError(f"the BLS panel has {len(fips)} counties; asked for {N}")
    return Y[:, :N].copy()


def panel(kind: str, N: int) -> np.ndarray:
    """Periods by ``N`` units of the named kind."""
    if kind == "bls":
        return bls(N)
    if kind == "noise":
        return common_shock(N, seed=0)
    if kind == "flat":
        return factor(N, seed=0)
    raise ValueError(f"unknown panel kind {kind!r}; the kinds are {KINDS}")
