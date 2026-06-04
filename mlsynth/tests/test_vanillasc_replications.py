"""Empirical replication regression tests for VanillaSC.

Locks the estimator against the published synthetic controls of the three
canonical studies, each trained on its full pre-treatment period:

* Prop 99 / California (ADH 2010): Colorado, Connecticut, Montana, Nevada, Utah.
* German reunification (ADH 2015): Austria, USA, Japan, Switzerland, Netherlands.
* Basque terrorism (Abadie-Gardeazabal 2003): Cataluna ~0.8, Madrid ~0.2.

Data live under ``basedata/`` at the repo root; tests skip if absent.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_BASEDATA = Path(__file__).resolve().parents[2] / "basedata"


def _topset(res, k=5, thresh=0.02):
    """Names of donors with weight > thresh (up to k), as a set."""
    w = {n: v for n, v in res.weights.donor_weights.items() if v > thresh}
    return set(sorted(w, key=w.get, reverse=True)[:k])


# --------------------------------------------------------------------------- #
# Prop 99 / California (ADH 2010)
# --------------------------------------------------------------------------- #
def _california():
    f = _BASEDATA / "augmented_cali_long.csv"
    if not f.exists():
        pytest.skip("augmented_cali_long.csv not available")
    d = pd.read_csv(f)
    for yr, col in [(1975, "cig_1975"), (1980, "cig_1980"), (1988, "cig_1988")]:
        m = d[d.year == yr].set_index("state").cigsale
        d[col] = d.state.map(m)
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    return d


def test_prop99_outcome_only():
    d = _california()
    res = VanillaSC({"df": d, "outcome": "cigsale", "treat": "treated",
                     "unitid": "state", "time": "year",
                     "backend": "outcome-only", "inference": False,
                     "display_graphs": False}).fit()
    # ADH donors (Utah/Montana/Nevada/Connecticut/Colorado) dominate.
    assert {"Utah", "Montana", "Nevada"} <= _topset(res, k=6)
    assert -25.0 < res.effects.att < -12.0           # ADH ~ -19 packs


def test_prop99_mscmt_covariates_matches_adh():
    d = _california()
    cov = ["p_cig", "pct15-24", "loginc", "pc_beer", "cig_1975", "cig_1980", "cig_1988"]
    win = {"p_cig": (1980, 1988), "pct15-24": (1980, 1988),
           "loginc": (1980, 1988), "pc_beer": (1984, 1988)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": d, "outcome": "cigsale", "treat": "treated",
                         "unitid": "state", "time": "year",
                         "backend": "mscmt", "covariates": cov,
                         "covariate_windows": win, "canonical_v": "min.loss.w",
                         "seed": 1, "mscmt_maxiter": 200, "mscmt_popsize": 20,
                         "inference": False, "display_graphs": False}).fit()
    # ADH Table 2 five-donor set.
    assert {"Utah", "Nevada", "Montana", "Colorado", "Connecticut"} <= _topset(res, k=5)
    assert -25.0 < res.effects.att < -12.0
    # predictor weights are non-identified here (lagged outcomes) -> flagged.
    assert res.weights.summary_stats.get("v_agreement") is not None


# --------------------------------------------------------------------------- #
# German reunification (ADH 2015)
# --------------------------------------------------------------------------- #
def _germany():
    f = _BASEDATA / "repgermany.dta"
    if not f.exists():
        pytest.skip("repgermany.dta not available")
    d = pd.read_stata(f)
    d["treated"] = ((d.country == "West Germany") & (d.year >= 1990)).astype(int)
    return d


def test_german_mscmt_matches_adh():
    d = _germany()
    cov = ["gdp", "trade", "infrate", "industry", "invest80", "schooling"]
    win = {"gdp": (1981, 1990), "trade": (1981, 1990), "infrate": (1981, 1990),
           "industry": (1981, 1990), "invest80": (1980, 1980), "schooling": (1980, 1985)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": d, "outcome": "gdp", "treat": "treated",
                         "unitid": "country", "time": "year",
                         "backend": "mscmt", "covariates": cov,
                         "covariate_windows": win, "seed": 1,
                         "mscmt_maxiter": 200, "mscmt_popsize": 20,
                         "inference": False, "display_graphs": False}).fit()
    # ADH 2015 donors: Austria/USA/Japan/Switzerland/Netherlands. Austria dominant.
    top = _topset(res, k=6)
    assert "Austria" in top and "USA" in top
    assert len({"Switzerland", "Japan", "Netherlands"} & top) >= 2
    assert res.effects.att < 0.0                     # reunification reduced GDP


# --------------------------------------------------------------------------- #
# Basque terrorism (Abadie-Gardeazabal 2003) -- full 1955-1974 pre-period
# --------------------------------------------------------------------------- #
def test_basque_outcome_only_full_preperiod():
    f = _BASEDATA / "basque_data.csv"
    if not f.exists():
        pytest.skip("basque_data.csv not available")
    b = pd.read_csv(f)
    b = b[b.regionno != 1]                            # drop Spain
    # terrorism indicator: first 1 (for the Basque) at 1975 -> train 1955-1974.
    b["treated"] = ((b.regionno == 17) & (b.year >= 1975)).astype(int)
    res = VanillaSC({"df": b, "outcome": "gdpcap", "treat": "treated",
                     "unitid": "regionno", "time": "year",
                     "backend": "outcome-only", "inference": False,
                     "display_graphs": False}).fit()
    w = res.weights.donor_weights
    top = max(w, key=w.get)
    assert int(top) == 10                             # Cataluna dominant
    assert w[top] > 0.6
    assert w.get("14", 0.0) > 0.05                    # Madrid second
    assert res.effects.att < 0.0                      # terrorism reduced GDP
