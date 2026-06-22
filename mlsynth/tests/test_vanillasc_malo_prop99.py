"""Malo et al. (2024) bilevel optimum: the California Prop 99 application.

Malo, Eskelinen, Zhou & Kuosmanen (2024, Computational Economics 64(2)),
Table 1, report the global bilevel optimum of the seminal ADH (2010) Prop 99
synthetic control -- a corner solution whose predictor weights ``V`` collapse
onto a single predictor (cigarette sales per capita in 1980) and whose donor
weights are the outcome-fit simplex. Standard packages (Synth, MSCMT) miss it.

mlsynth's ``backend="malo"`` is the staged corner search of that paper. This
test locks it to Table 1's Optimum column through ``VanillaSC.fit()``:

    Donor          Malo Table 1 "Optimum"
    Utah           0.3939
    Montana        0.2318
    Nevada         0.2049
    Connecticut    0.1091
    New Hampshire  0.0454
    Colorado       0.0148

The fix that makes this reachable: the bilevel stages solve the simplex
least-squares with the exact active-set QP rather than the FISTA primitive,
which under-converges on Prop 99's long (1970-1988) pre-period.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "augmented_cali_long.csv"

# Malo et al. (2024) Table 1, "Optimum" column.
_MALO = {"Utah": 0.3939, "Montana": 0.2318, "Nevada": 0.2049,
         "Connecticut": 0.1091, "New Hampshire": 0.0454, "Colorado": 0.0148}


@pytest.fixture(scope="module")
def prop99():
    if not _DATA.exists():
        pytest.skip("augmented_cali_long.csv not available")
    d = pd.read_csv(_DATA)
    d["treated"] = ((d.state == "California") & (d.year >= 1989)).astype(int)
    for L in (1975, 1980, 1988):
        m = d[d.year == L].set_index("state")["cigsale"]
        d[f"cig{L}"] = d["state"].map(m)
    return d


def _malo_fit(d):
    covs = ["loginc", "p_cig", "pct15-24", "pc_beer", "cig1975", "cig1980", "cig1988"]
    windows = {"loginc": (1980, 1988), "p_cig": (1980, 1988), "pct15-24": (1980, 1988),
               "pc_beer": (1984, 1988), "cig1975": (1975, 1975),
               "cig1980": (1980, 1980), "cig1988": (1988, 1988)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC({
            "df": d, "outcome": "cigsale", "treat": "treated",
            "unitid": "state", "time": "year",
            "backend": "malo", "covariates": covs, "covariate_windows": windows,
            "seed": 0, "display_graphs": False,
        }).fit()


def test_malo_reaches_table1_optimum(prop99):
    res = _malo_fit(prop99)
    w = {str(k): float(v) for k, v in res.weights.donor_weights.items()}
    for donor, target in _MALO.items():
        assert w.get(donor, 0.0) == pytest.approx(target, abs=0.01), (
            f"{donor}: got {w.get(donor, 0.0):.4f}, Malo Table 1 {target}")


def test_malo_matches_outcome_fit_objective(prop99):
    """Table 1's optimum is the outcome-fit corner: L_V (pre-period outcome MSE)
    is 2.74366 and R^2 = 0.97878. The malo backend must reach that objective,
    not a worse corner."""
    res = _malo_fit(prop99)
    rmse_pre = float(res.fit_diagnostics.rmse_pre)
    assert rmse_pre ** 2 == pytest.approx(2.74366, abs=0.05)
