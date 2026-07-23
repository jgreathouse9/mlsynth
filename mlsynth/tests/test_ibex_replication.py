"""VanillaSC reproduces the ibex synthetic control (Iberian exception, DAP).

Cross-validation of ``VanillaSC(backend="outcome-only")`` against the authors'
own replication code (``github.com/mharoruiz/ibex``) for Haro Ruiz, Schult and
Wunder (2024). Both solve the identical simplex program (non-negative weights
summing to one, matched on all pre-treatment outcome lags), so on the day-ahead
electricity price the donor weights must coincide to solver tolerance. The
durable, dashboard-facing version lives in ``benchmarks/cases/ibex_dap.py``;
this pins the same cross-validation in the fast unit suite, offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "basedata" / "ibex_day_ahead_price.csv"
_REF = json.loads(
    (_ROOT / "benchmarks" / "reference" / "ibex_dap" / "reference.json").read_text()
)
_TREAT_DATE = pd.Timestamp("2022-06-01")


@pytest.fixture(scope="module")
def fit():
    """One multi-treated VanillaSC fit: Spain and Portugal treated in June 2022,
    the 20 complete-series control countries as donors."""
    df = pd.read_csv(_DATA, parse_dates=["date"])
    complete = df.groupby("country")["DAP"].transform(lambda s: ~s.isna().any())
    df = df[complete].copy()
    df["treat"] = ((df.country.isin(["ES", "PT"])) & (df.date >= _TREAT_DATE)).astype(int)
    return VanillaSC({"df": df, "outcome": "DAP", "treat": "treat",
                      "unitid": "country", "time": "date",
                      "backend": "outcome-only", "alpha": 0.10,
                      "display_graphs": False}).fit()


@pytest.mark.parametrize("tu", ["ES", "PT"])
def test_donor_weights_match_ibex(fit, tu):
    """The per-country donor weights reproduce the ibex ``lsei`` SC value-for-value."""
    w = pd.Series(fit.sub_method_results[tu].donor_weights, dtype=float)
    ref = pd.Series(_REF["weights"][tu], dtype=float)
    keys = ref.index.union(w.index)
    l1 = float((w.reindex(keys).fillna(0.0) - ref.reindex(keys).fillna(0.0)).abs().sum())
    assert l1 < 1e-3                                   # simplex weights, solver-tight
    # Slovenia is the dominant donor in both syntheses
    assert w.idxmax() == "SI"
    assert w["SI"] == pytest.approx(_REF["values"][f"{tu.lower()}_si_weight"], abs=1e-3)


@pytest.mark.parametrize("tu,key", [("ES", "es_att_post"), ("PT", "pt_att_post")])
def test_post_att_matches_reference(fit, tu, key):
    """The mean post-treatment gap (day-ahead ATT) reproduces the ibex series and
    shows the paper's large (~40%) cut."""
    f = fit.sub_method_results[tu]
    t = pd.to_datetime(f.time_labels)
    gap = pd.Series(np.asarray(f.gap, dtype=float), index=t)
    att = float(gap[gap.index >= _TREAT_DATE].mean())
    assert att == pytest.approx(_REF["values"][key], abs=0.5)
    assert att < -30.0                                 # a large price reduction in both countries
