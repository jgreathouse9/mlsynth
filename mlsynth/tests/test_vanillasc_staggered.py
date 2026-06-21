"""Staggered-adoption VanillaSC, validated against the official ``scpi`` package.

Reference: Cattaneo, Feng, Palomba and Titiunik's multiple-treated-unit
illustration (``scpi_illustration-multi.py``) on the Germany reunification panel
with two staggered treated units -- West Germany (treated 1991) and Italy (treated
1992) -- and the 15 never-treated countries as donors. Run outcome-only with a
simplex constraint, ``scpi``'s ``scest`` reports the per-unit and unit-time
average treatment effects pinned below. mlsynth's staggered VanillaSC must
reproduce them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "scpi_germany.csv")

# scpi reference (outcome-only, simplex), from scest on the two-treated panel.
_SCPI = {
    "West Germany": -1.8476,
    "Italy": -1.1211,
    "overall_unit_time": -1.4989,
}


@pytest.fixture
def germany_staggered():
    df = pd.read_csv(os.path.abspath(_DATA))
    df["status"] = 0
    df.loc[(df["country"] == "West Germany") & (df["year"] >= 1991), "status"] = 1
    df.loc[(df["country"] == "Italy") & (df["year"] >= 1992), "status"] = 1
    return df[["country", "year", "gdp", "status"]]


def _fit(df):
    return VanillaSC({
        "df": df, "outcome": "gdp", "treat": "status",
        "unitid": "country", "time": "year", "display_graphs": False,
    }).fit()


def test_overall_att_matches_scpi(germany_staggered):
    res = _fit(germany_staggered)
    assert res.effects.att == pytest.approx(_SCPI["overall_unit_time"], abs=0.05)


def test_per_unit_att_matches_scpi(germany_staggered):
    res = _fit(germany_staggered)
    by_unit = {c.treated_unit_name: float(c.att)
               for c in res.sub_method_results.values()}
    assert by_unit["West Germany"] == pytest.approx(_SCPI["West Germany"], abs=0.05)
    assert by_unit["Italy"] == pytest.approx(_SCPI["Italy"], abs=0.05)


def test_donor_pool_is_never_treated(germany_staggered):
    """Each treated unit's synthetic control draws on the 15 never-treated
    countries, never on the other treated unit."""
    res = _fit(germany_staggered)
    for cohort in res.sub_method_results.values():
        donors = set(map(str, cohort.donor_names))
        assert "West Germany" not in donors
        assert "Italy" not in donors
        assert len(donors) == 15


def _synthetic_staggered(adopt, N=16, T=20, r=3, tau=3.0, seed=0):
    """Factor-model panel: ``adopt`` maps unit index -> adoption period; the rest
    are never-treated donors."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, r))
    Lam = rng.standard_normal((N, r))
    eps = rng.standard_normal((N, T)) * 0.3
    Y = Lam @ F.T + eps
    rows = []
    for i in range(N):
        Ti = adopt.get(i)
        for t in range(T):
            treated = int(Ti is not None and t >= Ti)
            rows.append({"id": f"u{i}", "time": t,
                         "y": float(Y[i, t] + (tau if treated else 0.0)),
                         "D": treated})
    return pd.DataFrame(rows)


def _fit_synth(df):
    return VanillaSC({"df": df, "outcome": "y", "treat": "D",
                      "unitid": "id", "time": "time",
                      "display_graphs": False}).fit()


def test_two_treated_same_cohort_smoke():
    """Two units adopting at the same time form one cohort; both are fit."""
    res = _fit_synth(_synthetic_staggered({0: 12, 1: 12}))
    assert len(res.sub_method_results) == 2
    assert set(res.sub_method_results) == {"u0", "u1"}
    # planted effect recovered to the ballpark
    assert res.effects.att == pytest.approx(3.0, abs=0.6)


def test_overall_att_is_postweighted_mean_of_units():
    res = _fit_synth(_synthetic_staggered({0: 10, 1: 13, 2: 16}))
    fits = list(res.sub_method_results.values())
    num = sum(f.att * f.post_periods for f in fits)
    den = sum(f.post_periods for f in fits)
    assert res.effects.att == pytest.approx(num / den, abs=1e-9)


def test_per_unit_weights_on_simplex():
    res = _fit_synth(_synthetic_staggered({0: 12, 1: 14}))
    for fit in res.sub_method_results.values():
        w = np.array(list(fit.donor_weights.values()), dtype=float)
        assert (w >= -1e-8).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
