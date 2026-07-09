"""Error-in-variables prediction intervals for VanillaSC (Hirshberg 2021).

``inference="eiv"`` forms normal/t prediction intervals for the synthetic-control
effect from the error-in-variables theory of Hirshberg (2021, arXiv 2104.08931):
the counterfactual prediction error is asymptotically normal with a variance
consistently estimated by the pre-treatment residual scale
``sqrt(sum u_t^2 / (T0 - df))``. Intervals use a ``t(T0 - df)`` reference; the
participation ratio ``p_eff = 1/||theta||^2`` is reported as a diagnostic.

Checked on the Abadie-Gardeazabal Basque terrorism panel (treated 1975) and on a
seeded low-rank simulation for coverage; the detailed coverage Monte Carlo lives
in the ``eiv_coverage_mc`` benchmark.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC
from mlsynth.utils.vanillasc_helpers.eiv import eiv_intervals

_BASQUE = pathlib.Path(__file__).resolve().parents[2] / "basedata" / "basque_data.csv"


def _toy(seed=0, n=8, T=24, T0=18):
    """Small panel: treated = convex combo of two donors + noise, no real effect."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 2))                       # two latent factors
    load = rng.normal(size=(n, 2))
    Y = f @ load.T + 0.1 * rng.normal(size=(T, n))    # low-rank + noise
    y = 0.6 * Y[:, 0] + 0.4 * Y[:, 1] + 0.1 * rng.normal(size=T)  # treated ~ donors 0,1
    rows = []
    for i in range(n):
        for t in range(T):
            rows.append({"unit": f"c{i}", "time": t, "y": Y[t, i], "d": 0})
    for t in range(T):
        rows.append({"unit": "T", "time": t, "y": y[t], "d": int(t >= T0)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# unit: the interval function
# ----------------------------------------------------------------------
def test_eiv_intervals_shapes_and_diagnostics():
    rng = np.random.default_rng(1)
    T, J, T0 = 30, 5, 20
    Y0 = rng.normal(size=(T, J))
    W = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    y = Y0 @ W + 0.1 * rng.normal(size=T)
    ev = eiv_intervals(y, Y0, T0, W, alpha=0.05)
    assert ev.tau.shape == (T - T0,)
    assert ev.lower.shape == ev.upper.shape == (T - T0,)
    assert np.all(ev.lower <= ev.upper)
    # participation ratio: 1/||theta||^2, here between 1 and #active donors
    assert 1.0 <= ev.metadata["p_eff"] <= 3.0 + 1e-9
    assert ev.metadata["dof"] == T0 - (3 - 1)          # 3 active weights
    assert ev.metadata["sigma_tau"] > 0


def test_eiv_effect_interval_brackets_point():
    ev = eiv_intervals(*_prep(_toy(0)))
    # per-period point effect sits inside its own interval
    assert np.all((ev.lower <= ev.tau) & (ev.tau <= ev.upper))
    assert ev.att_lower <= ev.att <= ev.att_upper


def _prep(df):
    piv = df.pivot(index="time", columns="unit", values="y")
    donors = [c for c in piv.columns if c != "T"]
    y = piv["T"].to_numpy(float)
    Y0 = piv[donors].to_numpy(float)
    T0 = int((df[df.unit == "T"].sort_values("time")["d"].to_numpy() == 0).sum())
    # simplex fit on the pre-period
    import cvxpy as cp
    th = cp.Variable(len(donors))
    cp.Problem(cp.Minimize(cp.sum_squares(y[:T0] - Y0[:T0] @ th)),
               [cp.sum(th) == 1, th >= 0]).solve(solver=cp.CLARABEL)
    return y, Y0, T0, np.clip(np.asarray(th.value).ravel(), 0, None)


# ----------------------------------------------------------------------
# integration: through VanillaSC.fit()
# ----------------------------------------------------------------------
def test_eiv_runs_through_vanillasc():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": _toy(2), "outcome": "y", "treat": "d",
                         "unitid": "unit", "time": "time",
                         "display_graphs": False, "inference": "eiv"}).fit()
    assert "Hirshberg" in res.inference.method
    det = res.inference.details
    assert set(det) >= {"tau", "pi_lower", "pi_upper", "sigma_tau", "p_eff",
                        "counterfactual_lower", "counterfactual_upper"}
    assert res.inference.ci_lower <= res.inference.ci_upper


def test_eiv_does_not_change_weights():
    base = {"df": _toy(3), "outcome": "y", "treat": "d", "unitid": "unit",
            "time": "time", "display_graphs": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_none = VanillaSC({**base, "inference": False}).fit().weights.donor_weights
        w_eiv = VanillaSC({**base, "inference": "eiv"}).fit().weights.donor_weights
    for k in w_none:
        assert w_eiv[k] == pytest.approx(w_none[k], abs=1e-9)


# ----------------------------------------------------------------------
# empirical: Basque terrorism (Abadie-Gardeazabal)
# ----------------------------------------------------------------------
@pytest.mark.skipif(not _BASQUE.exists(), reason="Basque data absent")
def test_eiv_basque_significant_negative():
    d = pd.read_csv(_BASQUE)[["regionname", "year", "gdpcap"]].dropna()
    d["treat"] = ((d.regionname == "Basque Country (Pais Vasco)") &
                  (d.year >= 1975)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VanillaSC({"df": d, "outcome": "gdpcap", "treat": "treat",
                         "unitid": "regionname", "time": "year",
                         "display_graphs": False, "inference": "eiv"}).fit()
    det = res.inference.details
    # ETA terrorism cut Basque GDP/capita ~-0.6 to -0.7 (Abadie-Gardeazabal 2003)
    assert det["att"] == pytest.approx(-0.69, abs=0.1)
    # the ATT interval excludes zero
    assert det["att_upper"] < 0.0
    # effect is significant by the mid-1980s (per-period CI excludes 0)
    years = list(det["periods"])
    lo = np.asarray(det["pi_lower"], float)
    hi = np.asarray(det["pi_upper"], float)
    i85 = years.index(1985)
    assert lo[i85] > 0 or hi[i85] < 0
