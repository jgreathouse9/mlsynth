"""Clean-room staggered SCPI engine (CFPT 2025, Section 4).

The engine reproduces ``scpi``'s multiple-treated-unit prediction intervals.
The headline invariant, proven against ``scpi`` at machine precision, is that
its published time-aggregated (TSUA) in-sample band carries a ``1 / iota**2``
scaling (one ``1 / iota`` too many): with the same draws, the scpi-compat band
is exactly ``1 / iota`` of the statistically correct default. These tests pin
that relationship and the engine's basic contract on a small synthetic panel,
keeping the (cvxpy) cost low.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.vanillasc_helpers import staggered_engine as E
from mlsynth.utils.vanillasc_helpers.staggered_engine import staggered_pi_bands

_GERMANY = os.path.join(os.path.dirname(__file__), "..", "..",
                        "basedata", "scpi_germany.csv")


def _panel(seed: int = 0, n_donor: int = 5, T: int = 14, r: int = 2):
    """Two staggered treated units + never-treated donors, factor structure."""
    rng = np.random.default_rng(seed)
    units = [f"d{j}" for j in range(n_donor)] + ["A", "B"]
    F = rng.standard_normal((T, r))
    rows = []
    adopt = {"A": T - 5, "B": T - 4}      # staggered adoption
    for u in units:
        lam = rng.standard_normal(r)
        base = F @ lam + 5.0 + rng.standard_normal(T) * 0.2
        for t in range(T):
            treated = u in adopt and t >= adopt[u]
            y = base[t] + (2.0 if treated else 0.0)
            rows.append({"country": u, "year": 2000 + t, "gdp": y,
                         "status": int(treated)})
    return pd.DataFrame(rows)


_COMMON = dict(outcome="gdp", unitid="country", time="year", treat="status",
               effect="time", sims=120, seed=7)


def test_pi_bands_smoke():
    df = _panel()
    out = staggered_pi_bands(df, scpi_compat=False, **_COMMON)
    n = len(out["index"])
    assert n >= 1
    for key in ("point", "insample_lb", "insample_ub", "lb", "ub"):
        assert np.asarray(out[key]).shape == (n,)
        assert np.all(np.isfinite(out[key]))
    # bands bracket the synthetic point and full band contains the in-sample band
    assert np.all(out["insample_lb"] <= out["point"] + 1e-9)
    assert np.all(out["insample_ub"] >= out["point"] - 1e-9)
    assert np.all(out["lb"] <= out["insample_lb"] + 1e-9)
    assert np.all(out["ub"] >= out["insample_ub"] - 1e-9)


def test_scpi_compat_scales_insample_by_one_over_iota():
    """With identical draws (same seed), the scpi-compat in-sample band is
    exactly 1/iota of the statistically correct default. iota = 2 here."""
    df = _panel()
    d = staggered_pi_bands(df, scpi_compat=False, **_COMMON)
    c = staggered_pi_bands(df, scpi_compat=True, **_COMMON)
    wd = d["insample_ub"] - d["insample_lb"]
    wc = c["insample_ub"] - c["insample_lb"]
    assert np.allclose(wc, wd / 2.0, rtol=1e-6, atol=1e-9)


def test_default_band_is_wider_than_compat():
    """The correct default (1/iota) in-sample band is wider than scpi's."""
    df = _panel(seed=1)
    d = staggered_pi_bands(df, scpi_compat=False, **_COMMON)
    c = staggered_pi_bands(df, scpi_compat=True, **_COMMON)
    assert np.all((d["ub"] - d["lb"]) >= (c["ub"] - c["lb"]) - 1e-9)


def _germany_covariate_unit_time():
    """scpi's multi-illustration unit-time spec: shared multi-features, constant,
    cointegration, constant+trend adjustment -- the case that exercises the
    near-null conic directions per cell."""
    df = pd.read_csv(os.path.abspath(_GERMANY))
    df["status"] = 0
    df.loc[(df.country == "West Germany") & (df.year >= 1991), "status"] = 1
    df.loc[(df.country == "Italy") & (df.year >= 1992), "status"] = 1
    md = E.scdata_multi(df, "gdp", {"features": ["gdp", "trade"]},
                        {"cov_adj": ["constant", "trend"]}, True, True, "unit-time")
    return E.scest(md)


def test_covariate_per_cell_conic_stays_bounded():
    """Per-cell (unit-time) bands with covariates + cointegration must stay
    finite and bounded -- the ECOS construction handles the near-null Q
    directions where a cvxpy reformulation produced an unbounded-ray blowup."""
    est = _germany_covariate_unit_time()
    out = E.scpi(est, sims=120, seed=8894, tsua_double_divide=True)
    lo, hi = out["sc_l_1"], out["sc_r_1"]
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.max(hi - lo) < 50.0           # the blowup bug gave widths ~8e3
    assert np.all(hi >= lo)


def test_covariate_point_estimates_match_scpi_reference():
    """The synthetic predictions for the covariate unit-time spec are the
    deterministic scest quantities; pinned to the scpi-reproduced values."""
    est = _germany_covariate_unit_time()
    yf = np.asarray(est["Y_post_fit"], float).ravel()
    assert len(yf) == 25
    np.testing.assert_allclose(yf[:3], [19.6354, 20.2760, 21.1597], atol=1e-3)
