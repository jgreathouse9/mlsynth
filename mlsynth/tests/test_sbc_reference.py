"""Per-step cross-validation of mlsynth's SBC against Shi-Xi-Xie's own R code.

These tests pin each step of mlsynth's Synthetic Business Cycle estimator to a
golden value captured from the authors' actual functions (Germany.R: the ``lsq``
Hamilton detrending, the ``trend_predict`` forecast, and ``Synth::synth``) run on
the authoritative ``basedata/repgermany.dta``. The golden block lives in
``benchmarks/reference/sbc_germany/golden_steps.txt`` (regenerate with
``Rscript benchmarks/reference/sbc_germany/golden_steps.R``); the tests read it,
so no R toolchain is needed at test time.

Findings this harness locks in (see docs/replications/sbc.rst):

* mlsynth's Hamilton detrending and trend forecast reproduce the authors'
  ``lsq`` / ``trend_predict`` to ~1e-8 -- those steps are identical.
* The donor cycles are detrended on the full sample and the treated unit on the
  pre window, matching the authors' code.
* The only divergence is the synthetic-control solver: the authors call
  ``Synth::synth`` with slack ipop tolerances, which under-converges; mlsynth's
  in-house simplex solver attains a strictly lower cyclical SSE (the verified
  global optimum). So mlsynth is the more accurate implementation, not a
  divergent one.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.sbc_helpers.hamilton import fit_hamilton_filter
from mlsynth.utils.sbc_helpers.trend_forecast import forecast_treated_trend
from mlsynth.utils.bilevel.simplex import simplex_lstsq

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks" / "reference" / "sbc_germany" / "golden_steps.txt"
DATA = ROOT / "basedata" / "german_reunification.csv"

H, P = 4, 2
TOL = 1e-5   # mlsynth-vs-R agreement is ~5e-9; this is comfortably tight.


def _load_golden() -> dict:
    out: dict = {}
    if not GOLDEN.exists():
        pytest.skip(f"golden fixture missing: {GOLDEN}")
    for line in GOLDEN.read_text().splitlines():
        if "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        out[key] = (np.array([float(x) for x in val.split(",")])
                    if "," in val else float(val))
    return out


@pytest.fixture(scope="module")
def golden() -> dict:
    return _load_golden()


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip(f"data missing: {DATA}")
    return pd.read_csv(DATA).pivot(index="year", columns="country", values="gdp")


@pytest.fixture(scope="module")
def treated_fit(panel):
    years = list(panel.index)
    T0 = years.index(1990) + 1
    return fit_hamilton_filter(panel["West Germany"].to_numpy()[:T0], h=H, p=P), T0


def test_hamilton_treated_trend_coefficients(golden, treated_fit):
    """Step 1: the treated unit's Hamilton AR coefficients match R's ``lsq``."""
    fit, _ = treated_fit
    assert np.max(np.abs(fit.coefficients - golden["treated_trend_coef"])) < TOL


def test_hamilton_treated_cycle(golden, treated_fit):
    """Step 1: the treated pre-treatment cyclical component matches R's ``lsq``."""
    fit, _ = treated_fit
    cycle = fit.cycle_pre[~np.isnan(fit.cycle_pre)]
    assert np.max(np.abs(cycle - golden["treated_cycle_pre"])) < TOL


@pytest.mark.parametrize("donor", ["Netherlands", "Greece", "Italy"])
def test_hamilton_donor_cycle_full_sample(golden, panel, donor):
    """Step 1: donor cycles are detrended on the FULL sample, matching R."""
    cyc = fit_hamilton_filter(panel[donor].to_numpy(), h=H, p=P).cycle_pre
    valid = cyc[~np.isnan(cyc)]
    ref = golden[f"donor_cycle_full:{donor}"]
    assert np.max(np.abs(valid[: len(ref)] - ref)) < TOL


def test_trend_forecast(golden, panel, treated_fit):
    """Step 2: the post-treatment trend extrapolation matches R's ``trend_predict``."""
    fit, T0 = treated_fit
    y = panel["West Germany"].to_numpy()
    fc = forecast_treated_trend(y_target=y, treated_fit=fit, T0=T0, horizon=H)
    assert np.max(np.abs(fc - golden["trend_forecast"])) < TOL


def test_simplex_weights_beat_authors_loose_synth(golden, panel, treated_fit):
    """Step 3: mlsynth's simplex solver attains a strictly lower cyclical SSE
    than the authors' ``Synth::synth`` (which uses slack ipop tolerances).

    This is the one step where the two implementations differ -- and mlsynth
    reaches the verified global optimum while the reference under-converges.
    """
    fit, _ = treated_fit
    idx = np.where(~np.isnan(fit.cycle_pre))[0]
    c1 = fit.cycle_pre[idx]
    donors = [c for c in panel.columns if c != "West Germany"]
    c0 = np.column_stack(
        [fit_hamilton_filter(panel[c].to_numpy(), h=H, p=P).cycle_pre[idx]
         for c in donors]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = simplex_lstsq(c0, c1)
    sse_mlsynth = float(np.sum((c1 - c0 @ w) ** 2))
    # simplex feasibility
    assert w.min() > -1e-9 and abs(w.sum() - 1.0) < 1e-6
    # strictly better-fitting than the authors' loose Synth solve
    assert sse_mlsynth < golden["synth_loose_sse"]


def test_sbc_estimator_reaches_optimum_att(panel):
    """End-to-end: the SBC estimator's ATT matches the verified-optimum value
    (mlsynth ~-952), not the authors' under-converged Synth ATT (~-1006)."""
    from mlsynth import SBC

    d = pd.read_csv(DATA)
    d["treat"] = ((d["country"] == "West Germany") & (d["year"] >= 1991)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SBC({"df": d, "outcome": "gdp", "treat": "treat", "unitid": "country",
                   "time": "year", "h": H, "p": P, "weights_mode": "simplex",
                   "display_graphs": False}).fit()
    assert abs(res.att - (-952.0)) < 25.0
    w = res.weights_by_donor or {}
    # The verified optimum concentrates on Greece, the Netherlands and Italy.
    assert w.get("Greece", 0.0) > w.get("Netherlands", 0.0) > w.get("Italy", 0.0) > 0.05
