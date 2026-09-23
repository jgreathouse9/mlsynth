"""Per-step cross-validation of mlsynth's SBC against Shi-Xi-Xie's own R code,
on the Hong Kong handover panel.

The German panel has been pinned step by step since the SBC estimator landed
(``test_sbc_reference.py``). The handover had not: its reference bundle captures
what the authors' ``Synth::synth`` produces -- the ATT, the cyclical SSE, the
weights -- and nothing about the detrending that feeds it. The ``sbc_hongkong``
benchmark carried a detrending check all the same, but that check compares
mlsynth's Hamilton filter to a base-numpy rebuild of the authors' design, so it
says the two Python readings agree and says nothing about R.

These tests close that. ``benchmarks/reference/sbc_hongkong/golden_steps.R``
runs the authors' ``lsq`` and ``trend_predict`` on the same in-repo panel
(regenerate with ``Rscript benchmarks/reference/sbc_hongkong/golden_steps.R >
benchmarks/reference/sbc_hongkong/golden_steps.txt``; base R, no CRAN package
needed), and mlsynth is pinned against each step at the precision the capture
carries. What bounds that precision, and why exact equality is not on offer, is
in ``mlsynth/tests/_golden_steps.py``.

The convention this locks in, matching the German panel: the treated unit is
detrended on the pre-treatment window and the donors on the full sample, since
the donors are untreated. Getting that backwards changes the donor cycles and
therefore the weights, which is why it is asserted here and not assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _golden_steps import (assert_reproduces, assert_resolves_enough_digits,
                           load as load_golden)
from mlsynth.utils.sbc_helpers.hamilton import fit_hamilton_filter
from mlsynth.utils.sbc_helpers.trend_forecast import forecast_treated_trend

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "benchmarks" / "reference" / "sbc_hongkong" / "golden_steps.txt"
DATA = ROOT / "basedata" / "hong_kong_handover.csv"

H, P = 4, 2
TREATED = "Hong Kong"
DONORS = ["Australia", "Austria", "Korea", "Canada", "Denmark", "France",
          "Germany", "Italy", "Netherlands", "New Zealand", "US"]
CAPTURED_DONORS = ["Italy", "Germany", "US", "Korea"]
COMPARED_KEYS = (("treated_trend_coef", "treated_cycle_pre", "trend_forecast")
                 + tuple(f"donor_cycle_full:{d}" for d in CAPTURED_DONORS))


@pytest.fixture(scope="module")
def golden() -> dict:
    return load_golden(GOLDEN)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip(f"data missing: {DATA}")
    wide = pd.read_csv(DATA).pivot(index="year", columns="country", values="gdp")
    return wide.sort_index()


@pytest.fixture(scope="module")
def treated_fit(panel):
    """1997 is the first post-treatment period, so the pre-window ends in 1996."""
    T0 = int((panel.index < 1997).sum())
    return fit_hamilton_filter(panel[TREATED].to_numpy()[:T0], h=H, p=P), T0


def test_golden_fixture_resolves_eight_significant_digits(golden):
    """The capture's precision is the ceiling on every comparison here, so it
    fails first if a regeneration ever records fewer digits."""
    assert_resolves_enough_digits(golden, COMPARED_KEYS)


def test_hamilton_treated_trend_coefficients(golden, treated_fit):
    """Step 1: Hong Kong's Hamilton AR coefficients match R's ``lsq``."""
    fit, _ = treated_fit
    assert_reproduces(fit.coefficients, golden, "treated_trend_coef")


def test_hamilton_treated_cycle(golden, treated_fit):
    """Step 1: the pre-handover cyclical component matches R's ``lsq``."""
    fit, _ = treated_fit
    assert_reproduces(fit.cycle_pre[~np.isnan(fit.cycle_pre)],
                      golden, "treated_cycle_pre")


@pytest.mark.parametrize("donor", CAPTURED_DONORS)
def test_hamilton_donor_cycle_full_sample(golden, panel, donor):
    """Step 1: donor cycles are detrended on the FULL sample, matching R."""
    cyc = fit_hamilton_filter(panel[donor].to_numpy(), h=H, p=P).cycle_pre
    assert_reproduces(cyc[~np.isnan(cyc)], golden, f"donor_cycle_full:{donor}")


def test_trend_forecast(golden, panel, treated_fit):
    """Step 2: the post-handover trend extrapolation matches R's ``trend_predict``."""
    fit, T0 = treated_fit
    fc = forecast_treated_trend(y_target=panel[TREATED].to_numpy(),
                                treated_fit=fit, T0=T0, horizon=H)
    assert_reproduces(fc, golden, "trend_forecast")


def test_donors_are_not_detrended_on_the_pre_window(golden, panel):
    """The asymmetry is the point: donors use the full sample, the treated unit
    does not.

    Detrending a donor on the pre-window instead produces a different cycle, and
    a different cycle produces different weights. Since both readings of the
    procedure look reasonable in prose, this asserts which one the reference
    does, by showing the pre-window reading fails the same comparison the full
    sample passes.
    """
    T0 = int((panel.index < 1997).sum())
    ref, _tol = golden["donor_cycle_full:Italy"]
    pre_only = fit_hamilton_filter(panel["Italy"].to_numpy()[:T0], h=H, p=P).cycle_pre
    pre_only = pre_only[~np.isnan(pre_only)]
    # The pre-window reading is shorter, and disagrees where the two overlap.
    assert len(pre_only) < len(ref)
    n = len(pre_only)
    assert np.max(np.abs(pre_only - ref[:n])) > 1.0


def test_the_cyclical_program_matches_the_reference_dimensions(golden, panel, treated_fit):
    """Step 3's inputs: the treated cycle and the donor matrix the weight solve
    sees are the ones R hands to ``Synth::synth`` -- same rows, same donors."""
    fit, _ = treated_fit
    idx = np.where(~np.isnan(fit.cycle_pre))[0]
    c0 = np.column_stack(
        [fit_hamilton_filter(panel[d].to_numpy(), h=H, p=P).cycle_pre[idx]
         for d in DONORS])
    assert c0.shape == (len(golden["treated_cycle_pre"][0]), len(DONORS))
    assert np.all(np.isfinite(c0))
