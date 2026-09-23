"""The conformal test on GEOX's backtests: which block, and a valid p-value.

Three defects, found by Monte-Carloing the design against a known truth and by
reading the result against GeoCausality's independent implementation of the same
GeoLift idea.

Scope. ``conformal_pvalue`` itself was never wrong -- on a true null its
p-values come back near uniform. What was wrong is how the design calls it, and
what the p-value is allowed to return at the extreme.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
from mlsynth.utils.geox_helpers.engines import resolve_engine
from mlsynth.utils.geox_helpers.windows import (
    backtest_pre_periods,
    backtest_treatment_window,
)

T, D, J = 80, 10, 12


@pytest.fixture(scope="module")
def series():
    """A donor pool and a treated series that tracks it, with AR(1) structure."""
    rng = np.random.default_rng(0)
    f = np.zeros((T, 2))
    for k in range(2):
        e = rng.normal(size=T)
        f[0, k] = e[0]
        for t in range(1, T):
            f[t, k] = 0.6 * f[t - 1, k] + e[t]
    load = rng.normal(size=(J, 2)) + 2.0
    Y0 = 100.0 + f @ load.T + rng.normal(scale=0.3, size=(T, J))
    y = 100.0 + f @ load.mean(axis=0) + rng.normal(scale=0.3, size=T)
    return y, Y0


def _fit(y, Y0, n_pre, a, b):
    eng = resolve_engine("augsynth")
    return eng, eng.fit_once(y, Y0, n_pre, a, b, 1,
                             augment="ridge", fixed_effects=True)


# --- the tested block is the backtest window ------------------------------


@pytest.mark.parametrize("s", range(1, 9))
def test_periods_after_the_window_cannot_move_the_p_value(series, s):
    """A shock outside the pseudo-experiment must not change its p-value.

    The design tested ``resids[n_pre:]`` -- everything to the end of the panel
    -- so for backtest ``s`` it evaluated ``D + s - 1`` periods instead of the
    window's ``D``. Only ``s = 1`` coincided. The extra periods were never part
    of the pseudo-experiment and carry none of its injected effect.

    Corrupting them is the sharpest statement of the bug: if they enter the
    statistic, the p-value moves; if the tested block is the window, it cannot.
    """
    y, Y0 = series
    n_pre = backtest_pre_periods(T, D, s)
    a, b = backtest_treatment_window(T, D, s)
    eng, fit = _fit(y, Y0, n_pre, a, b)

    y_shock = np.array(y, dtype=float)
    y_shock[b + 1:] += 500.0                      # far outside the window
    if b + 1 >= T:                                # s = 1 has nothing after it
        pytest.skip("backtest 1 ends flush with the panel")

    base, _ = eng.point_inference(fit, y, Y0, n_pre, a, b,
                                  inference="conformal", ns=200, seed=0)
    shocked, _ = eng.point_inference(fit, y_shock, Y0, n_pre, a, b,
                                     inference="conformal", ns=200, seed=0)
    assert base == shocked


@pytest.mark.parametrize("s", range(1, 9))
def test_the_sweep_tests_the_window_too(series, s):
    """Same property through ``sweep_p_values``, which is what the design uses."""
    y, Y0 = series
    n_pre = backtest_pre_periods(T, D, s)
    a, b = backtest_treatment_window(T, D, s)
    eng, fit = _fit(y, Y0, n_pre, a, b)
    if b + 1 >= T:
        pytest.skip("backtest 1 ends flush with the panel")

    y_shock = np.array(y, dtype=float)
    y_shock[b + 1:] += 500.0
    kw = dict(inference="conformal", ns=200, seed=0, analytic=True,
              augment="ridge", fixed_effects=True)
    base = eng.sweep_p_values(fit, y, Y0, n_pre, a, b, [0.0, 0.1], **kw)
    shocked = eng.sweep_p_values(fit, y_shock, Y0, n_pre, a, b, [0.0, 0.1], **kw)
    assert base["p_value"] == shocked["p_value"]


def test_the_realized_readout_is_unaffected(series):
    """The readout's window already ends at the panel, so nothing changes there.

    Truncation is a no-op when ``end`` is the last period, which is the case for
    every realized readout. Pinning it keeps the fix from being read as a change
    to the numbers a finished experiment reports.
    """
    y, Y0 = series
    n_pre = T - D
    eng, fit = _fit(y, Y0, n_pre, n_pre, T - 1)
    p, details = eng.point_inference(fit, y, Y0, n_pre, n_pre, T - 1,
                                     inference="conformal", ns=200, seed=0)
    assert 0.0 < p <= 1.0
    assert details["method"] == "conformal"


# --- a p-value is never zero ----------------------------------------------


def test_the_p_value_is_bounded_away_from_zero(series):
    """``(1 + #{perm >= obs}) / (1 + ns)``, so never exactly zero.

    ``mean(obs <= perm)`` returns 0 when the observed statistic exceeds every
    permutation, and zero is not a valid p-value: with ``ns`` draws the evidence
    cannot exceed ``1 / (ns + 1)``. Under a large injected effect the old form
    returned exactly 0 in half of all backtests.
    """
    y, Y0 = series
    n_pre = T - D
    huge = np.array(y, dtype=float)
    huge[n_pre:] *= 5.0                            # far beyond any permutation
    p = conformal_pvalue(huge, Y0, n_pre, ns=200, seed=0,
                         fixed_effects=True, finite_sample=True)
    assert p > 0.0
    assert p == pytest.approx(1.0 / 201.0)


def test_the_uncorrected_form_is_still_reachable(series):
    """augsynth's convention stays available, since a benchmark reproduces it.

    ``geox_augsynth_geolift`` matches GeoLift's published values, and GeoLift
    inherits augsynth's ``mean(obs <= perm)``. The correction is therefore an
    option and not a silent replacement.
    """
    y, Y0 = series
    n_pre = T - D
    huge = np.array(y, dtype=float)
    huge[n_pre:] *= 5.0
    assert conformal_pvalue(huge, Y0, n_pre, ns=200, seed=0,
                            fixed_effects=True, finite_sample=False) == 0.0


@pytest.mark.parametrize("finite_sample", [True, False])
def test_the_p_value_stays_a_probability(series, finite_sample):
    y, Y0 = series
    n_pre = T - D
    p = conformal_pvalue(y, Y0, n_pre, ns=200, seed=0, fixed_effects=True,
                         finite_sample=finite_sample)
    assert 0.0 <= p <= 1.0


def test_the_correction_never_lowers_the_p_value(series):
    """It shifts upward by construction, so it cannot manufacture significance."""
    y, Y0 = series
    for s in range(1, 6):
        n_pre = backtest_pre_periods(T, D, s)
        a, b = backtest_treatment_window(T, D, s)
        inj = np.array(y, dtype=float)
        inj[a:b + 1] *= 1.15
        kw = dict(ns=200, seed=0, fixed_effects=True)
        assert (conformal_pvalue(inj[:b + 1], Y0[:b + 1], n_pre,
                                 finite_sample=True, **kw)
                >= conformal_pvalue(inj[:b + 1], Y0[:b + 1], n_pre,
                                    finite_sample=False, **kw))


# --- block is the default for a serially dependent panel ------------------


def test_block_is_the_default_conformal_scheme():
    """Geo panels are serially dependent, which iid permutation denies.

    Cyclic shifts preserve the dependence and have the panel's own length as
    their reference set, so they cannot return a degenerate zero.
    """
    from mlsynth.utils.geox_helpers.config import GEOXConfig

    assert GEOXConfig.model_fields["conformal_type"].default == "block"


def test_iid_remains_selectable(series):
    y, Y0 = series
    n_pre = T - D
    for scheme in ("iid", "block"):
        p = conformal_pvalue(y, Y0, n_pre, ns=200, seed=0, fixed_effects=True,
                             conformal_type=scheme)
        assert 0.0 <= p <= 1.0
