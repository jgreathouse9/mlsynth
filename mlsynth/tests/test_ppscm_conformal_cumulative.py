"""PPSCM per-unit cumulative conformal bands.

PPSCM reports each treated unit's cumulative effect as a point estimate and has no
interval calibrated for it: the existing per-unit bands come from the CFPT
out-of-sample term alone, which under-covers, and the in-sample bound mlsynth ships
is derived for unconstrained weights while PPSCM's live on a simplex.

This suite pins the calibrated alternative. The construction is the one added in
#431, with the scores built the way PPSCM can afford: partially-pooled SCM fits every
treated unit in a single solve, so one rolling pass over origins yields a score vector
for *each* unit at the cost of one pooled solve per origin -- not one per unit per
origin. The calibration itself is not reimplemented; it is
``conformal.cumulative_conformal_interval``, so a defect in the order statistic has
one place to be fixed.

Levels: smoke, unit invariants, edge, failure.
"""
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.conformal import cumulative_conformal_interval
from mlsynth.utils.ppscm_helpers.engine import run_multisynth


def _panel(n_ctrl=10, n_trt=2, t0=40, horizon=4, rho=0.0, sigma=0.3, effect=0.0,
           n_factors=2, seed=0):
    """Small factor panel with simultaneous adoption (fast enough for a QP loop)."""
    rng = np.random.default_rng(seed)
    T = t0 + horizon
    n = n_ctrl + n_trt
    common = rng.uniform(0.3, 1.3, (n, n_factors)) @ rng.normal(size=(n_factors, T))
    e = rng.normal(scale=sigma, size=(n, T))
    if rho:
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    Xy = rng.normal(scale=2.0, size=n)[:, None] + common + e
    Xy[:n_trt, t0:] += effect
    trt = np.full(n, np.inf)
    trt[:n_trt] = t0
    return Xy, trt


@pytest.fixture(scope="module")
def panel():
    return _panel()


def _call(panel, **kw):
    from mlsynth.utils.ppscm_helpers.inference import cumulative_conformal_per_unit
    Xy, trt = panel
    base = dict(d=40, n_leads=4, n_lags=40, fixedeff=True, time_cohort=False,
                nu_used=0.5, lam=0.0, solver=None, alpha=0.1, horizon=4,
                min_train_frac=0.3)
    base.update(kw)
    return cumulative_conformal_per_unit(Xy, trt, **base)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_smoke_returns_one_band_per_treated_unit(panel):
    point, lower, upper, n_scores = _call(panel)
    for arr in (point, lower, upper, n_scores):
        assert arr.shape == (2,)                     # one row per treated cohort
    assert np.isfinite(point).all()
    assert (lower <= point).all() and (point <= upper).all()


# ---------------------------------------------------------------------------
# Unit invariants
# ---------------------------------------------------------------------------

def test_point_is_each_unit_summed_effect_over_the_horizon(panel):
    Xy, trt = panel
    full = run_multisynth(Xy, trt, 40, 4, 40, fixedeff=True, time_cohort=False,
                          nu=0.5, lam=0.0, solver=None)
    expected = [float(np.sum(np.asarray(full["tau_rel"][k], float)[:4]))
                for k in range(len(full["groups"]))]
    point, *_ = _call(panel)
    np.testing.assert_allclose(point, expected, rtol=1e-6, atol=1e-8)


def test_bands_are_symmetric_about_their_points(panel):
    point, lower, upper, _ = _call(panel)
    np.testing.assert_allclose(point - lower, upper - point, rtol=1e-9, atol=1e-9)


def test_calibration_is_the_shared_combiner_not_a_second_implementation(panel):
    """Guards the reuse: a divergence from the package must fail, not drift."""
    from mlsynth.utils.ppscm_helpers.inference import (
        cumulative_conformal_per_unit, rolling_pooled_block_sums)
    Xy, trt = panel
    scores = rolling_pooled_block_sums(
        Xy, trt, d=40, n_leads=4, n_lags=40, fixedeff=True, time_cohort=False,
        nu_used=0.5, lam=0.0, solver=None, horizon=4, min_train_frac=0.3)
    point, lower, upper, n_scores = cumulative_conformal_per_unit(
        Xy, trt, d=40, n_leads=4, n_lags=40, fixedeff=True, time_cohort=False,
        nu_used=0.5, lam=0.0, solver=None, alpha=0.1, horizon=4, min_train_frac=0.3)
    for k in range(len(point)):
        expected = cumulative_conformal_interval(point[k], scores[k], alpha=0.1, horizon=4)
        assert lower[k] == pytest.approx(expected.lower)
        assert upper[k] == pytest.approx(expected.upper)
        assert n_scores[k] == expected.n_scores


def test_one_pooled_solve_per_origin_not_one_per_unit(panel, monkeypatch):
    """The efficiency the pooled fit buys: scores for every unit from one solve."""
    import mlsynth.utils.ppscm_helpers.inference as inf
    calls = {"n": 0}
    real = inf.run_multisynth

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(inf, "run_multisynth", counting)
    _, _, _, n_scores = _call(panel)
    n_units = len(n_scores)
    n_origins = int(n_scores[0])
    assert n_units == 2 and n_origins >= 2
    # one solve per origin, plus the full fit -- NOT n_origins * n_units
    assert calls["n"] == n_origins + 1
    assert calls["n"] < n_origins * n_units + 1


def test_smaller_alpha_never_narrows_the_bands(panel):
    _, lo_tight, hi_tight, _ = _call(panel, alpha=0.05)
    _, lo_loose, hi_loose, _ = _call(panel, alpha=0.30)
    assert ((hi_tight - lo_tight) >= (hi_loose - lo_loose) - 1e-12).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_too_few_windows_gives_honest_infinite_bands(panel):
    """A horizon that leaves too few windows must say so, not guess."""
    _, lower, upper, n_scores = _call(panel, horizon=4, min_train_frac=0.9)
    assert (n_scores < 9).all()
    assert np.isinf(lower).all() and np.isinf(upper).all()


def test_single_treated_unit_is_supported():
    p = _panel(n_trt=1)
    point, lower, upper, _ = _call(p)
    assert point.shape == (1,)
    assert np.isfinite(point).all()


# ---------------------------------------------------------------------------
# Failure -- translated errors, reported not swallowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_invalid_alpha_raises_config_error(panel, bad):
    with pytest.raises(MlsynthConfigError):
        _call(panel, alpha=bad)


@pytest.mark.parametrize("bad", [0, -2])
def test_nonpositive_horizon_raises_config_error(panel, bad):
    with pytest.raises(MlsynthConfigError):
        _call(panel, horizon=bad)


def test_horizon_beyond_the_post_window_raises_data_error(panel):
    with pytest.raises(MlsynthDataError):
        _call(panel, horizon=99)


def test_no_treated_units_raises_data_error():
    Xy, trt = _panel()
    trt = np.full(trt.shape, np.inf)                 # nobody adopts
    with pytest.raises(MlsynthDataError):
        _call((Xy, trt))
