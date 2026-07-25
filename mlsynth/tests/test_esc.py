"""Tests for ESC -- Ensemble Synthetic Control (Lian & Pu 2026).

Layered per the repo's TDD contract: a smoke test, recovery of a known
counterfactual, prediction-interval invariants, determinism, config-validation
failures, data-error failures, edge cases, and a plotting smoke test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import ESC
from mlsynth.utils.esc_helpers import ESCConfig, ESCResults
from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError

# Fast settings so the ensemble tests stay quick.
_FAST = dict(n_learners=40, n_alphas=12, cv_folds=3, seed=0)


def _panel(J=12, T=16, T0=8, effect=-5.0, noise=0.05, seed=1):
    """Treated unit = a sparse linear combo of donors + a known post effect."""
    rng = np.random.default_rng(seed)
    # donor outcome series: shared factors + idiosyncratic drift
    F = np.cumsum(rng.normal(0, 1, size=(T, 3)), axis=0)
    load = rng.normal(0, 1, size=(3, J))
    donors = F @ load + rng.normal(0, 0.3, size=(T, J)) + 20.0
    w = np.zeros(J); k = min(3, J); w[:k] = [0.5, 0.3, 0.2][:k]
    treated = donors @ w + rng.normal(0, noise, size=T)
    treated[T0:] += effect                      # inject the post-treatment effect
    rows = []
    for j in range(J):
        for t in range(T):
            rows.append(("d%02d" % j, t, donors[t, j], 0))
    for t in range(T):
        rows.append(("treated", t, treated[t], int(t >= T0)))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "treat"])


def _cfg(df, **extra):
    base = dict(df=df, outcome="y", treat="treat", unitid="unit", time="time",
                display_graphs=False, **_FAST)
    base.update(extra)
    return base


# ---------------------------------------------------------------- smoke
def test_smoke_returns_effectresult():
    r = ESC(_cfg(_panel())).fit()
    assert isinstance(r, ESCResults)
    assert np.isfinite(r.effects.att)
    ts = r.time_series
    T = len(_panel().time.unique())
    assert np.asarray(ts.counterfactual_outcome).shape == (T,)
    assert np.all(np.isfinite(ts.counterfactual_outcome))
    # flat convenience accessors resolve
    assert r.att == pytest.approx(r.effects.att)
    assert np.asarray(r.counterfactual).shape == (T,)
    assert isinstance(r.donor_weights, dict)


# ------------------------------------------------------------- recovery
def test_recovers_known_effect():
    r = ESC(_cfg(_panel(effect=-5.0, noise=0.05), lambda_rule="min")).fit()
    assert r.effects.att == pytest.approx(-5.0, abs=2.0)
    assert r.fit_diagnostics.rmse_pre < 2.0
    # the ensemble counterfactual tracks the treated pre-period (recovery)
    obs = np.asarray(r.time_series.observed_outcome, float)
    cf = np.asarray(r.time_series.counterfactual_outcome, float)
    assert np.corrcoef(obs[:8], cf[:8])[0, 1] > 0.9
    assert len(r.weights.donor_weights) >= 1


# ---------------------------------------------------- prediction interval
def test_prediction_interval_present_and_ordered():
    r = ESC(_cfg(_panel(), alpha=0.10)).fit()
    ts = r.time_series
    assert ts.has_prediction_interval
    lo = np.asarray(ts.counterfactual_lower, float)
    hi = np.asarray(ts.counterfactual_upper, float)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi >= lo - 1e-9)                       # ordered everywhere
    assert ts.prediction_interval_level == pytest.approx(0.90)
    assert ts.prediction_interval_kind and "esc" in ts.prediction_interval_kind.lower()


def test_att_interval_from_ensemble():
    r = ESC(_cfg(_panel(), inference=True)).fit()
    assert r.inference is not None
    lo, hi = r.att_ci
    assert lo is not None and hi is not None and lo <= r.effects.att <= hi
    assert r.inference.confidence_level == pytest.approx(0.90)


def test_wider_band_under_more_noise():
    narrow = ESC(_cfg(_panel(noise=0.05, seed=3))).fit()
    wide = ESC(_cfg(_panel(noise=3.0, seed=3))).fit()
    T0 = 8
    wn = np.nanmean((narrow.time_series.counterfactual_upper
                     - narrow.time_series.counterfactual_lower)[T0:])
    ww = np.nanmean((wide.time_series.counterfactual_upper
                     - wide.time_series.counterfactual_lower)[T0:])
    assert ww > wn


# ----------------------------------------------------------- determinism
def test_deterministic_under_seed():
    a = ESC(_cfg(_panel(), seed=7)).fit()
    b = ESC(_cfg(_panel(), seed=7)).fit()
    assert a.effects.att == pytest.approx(b.effects.att)
    np.testing.assert_allclose(a.time_series.counterfactual_outcome,
                               b.time_series.counterfactual_outcome)


# ------------------------------------------------- config-validation errors
@pytest.mark.parametrize("bad", [
    {"subspace_ratio": 1.5}, {"subspace_ratio": 0.0}, {"n_learners": 1},
    {"block_length": 0}, {"alpha": 0.0}, {"alpha": 1.0}, {"block_keep": 0.0},
    {"lambda_rule": "aic"}, {"aggregate": "mode"}, {"cv_folds": 1},
])
def test_bad_config_raises(bad):
    with pytest.raises(MlsynthConfigError):
        ESC(_cfg(_panel(), **bad))


def test_trimmed_without_trim_raises():
    with pytest.raises(MlsynthConfigError):
        ESC(_cfg(_panel(), aggregate="trimmed", trim=0.0))


# ---------------------------------------------------------- data errors
@pytest.mark.parametrize("agg", ["median", "trimmed"])
def test_aggregation_variants(agg):
    r = ESC(_cfg(_panel(), aggregate=agg)).fit()
    assert np.isfinite(r.effects.att)
    assert r.time_series.has_prediction_interval


def test_multiple_treated_cohorts_raises():
    df = _panel()
    df.loc[(df.unit == "d00") & (df.time >= 6), "treat"] = 1   # 2nd treated, other time
    with pytest.raises(MlsynthDataError):
        ESC(_cfg(df)).fit()


def test_single_pre_period_raises():
    df = _panel()
    df.loc[(df.unit == "treated") & (df.time >= 1), "treat"] = 1  # only t=0 is pre
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        ESC(_cfg(df)).fit()


def test_no_donors_raises():
    df = _panel()
    df = df[df.unit == "treated"]                        # only the treated unit
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        ESC(_cfg(df)).fit()


def test_no_pretreatment_periods_raises():
    df = _panel()
    df.loc[df.unit == "treated", "treat"] = 1            # treated from t=0
    with pytest.raises((MlsynthDataError, MlsynthConfigError)):
        ESC(_cfg(df)).fit()


# ------------------------------------------------------------- edge cases
def test_single_donor_runs():
    df = _panel(J=1, T=14, T0=7, noise=0.1)
    r = ESC(_cfg(df, subspace_ratio=1.0)).fit()
    assert np.isfinite(r.effects.att)


def test_short_pre_window_runs():
    df = _panel(J=8, T=10, T0=3, noise=0.2)
    r = ESC(_cfg(df, block_length=2)).fit()
    assert np.isfinite(r.effects.att)
    assert r.time_series.has_prediction_interval


# ------------------------------------------------------ plotting smoke
def test_plot_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = str(tmp_path / "esc")
    r = ESC(_cfg(_panel(), display_graphs=True, save=out)).fit()
    assert np.isfinite(r.effects.att)
