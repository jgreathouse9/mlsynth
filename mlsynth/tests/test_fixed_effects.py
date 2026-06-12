"""Unit tests for the augsynth ``fixed_effects`` (per-unit demeaning) path.

Covers the fit helper (``fit_augsynth_once``) and the conformal inference
(``conformal_pvalue`` / ``conformal_intervals``) added to reproduce augsynth's
``fixed_effects=TRUE`` behaviour.
"""

import numpy as np
import pytest

from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
from mlsynth.utils.geolift_helpers.marketselect.helpers.fit import fit_augsynth_once


def _panel(T=60, J=6, seed=1):
    """Donor panel with heterogeneous unit levels + a shared trend."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0.0, 3.0, T)
    levels = rng.uniform(5.0, 20.0, J)                     # unit fixed effects
    Y0 = levels[None, :] + trend[:, None] + rng.normal(scale=0.3, size=(T, J))
    return Y0, trend, rng


# --------------------------- fit_augsynth_once ----------------------------

def test_fixed_effects_ridge_produces_nonzero_intercept():
    """fixed_effects demeans each unit, so the ridge fit carries a level intercept."""
    Y0, trend, rng = _panel()
    y = 12.0 + trend + Y0[:, :2].mean(1) * 0.0 + rng.normal(scale=0.3, size=Y0.shape[0])
    fe = fit_augsynth_once(y, Y0, augment="ridge", fixed_effects=True)
    raw = fit_augsynth_once(y, Y0, augment="ridge", fixed_effects=False)
    assert fe.intercept != 0.0                              # level restored via intercept
    assert raw.intercept == 0.0                             # raw ridge centers internally


def test_fixed_effects_counterfactual_is_intercept_plus_weighted_donors():
    """predict == intercept + Y0 @ W (the fixed-effect add-back)."""
    Y0, trend, rng = _panel()
    y = 10.0 + trend + rng.normal(scale=0.3, size=Y0.shape[0])
    fe = fit_augsynth_once(y, Y0, augment="ridge", fixed_effects=True)
    expected = fe.intercept + Y0 @ fe.weights
    np.testing.assert_allclose(fe.predict(Y0), expected, rtol=1e-10)


def test_fixed_effects_recovers_level_shifted_donor():
    """A treated unit equal to a donor + a constant is matched up to the intercept."""
    Y0, trend, rng = _panel()
    y = Y0[:, 0] + 7.0                                      # donor 0 shifted up by 7
    fe = fit_augsynth_once(y, Y0, augment="ridge", fixed_effects=True)
    # gap over the (training) window is ~0: the intercept absorbs the +7 level
    gap = y - fe.predict(Y0)
    assert np.abs(gap).mean() < 0.5
    assert fe.weights[0] > 0.5                              # weight concentrates on donor 0


# ----------------------------- conformal ----------------------------------

def test_conformal_fixed_effects_detects_post_level_shift():
    """A clear post-period level shift is detected by the fixed-effect conformal."""
    Y0, trend, rng = _panel(T=80, J=8, seed=3)
    pre = 60
    y = Y0[:, 0] + 2.0 + rng.normal(scale=0.2, size=Y0.shape[0])
    y[pre:] += 6.0                                          # clear post-period effect
    p_fe = conformal_pvalue(y, Y0, pre, ns=1500, seed=0, fixed_effects=True)
    assert p_fe < 0.10                                      # detected


def test_conformal_fixed_effects_calibrated_under_null():
    """No post effect -> the fixed-effect conformal does not falsely reject."""
    Y0, trend, rng = _panel(T=80, J=8, seed=5)
    pre = 60
    y = Y0[:, 0] + 2.0 + rng.normal(scale=0.2, size=Y0.shape[0])   # no post shift
    p = conformal_pvalue(y, Y0, pre, ns=1500, seed=0, fixed_effects=True)
    assert p > 0.10
