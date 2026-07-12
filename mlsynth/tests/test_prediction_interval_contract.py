"""Canonical per-period prediction-interval band on the result contract.

Every method that emits a per-period band around the counterfactual stores it in
its own typed object under its own names (scpi's ``cf_lower/cf_upper``,
conformal's ``pointwise_lower/upper``, a Bayesian credible band ...). This is the
fragmentation the library exists to remove, pushed inside the result objects.

This pins a single canonical representation on
:class:`~mlsynth.config_models.TimeSeriesResults`: per-period
``counterfactual_lower`` / ``counterfactual_upper`` (pointwise) and their
``*_simultaneous`` (joint-coverage) siblings, aligned to ``time_periods`` and NaN
where the method has no band, plus a ``prediction_interval_level`` /
``prediction_interval_kind`` tag so consumers can label honestly. Post-only bands
are spread onto the full time axis by
:func:`~mlsynth.utils.results_helpers.normalize_counterfactual_band`, and
:func:`~mlsynth.utils.results_helpers.build_effect_submodels` populates the fields
from one normalized spec.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.config_models import TimeSeriesResults
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.results_helpers import (
    build_effect_submodels, normalize_counterfactual_band,
)


# --------------------------------------------------------------------------
# the contract field
# --------------------------------------------------------------------------
def test_band_fields_default_absent():
    ts = TimeSeriesResults()
    for f in ("counterfactual_lower", "counterfactual_upper",
              "counterfactual_lower_simultaneous",
              "counterfactual_upper_simultaneous",
              "prediction_interval_level", "prediction_interval_kind"):
        assert getattr(ts, f) is None
    assert ts.has_prediction_interval is False


def test_has_prediction_interval_true_when_present():
    ts = TimeSeriesResults(
        counterfactual_lower=np.array([np.nan, np.nan, 1.0, 1.2]),
        counterfactual_upper=np.array([np.nan, np.nan, 2.0, 2.4]))
    assert ts.has_prediction_interval is True


# --------------------------------------------------------------------------
# the normalizer
# --------------------------------------------------------------------------
def test_full_length_band_passes_through():
    t = np.arange(5)
    lo, hi = normalize_counterfactual_band(
        lower=[0, 0, 1, 1, 1], upper=[2, 2, 3, 3, 3], time_periods=t)
    assert np.allclose(lo, [0, 0, 1, 1, 1])
    assert np.allclose(hi, [2, 2, 3, 3, 3])


def test_post_only_band_aligns_onto_axis():
    t = np.array([2000, 2001, 2002, 2003, 2004])   # T0 = 3
    lo, hi = normalize_counterfactual_band(
        lower=[1.0, 1.5], upper=[2.0, 2.5],
        periods=[2003, 2004], time_periods=t)
    assert np.isnan(lo[:3]).all() and np.isnan(hi[:3]).all()   # pre-period empty
    assert np.allclose(lo[3:], [1.0, 1.5])
    assert np.allclose(hi[3:], [2.0, 2.5])


def test_one_sided_band_rejected():
    with pytest.raises(MlsynthConfigError):
        normalize_counterfactual_band(lower=[1, 2], upper=None,
                                      time_periods=np.arange(2))


def test_length_mismatch_rejected():
    with pytest.raises(MlsynthConfigError):
        normalize_counterfactual_band(lower=[1, 2, 3], upper=[1, 2],
                                      time_periods=np.arange(3))


def test_absent_band_returns_none():
    lo, hi = normalize_counterfactual_band(lower=None, upper=None,
                                           time_periods=np.arange(4))
    assert lo is None and hi is None


# --------------------------------------------------------------------------
# wired through the shared builder
# --------------------------------------------------------------------------
def test_build_effect_submodels_populates_canonical_band():
    obs = np.array([10.0, 11, 12, 9, 8], dtype=float)
    cf = np.array([10.0, 11, 12, 12, 12], dtype=float)
    sub = build_effect_submodels(
        obs, cf, n_pre_periods=3, n_post_periods=2,
        time_periods=np.array([2000, 2001, 2002, 2003, 2004]),
        prediction_interval={
            "lower": [10.5, 10.0], "upper": [13.5, 14.0],
            "periods": [2003, 2004], "level": 0.90, "kind": "scpi:simplex",
        },
    )
    ts = sub["time_series"]
    assert ts.has_prediction_interval is True
    assert len(ts.counterfactual_lower) == 5
    assert np.isnan(ts.counterfactual_lower[:3]).all()
    assert np.allclose(ts.counterfactual_lower[3:], [10.5, 10.0])
    assert np.allclose(ts.counterfactual_upper[3:], [13.5, 14.0])
    assert ts.prediction_interval_level == pytest.approx(0.90)
    assert ts.prediction_interval_kind == "scpi:simplex"


def test_build_effect_submodels_simultaneous_band():
    obs = np.zeros(4); cf = np.zeros(4)
    sub = build_effect_submodels(
        obs, cf, n_pre_periods=2, n_post_periods=2,
        time_periods=np.arange(4),
        prediction_interval={
            "lower": [-1, -1, -1, -1], "upper": [1, 1, 1, 1],
            "lower_simultaneous": [-2, -2, -2, -2],
            "upper_simultaneous": [2, 2, 2, 2],
        },
    )
    ts = sub["time_series"]
    assert np.allclose(ts.counterfactual_lower_simultaneous, [-2, -2, -2, -2])
    assert np.allclose(ts.counterfactual_upper_simultaneous, [2, 2, 2, 2])


def test_build_effect_submodels_no_band_leaves_fields_none():
    obs = np.zeros(4); cf = np.zeros(4)
    sub = build_effect_submodels(obs, cf, n_pre_periods=2, n_post_periods=2)
    assert sub["time_series"].has_prediction_interval is False
    assert sub["time_series"].counterfactual_lower is None
