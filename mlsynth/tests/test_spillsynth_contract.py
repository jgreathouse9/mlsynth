"""Result-contract conformance for SPILLSYNTH.

SPILLSYNTH's dispatcher result used to be a bespoke dataclass outside the
standardized two-family contract. These tests pin the migration: every method's
result is a :class:`~mlsynth.config_models.BaseEstimatorResults` exposing the
standard sub-models (so the shared tooling -- including
:func:`~mlsynth.utils.counterfactual_compare.compare_counterfactuals` -- reads it
without bespoke glue), while keeping the spillover-specific accessors intact.

Test-first per CLAUDE.md. SAR conformance lives in ``test_spillsynth_sar.py``
because that method needs spatial-weight inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlsynth import SPILLSYNTH
from mlsynth.config_models import (
    BaseEstimatorResults,
    EffectResult,
    MlsynthResult,
)
from mlsynth.utils.counterfactual_compare import compare_counterfactuals


def _panel(*, N=8, T=40, T0=30, treatment=-3.0, spillover=1.5, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=N)
    f = np.cumsum(rng.standard_normal(T)) * 0.4 + 0.05 * np.arange(T)
    intercept = rng.uniform(-1, 1, size=N)
    Y = intercept[:, None] + np.outer(loadings, f) + 0.1 * rng.standard_normal((N, T))
    Y[0, T0:] += treatment
    Y[1, T0:] += spillover
    D = np.zeros((N, T))
    D[0, T0:] = 1
    rows = [
        {"unit": f"u{i}", "year": t, "y": float(Y[i, t]), "treat": int(D[i, t])}
        for i in range(N) for t in range(T)
    ]
    return pd.DataFrame(rows)


def _cfg(method):
    return dict(df=_panel(), outcome="y", treat="treat", unitid="unit",
                time="year", method=method, affected_units=["u1"],
                display_graphs=False)


METHODS = ["cd", "iscm", "grossi", "iterative"]


@pytest.fixture(scope="module", params=METHODS)
def fitted(request):
    return request.param, SPILLSYNTH(_cfg(request.param)).fit()


# --------------------------------------------------------------------------- #
# Family membership
# --------------------------------------------------------------------------- #
def test_is_effect_result(fitted):
    _, res = fitted
    assert isinstance(res, MlsynthResult)
    assert isinstance(res, BaseEstimatorResults)
    assert isinstance(res, EffectResult)


# --------------------------------------------------------------------------- #
# Standard sub-models populated
# --------------------------------------------------------------------------- #
def test_standard_submodels_populated(fitted):
    _, res = fitted
    assert res.effects is not None and res.effects.att is not None
    assert res.time_series is not None
    assert res.time_series.counterfactual_outcome is not None
    assert res.weights is not None
    assert res.method_details is not None and res.method_details.method_name


def test_method_name_records_the_method(fitted):
    method, res = fitted
    assert method in res.method_details.method_name


def test_flat_att_matches_effects(fitted):
    _, res = fitted
    assert isinstance(res.att, float)
    assert res.att == pytest.approx(res.effects.att)


def test_pre_rmse_is_finite_float(fitted):
    _, res = fitted
    assert isinstance(res.pre_rmse, float)
    assert np.isfinite(res.pre_rmse)


# --------------------------------------------------------------------------- #
# The standardized series is the full pre+post path against observed
# --------------------------------------------------------------------------- #
def test_full_series_spans_the_panel(fitted):
    _, res = fitted
    inp = res.inputs
    cf = np.asarray(res.time_series.counterfactual_outcome, dtype=float)
    obs = np.asarray(res.time_series.observed_outcome, dtype=float)
    t = np.asarray(res.time_series.time_periods)
    assert cf.shape == obs.shape == t.shape
    assert len(cf) == len(inp.time_labels)
    # the post block is the routed spillover-adjusted counterfactual
    np.testing.assert_allclose(cf[inp.T0:], np.asarray(res.counterfactual, float))
    # the gap is observed minus counterfactual
    np.testing.assert_allclose(
        np.asarray(res.time_series.estimated_gap, float), obs - cf)


def test_post_accessors_keep_shape(fitted):
    _, res = fitted
    cf_post = np.asarray(res.counterfactual)
    gap_post = np.asarray(res.gap)
    assert cf_post.ndim == 1 and cf_post.shape == gap_post.shape


# --------------------------------------------------------------------------- #
# Serializes, and the spillover-specific API still works
# --------------------------------------------------------------------------- #
def test_serializable(fitted):
    _, res = fitted
    dumped = res.model_dump(include={"effects", "fit_diagnostics", "inference"})
    assert dumped["effects"]["att"] is not None


def test_spillover_accessors_preserved(fitted):
    _, res = fitted
    assert isinstance(res.att_scm, float)
    assert res.gap_scm.ndim == 1
    assert "u1" in res.spillover_effects


# --------------------------------------------------------------------------- #
# The whole point: the helper reads a SPILLSYNTH result with no bespoke glue
# --------------------------------------------------------------------------- #
def test_compare_counterfactuals_auto_resolves(fitted):
    method, res = fitted
    cmp = compare_counterfactuals({method: res})
    # observed and att came straight off the standardized surface
    assert cmp.observed is not None
    assert cmp.summary.loc[method, "att"] == pytest.approx(res.att)
    n = len(res.inputs.time_labels)
    assert len(cmp.curves) == n
