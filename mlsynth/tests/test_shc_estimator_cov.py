"""End-to-end coverage tests for mlsynth.estimators.shc.SHC.

These exercise the full SHC pipeline (setup -> orchestration -> inference ->
plotting) via small synthetic panels with fixed seeds, plus the estimator's
config-normalisation, plotting-color, and error-propagation branches.

Structural assertions only (shapes, weights summing to 1, finite ATT, CI
ordering, determinism) -- no magic numbers copied from current output.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SHC
from mlsynth.config_models import SHCConfig
from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from mlsynth.estimators import shc as shc_mod
from mlsynth.utils.shc_helpers.structures import SHCResults


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _panel(T0: int = 8, n_post: int = 2, seed: int = 0) -> pd.DataFrame:
    """Small balanced single-unit panel with a smooth-ish trend."""
    T = T0 + n_post
    t = np.arange(1, T + 1)
    rng = np.random.RandomState(seed)
    y = np.linspace(1.0, 2.0, T) + 0.3 * np.sin(t / 2.0) + rng.normal(0.0, 0.02, T)
    return pd.DataFrame(
        {
            "unit": np.ones(T, dtype=int),
            "time": t,
            "y": y,
            "treated": (t > T0).astype(int),
        }
    )


def _base_config(df, **overrides) -> dict:
    cfg = {
        "df": df,
        "outcome": "y",
        "treat": "treated",
        "unitid": "unit",
        "time": "time",
        "m": 4,
        "display_graphs": False,
    }
    cfg.update(overrides)
    return cfg


def test_dict_config_is_coerced_and_fields_propagate():
    df = _panel()
    est = SHC(_base_config(df))
    assert isinstance(est.config, SHCConfig)
    assert est.m == 4
    assert est.outcome == "y"
    assert est.use_augmented is False
    # default counterfactual_color is a list ["red"]
    assert isinstance(est.counterfactual_color, (list, tuple))


def test_invalid_dict_config_pydantic_error_wrapped():
    """A pydantic ValidationError (bad field type) is wrapped as MlsynthConfigError."""
    df = _panel()
    # m must be int; a non-coercible type triggers a pydantic ValidationError,
    # which __init__ wraps into MlsynthConfigError.
    with pytest.raises(MlsynthConfigError, match="Invalid SHC configuration"):
        SHC(_base_config(df, m="not-an-int"))


def test_invalid_dict_config_validator_error_propagates():
    """The model_validator raises MlsynthConfigError directly (not wrapped)."""
    df = _panel()
    with pytest.raises(MlsynthConfigError):
        SHC(_base_config(df, m=0))


def test_config_object_accepted_directly():
    df = _panel()
    cfg = SHCConfig(**_base_config(df))
    est = SHC(cfg)
    assert est.config is cfg


def test_fit_plain_structure_and_determinism():
    df = _panel()
    res1 = SHC(_base_config(df)).fit()
    res2 = SHC(_base_config(df)).fit()

    assert isinstance(res1, SHCResults)
    # weights live on a simplex
    w = res1.design.weights
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-8)
    # counterfactual window spans m + n
    assert res1.counterfactual.shape[0] == res1.inputs.m + res1.inputs.n
    assert res1.observed.shape == res1.counterfactual.shape
    assert np.isfinite(res1.att)
    assert np.isfinite(res1.att_percent)
    # gap == observed - counterfactual
    np.testing.assert_allclose(res1.gap, res1.observed - res1.counterfactual)
    # metadata sanity
    assert res1.metadata["m"] == 4
    assert res1.metadata["use_augmented"] is False
    assert res1.metadata["best_lambda"] is None
    # determinism
    np.testing.assert_allclose(res1.counterfactual, res2.counterfactual)
    assert res1.att == res2.att


def test_fit_augmented_path_sets_lambda():
    df = _panel()
    res = SHC(_base_config(df, use_augmented=True, bandwidth_grid=[0.5, 1.0, 2.0])).fit()
    assert res.design.use_augmented is True
    assert res.design.best_lambda is not None
    assert np.isfinite(res.design.best_lambda)
    assert res.metadata["use_augmented"] is True
    # augmented weights still sum to 1 (sum-to-one constraint retained)
    assert np.isclose(res.design.weights.sum(), 1.0)


def test_fit_inference_structure_and_ci_ordering():
    df = _panel()
    res = SHC(_base_config(df)).fit()
    inf = res.inference
    assert inf is not None
    assert inf.method == "conformal_permutation"
    assert 0.0 <= inf.p_value <= 1.0
    # conformal band ordering where finite
    lo = np.asarray(inf.conformal_lower, dtype=float)
    hi = np.asarray(inf.conformal_upper, dtype=float)
    finite = np.isfinite(lo) & np.isfinite(hi)
    assert finite.any()
    assert np.all(lo[finite] <= hi[finite])
    # critical values increase as level shrinks (tighter level -> larger cv)
    cvs = inf.critical_values
    assert cvs[0.01] >= cvs[0.05] >= cvs[0.10]


def test_fit_with_custom_bandwidth_grid():
    df = _panel()
    res = SHC(_base_config(df, bandwidth_grid=[0.5, 1.0, 1.5, 2.0])).fit()
    assert 0.5 <= res.design.bandwidth <= 2.0


def test_display_graphs_runs_under_agg(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = _panel()
    res = SHC(_base_config(df, display_graphs=True)).fit()
    assert isinstance(res, SHCResults)
    # no files written to the working dir
    assert list(tmp_path.iterdir()) == []


def test_display_graphs_scalar_counterfactual_color(monkeypatch, tmp_path):
    """Scalar (non-list) counterfactual_color skips the list-unpack branch.

    SHCConfig validates the color as a List, so we mutate the estimator
    attribute directly to reach the scalar branch in fit().
    """
    monkeypatch.chdir(tmp_path)
    df = _panel()
    captured = {}

    def fake_plot(results, *, treated_color, counterfactual_color):
        captured["cf"] = counterfactual_color

    monkeypatch.setattr(shc_mod, "plot_shc", fake_plot)
    est = SHC(_base_config(df, display_graphs=True))
    est.counterfactual_color = "blue"  # scalar, not a list
    est.fit()
    assert captured["cf"] == "blue"


def test_display_graphs_empty_color_list_falls_back(monkeypatch, tmp_path):
    """Empty counterfactual_color list -> falls back to 'red' (the `else` branch)."""
    monkeypatch.chdir(tmp_path)
    df = _panel()
    captured = {}

    def fake_plot(results, *, treated_color, counterfactual_color):
        captured["cf"] = counterfactual_color

    monkeypatch.setattr(shc_mod, "plot_shc", fake_plot)
    SHC(_base_config(df, display_graphs=True, counterfactual_color=[])).fit()
    assert captured["cf"] == "red"


def test_plotting_generic_exception_warns_not_raises(monkeypatch):
    df = _panel()

    def boom(*args, **kwargs):
        raise RuntimeError("plot exploded")

    monkeypatch.setattr(shc_mod, "plot_shc", boom)
    with pytest.warns(UserWarning, match="SHC plotting failed"):
        res = SHC(_base_config(df, display_graphs=True)).fit()
    assert isinstance(res, SHCResults)


def test_plotting_plotting_error_propagates(monkeypatch):
    df = _panel()

    def boom(*args, **kwargs):
        raise MlsynthPlottingError("bad plot")

    monkeypatch.setattr(shc_mod, "plot_shc", boom)
    with pytest.raises(MlsynthPlottingError):
        SHC(_base_config(df, display_graphs=True)).fit()


def test_setup_data_error_propagates():
    """M too large for the pre-period -> MlsynthDataError from setup, re-raised."""
    df = _panel(T0=8, n_post=2)
    with pytest.raises(MlsynthDataError):
        SHC(_base_config(df, m=8)).fit()


def test_setup_generic_exception_wrapped_as_data_error(monkeypatch):
    df = _panel()

    def boom(**kwargs):
        raise RuntimeError("setup boom")

    monkeypatch.setattr(shc_mod, "prepare_shc_inputs", boom)
    with pytest.raises(MlsynthDataError, match="Error preparing SHC inputs"):
        SHC(_base_config(df)).fit()


def test_estimation_known_error_propagates(monkeypatch):
    df = _panel()

    def boom(*args, **kwargs):
        raise MlsynthEstimationError("solve failed")

    monkeypatch.setattr(shc_mod, "solve_shc", boom)
    with pytest.raises(MlsynthEstimationError, match="solve failed"):
        SHC(_base_config(df)).fit()


def test_estimation_generic_exception_wrapped(monkeypatch):
    df = _panel()

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(shc_mod, "solve_shc", boom)
    with pytest.raises(MlsynthEstimationError, match="SHC estimation failed"):
        SHC(_base_config(df)).fit()
