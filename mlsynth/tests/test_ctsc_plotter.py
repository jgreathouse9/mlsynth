"""Coverage tests for mlsynth/utils/ctsc_helpers/plotter.py."""

import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import pytest

from mlsynth.utils.ctsc_helpers.plotter import plot_ctsc
from mlsynth.utils.ctsc_helpers.structures import (
    CTSCInference,
    CTSCInputs,
    CTSCResults,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _make_inputs(n=5, T=4, K=2, treatment_names=None):
    if treatment_names is None:
        treatment_names = [f"tx{k}" for k in range(K)]
    return CTSCInputs(
        Y=np.random.RandomState(0).randn(n, T),
        D=np.random.RandomState(1).randn(n, T, K),
        unit_names=[f"u{i}" for i in range(n)],
        time_labels=np.arange(T),
        treatment_names=treatment_names,
        population_weights=np.full(n, 1.0 / n),
    )


def _make_results(K=2, inference=None, treatment_names=None):
    n = 5
    inputs = _make_inputs(n=n, K=K, treatment_names=treatment_names)
    return CTSCResults(
        inputs=inputs,
        average_effect=np.array([0.3 * (k + 1) for k in range(K)]),
        unit_effects=np.random.RandomState(2).randn(n, K),
        unit_weight_matrix=np.zeros((n, n)),
        fit_metric=np.ones(n),
        objective=1.23,
        inference=inference,
    )


def test_plot_ctsc_show_no_inference():
    """Save=False, no inference -> show branch, no CI band, no p-value."""
    plot_ctsc(_make_results(), save=False)


def test_plot_ctsc_with_inference(monkeypatch, tmp_path):
    """Finite se and p-value -> CI band drawn and p-value in title."""
    monkeypatch.chdir(tmp_path)
    K = 2
    inf = CTSCInference(
        method="sign_flip_wald",
        null_value=np.zeros(K),
        wald_stat=np.array([2.0, 3.0]),
        p_value=np.array([0.04, 0.01]),
        se=np.array([0.1, 0.2]),
        n_draws=100,
    )
    plot_ctsc(_make_results(K=K, inference=inf), save=True)
    assert (tmp_path / "CTSC_unit_effects.png").exists()


def test_plot_ctsc_inference_nonfinite(monkeypatch, tmp_path):
    """Non-finite se/p_value skip the CI band and p-value annotation."""
    monkeypatch.chdir(tmp_path)
    K = 2
    inf = CTSCInference(
        method="sign_flip_wald",
        null_value=np.zeros(K),
        wald_stat=np.array([np.nan, np.nan]),
        p_value=np.array([np.nan, np.nan]),
        se=np.array([np.nan, np.nan]),
        n_draws=100,
    )
    plot_ctsc(_make_results(K=K, inference=inf), save=str(tmp_path / "x.png"))
    assert (tmp_path / "x.png").exists()


def test_plot_ctsc_save_no_ext(monkeypatch, tmp_path):
    """String save without extension appends .png."""
    monkeypatch.chdir(tmp_path)
    plot_ctsc(_make_results(K=1), save="effects")
    assert (tmp_path / "effects.png").exists()


def test_plot_ctsc_fewer_names_than_K():
    """When K exceeds the number of names, fall back to 'variable {k}' title."""
    # Only one name supplied but K=2 -> second panel uses fallback title.
    plot_ctsc(_make_results(K=2, treatment_names=["only_one"]), save=False)
