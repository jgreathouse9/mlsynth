"""Coverage tests for mlsynth/utils/nsc_helpers/plotter.py."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.nsc_helpers import plotter as nsc_plotter
from mlsynth.utils.nsc_helpers.plotter import plot_nsc
from mlsynth.utils.nsc_helpers.structures import (
    NSCDesign,
    NSCInference,
    NSCInputs,
    NSCResults,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _make_inputs(T=6, T0=3, time_labels=None):
    if time_labels is None:
        time_labels = np.arange(T)
    J = 2
    return NSCInputs(
        treated_outcome=np.linspace(1.0, 3.0, T),
        donor_outcomes=np.random.RandomState(0).randn(T, J),
        matching_matrix=np.random.RandomState(1).randn(J, T0),
        treated_matching_vector=np.random.RandomState(2).randn(T0),
        donor_names=np.array(["d0", "d1"]),
        treated_unit_name="treated",
        T=T,
        T0=T0,
        time_labels=time_labels,
    )


def _make_inference(T=6, with_ci=True):
    if with_ci:
        gap_lower = np.full(T, -0.5)
        gap_upper = np.full(T, 0.5)
    else:
        gap_lower = np.asarray([], dtype=float)
        gap_upper = np.asarray([], dtype=float)
    return NSCInference(
        method="doudchenko_imbens",
        alpha=0.1,
        gap=np.linspace(-0.3, 0.3, T),
        gap_lower=gap_lower,
        gap_upper=gap_upper,
        att=0.1,
        att_se=0.05,
    )


def _make_results(inputs=None, with_ci=True):
    if inputs is None:
        inputs = _make_inputs()
    T = inputs.T
    design = NSCDesign(
        w=np.array([0.5, 0.5]),
        donor_weights={"d0": 0.5, "d1": 0.5},
        a_star=0.1,
        b_star=0.2,
        a_scaled=1.0,
        b_scaled=2.0,
        eigvals=np.array([1.0, 2.0]),
    )
    return NSCResults(
        inputs=inputs,
        design=design,
        cv_trace=None,
        inference=_make_inference(T=T, with_ci=with_ci),
        counterfactual=np.linspace(0.9, 2.8, T),
        gap=np.linspace(-0.3, 0.3, T),
        att=0.1,
        pre_rmse=0.05,
    )


def test_plot_nsc_with_ci():
    """CI band present (gap_lower non-empty) + axvline branch."""
    plot_nsc(_make_results(with_ci=True))


def test_plot_nsc_no_ci():
    """Empty gap_lower skips the fill_between CI band."""
    plot_nsc(_make_results(with_ci=False))


def test_plot_nsc_mismatched_time_labels():
    """time_labels size != T falls back to arange; axvline still in range."""
    inputs = _make_inputs(T=6, T0=3, time_labels=np.arange(2))
    plot_nsc(_make_results(inputs=inputs))


def test_plot_nsc_t0_out_of_range():
    """T0-1 = -1 skips both axvline calls."""
    inputs = _make_inputs(T=6, T0=0)
    plot_nsc(_make_results(inputs=inputs))


def test_plot_nsc_show_raises(monkeypatch):
    """plt.show raising is wrapped in MlsynthPlottingError."""

    def _boom():
        raise RuntimeError("no backend")

    monkeypatch.setattr(nsc_plotter.plt, "show", _boom)
    with pytest.raises(MlsynthPlottingError, match="NSC plotting failed"):
        plot_nsc(_make_results())
