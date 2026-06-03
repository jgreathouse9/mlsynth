"""Coverage tests for mlsynth/utils/proximal_helpers/plotter.py."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.proximal_helpers import plotter as prox_plotter
from mlsynth.utils.proximal_helpers.plotter import plot_proximal
from mlsynth.utils.proximal_helpers.structures import (
    PROXIMALInputs,
    PROXIMALResults,
    ProximalMethodFit,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _make_fit(name, T=6):
    cf = np.linspace(1.0, 2.0, T)
    gap = np.linspace(-0.5, 0.5, T)
    return ProximalMethodFit(
        name=name,
        counterfactual=cf,
        gap=gap,
        time_varying_effect=gap,
        att=float(gap[-1]),
        att_se=0.1,
        pre_rmse=0.05,
        post_rmse=0.06,
        alpha_weights=np.array([0.5, 0.5]),
        donor_weights={"d0": 0.5, "d1": 0.5},
    )


def _make_inputs(T=6, T0=3, time_labels=None):
    if time_labels is None:
        time_labels = np.arange(T)
    return PROXIMALInputs(
        y=np.linspace(1.0, 3.0, T),
        donor_outcomes=np.random.RandomState(0).randn(T, 2),
        donor_proxies=np.random.RandomState(1).randn(T, 2),
        surrogate_outcomes=None,
        surrogate_proxies=None,
        T=T,
        T0=T0,
        bandwidth=1,
        time_labels=time_labels,
        treated_unit_name="treated",
        donor_names=["d0", "d1"],
    )


def _make_results(methods, inputs=None):
    if inputs is None:
        inputs = _make_inputs()
    kw = dict(pi=None, pis=None, pipost=None)
    if "PI" in methods:
        kw["pi"] = _make_fit("PI", inputs.T)
    if "PIS" in methods:
        kw["pis"] = _make_fit("PIS", inputs.T)
    if "PIPost" in methods:
        kw["pipost"] = _make_fit("PIPost", inputs.T)
    return PROXIMALResults(inputs=inputs, **kw)


def test_plot_proximal_known_styles():
    """All three known method styles + axvline branch reachable."""
    plot_proximal(_make_results(["PI", "PIS", "PIPost"]))


def test_plot_proximal_unknown_method_style():
    """A method name not in _STYLES uses the default fallback style."""
    inputs = _make_inputs()
    res = PROXIMALResults(
        inputs=inputs,
        pi=None,
        pis=None,
        pipost=None,
        spsc=_make_fit("SPSC", inputs.T),
    )
    plot_proximal(res)


def test_plot_proximal_mismatched_time_labels():
    """time_labels size != T falls back to arange and skips the axvline."""
    # Wrong-length labels trigger the `t = np.arange(T)` branch.
    inputs = _make_inputs(T=6, T0=3, time_labels=np.arange(3))
    plot_proximal(_make_results(["PI"], inputs=inputs))


def test_plot_proximal_t0_out_of_range():
    """T0-1 out of range skips both axvline calls."""
    inputs = _make_inputs(T=6, T0=0)  # T0-1 = -1 -> condition false
    plot_proximal(_make_results(["PI"], inputs=inputs))


def test_plot_proximal_show_raises(monkeypatch):
    """plt.show raising is wrapped in MlsynthPlottingError."""

    def _boom():
        raise RuntimeError("backend down")

    monkeypatch.setattr(prox_plotter.plt, "show", _boom)
    with pytest.raises(MlsynthPlottingError, match="PROXIMAL plotting failed"):
        plot_proximal(_make_results(["PI"]))
