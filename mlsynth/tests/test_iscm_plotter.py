"""Coverage tests for mlsynth/utils/iscm_helpers/plotter.py."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from mlsynth.utils.iscm_helpers.plotter import plot_iscm
from mlsynth.utils.iscm_helpers.structures import (
    ISCMInference,
    ISCMInputs,
    ISCMResults,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _make_inputs(N=4, T=5):
    return ISCMInputs(
        Y=np.random.RandomState(0).randn(N, T),
        D=np.zeros((N, T)),
        T0=3,
        unit_names=[f"u{i}" for i in range(N)],
        time_labels=np.arange(T),
        treated_idx=np.array([0]),
    )


def _make_results(unit_att=None, contribution=None, inference=None, N=4):
    inputs = _make_inputs(N=N)
    if unit_att is None:
        unit_att = np.array([0.5, 0.3, -0.2, 0.1])
    if contribution is None:
        contribution = np.array([0.4, 0.3, 0.2, 0.1])
    return ISCMResults(
        inputs=inputs,
        att=0.25,
        unit_weight_matrix=np.zeros((N, N)),
        fit_metric=np.ones(N),
        unit_att=unit_att,
        contribution=contribution,
        residuals=np.zeros((N, 5)),
        exposure=np.zeros((N, 5)),
        inference=inference,
    )


def test_plot_iscm_show_no_inference():
    """save=False, no inference -> show branch, no CI span."""
    plot_iscm(_make_results(), save=False)


def test_plot_iscm_with_inference(monkeypatch, tmp_path):
    """Finite CI -> the axvspan band branch is taken."""
    monkeypatch.chdir(tmp_path)
    inf = ISCMInference(
        method="ibragimov_muller",
        null_value=0.0,
        t_stat=2.0,
        p_value=0.05,
        se=0.1,
        ci=(0.1, 0.4),
        alpha_level=0.05,
        n_contributing=4,
        n_draws=100,
    )
    plot_iscm(_make_results(inference=inf), save=True)
    assert (tmp_path / "ISCM_unit_effects.png").exists()


def test_plot_iscm_inference_nonfinite_ci():
    """Non-finite CI lower bound skips the axvspan band."""
    inf = ISCMInference(
        method="ibragimov_muller",
        null_value=0.0,
        t_stat=2.0,
        p_value=0.05,
        se=0.1,
        ci=(np.nan, np.nan),
        alpha_level=0.05,
        n_contributing=4,
        n_draws=100,
    )
    plot_iscm(_make_results(inference=inf), save=False)


def test_plot_iscm_save_str_no_ext(monkeypatch, tmp_path):
    """String save without extension appends .png."""
    monkeypatch.chdir(tmp_path)
    plot_iscm(_make_results(), save="iscm_out")
    assert (tmp_path / "iscm_out.png").exists()


def test_plot_iscm_save_str_with_ext(monkeypatch, tmp_path):
    """String save with extension is kept verbatim."""
    monkeypatch.chdir(tmp_path)
    fname = str(tmp_path / "named.png")
    plot_iscm(_make_results(), save=fname, unit_label="State", effect_label="E")
    assert (tmp_path / "named.png").exists()


def test_plot_iscm_no_contributing():
    """No contributing units (all NaN / zero contribution) -> early return."""
    res = _make_results(
        unit_att=np.full(4, np.nan),
        contribution=np.zeros(4),
    )
    assert plot_iscm(res, save=True) is None
