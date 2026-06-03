"""Coverage tests for mlsynth/utils/dsc_helpers/plotter.py."""

import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import pytest

from mlsynth.utils.dsc_helpers.plotter import plot_dsc
from mlsynth.utils.dsc_helpers.structures import (
    DSCInputs,
    DSCResults,
    QTECurve,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _make_results(att=0.5):
    q = np.linspace(0.05, 0.95, 10)
    curves = []
    for t in range(2):
        observed = q + t * 0.1
        counterfactual = q * 0.8
        curves.append(
            QTECurve(
                time_label=t,
                quantiles=q,
                observed=observed,
                counterfactual=counterfactual,
                qte=observed - counterfactual,
            )
        )
    avg_qte = np.mean([c.qte for c in curves], axis=0)
    inputs = DSCInputs(
        cell_samples={},
        unit_names=["treated", "d1", "d2"],
        time_labels=np.arange(4),
        T=4,
        T0=2,
        treated_unit_name="treated",
    )
    return DSCResults(
        inputs=inputs,
        donor_weights={"d1": 0.5, "d2": 0.5},
        period_weights={},
        lambda_weights=np.array([0.5, 0.5]),
        qte_curves=curves,
        average_qte=avg_qte,
        att=att,
        pre_period_wasserstein=np.array([0.1, 0.2]),
    )


def test_plot_dsc_show():
    """Cover the plt.show() branch (save=False)."""
    plot_dsc(_make_results(), save=False)


def test_plot_dsc_save_bool(monkeypatch, tmp_path):
    """Save=True uses the default filename and appends .png; lands in tmp."""
    monkeypatch.chdir(tmp_path)
    plot_dsc(_make_results(), save=True)
    assert (tmp_path / "DSC_qte.png").exists()


def test_plot_dsc_save_str_with_ext(monkeypatch, tmp_path):
    """Save as a string with extension keeps the given name."""
    monkeypatch.chdir(tmp_path)
    fname = str(tmp_path / "my_dsc.png")
    plot_dsc(_make_results(), save=fname, outcome_label="Y")
    assert os.path.exists(fname)


def test_plot_dsc_save_str_no_ext(monkeypatch, tmp_path):
    """Save as a string without extension appends .png."""
    monkeypatch.chdir(tmp_path)
    plot_dsc(_make_results(att=-0.25), save="noext")
    assert (tmp_path / "noext.png").exists()


def test_plot_dsc_empty_curves():
    """No QTE curves -> early return, nothing plotted."""
    res = _make_results()
    res = DSCResults(
        inputs=res.inputs,
        donor_weights=res.donor_weights,
        period_weights=res.period_weights,
        lambda_weights=res.lambda_weights,
        qte_curves=[],
        average_qte=res.average_qte,
        att=res.att,
        pre_period_wasserstein=res.pre_period_wasserstein,
    )
    assert plot_dsc(res, save=True) is None
