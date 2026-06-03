"""Coverage tests for mlsynth.utils.msqrt_helpers.plotter.plot_msqrt."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mlsynth import MSQRT
from mlsynth.utils.msqrt_helpers import simulate_msqrt_panel
from mlsynth.utils.msqrt_helpers.plotter import plot_msqrt


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _fit(n_tr=3):
    df = simulate_msqrt_panel(n_treated=n_tr, n_control=8, T0=24,
                              n_post=6, att=2.0, seed=1)
    return MSQRT({"df": df, "outcome": "Y", "treat": "treated",
                  "unitid": "unit", "time": "time",
                  "display_graphs": False}).fit()


def test_multi_treated_str_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_msqrt(_fit(n_tr=3), counterfactual_color="red", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_single_treated_list_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_msqrt(_fit(n_tr=1), counterfactual_color=["green"], save=False)


def test_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_msqrt(_fit(n_tr=1),
               save={"filename": "msqrt_plot", "directory": str(tmp_path)})
    assert (tmp_path / "msqrt_plot.png").exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = simulate_msqrt_panel(n_treated=1, n_control=8, T0=24, n_post=6,
                              att=2.0, seed=1)
    MSQRT({"df": df, "outcome": "Y", "treat": "treated", "unitid": "unit",
           "time": "time", "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
