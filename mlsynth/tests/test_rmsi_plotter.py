"""Coverage tests for mlsynth.utils.rmsi_helpers.plotter.plot_rmsi."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mlsynth import RMSI
from mlsynth.utils.rmsi_helpers import simulate_rmsi_panel
from mlsynth.utils.rmsi_helpers.plotter import plot_rmsi


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _fit(n_treated=8):
    df = simulate_rmsi_panel(n_units=40, n_treated=n_treated, T0=20,
                             n_post=11, d_unit=2, d_time=2, att=5.0, seed=0)
    return RMSI({"df": df, "outcome": "Y", "treat": "treated",
                 "unitid": "unit", "time": "time",
                 "unit_covariates": ["x0", "x1"],
                 "time_covariates": ["z0", "z1"],
                 "display_graphs": False}).fit()


def test_multi_treated_str_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_rmsi(_fit(n_treated=8), counterfactual_color="red", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_single_treated_list_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_rmsi(_fit(n_treated=1), counterfactual_color=["green"], save=False)


def test_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_rmsi(_fit(n_treated=1),
              save={"filename": "rmsi_plot", "directory": str(tmp_path)})
    assert (tmp_path / "rmsi_plot.png").exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = simulate_rmsi_panel(n_units=20, n_treated=1, T0=15, n_post=6, seed=3)
    RMSI({"df": df, "outcome": "Y", "treat": "treated", "unitid": "unit",
          "time": "time", "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
