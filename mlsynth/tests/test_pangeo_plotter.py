"""Coverage tests for mlsynth.utils.pangeo_helpers.plotter.plot_pangeo."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlsynth import PANGEO
from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel
from mlsynth.utils.pangeo_helpers.plotter import plot_pangeo
from mlsynth.utils.pangeo_helpers.structures import PangeoResults


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _design_only():
    df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                   T=120, seed=0)
    return PANGEO({"df": df, "outcome": "sales", "arm": "arm", "unitid": "unit",
                   "time": "time", "max_supergeo_size": 3,
                   "display_graphs": False}).fit()


def _realised():
    df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B", "C"),
                                   T=80, n_post=8, seed=0)
    return PANGEO({"df": df, "outcome": "sales", "arm": "arm", "unitid": "unit",
                   "time": "time", "max_supergeo_size": 3,
                   "post_col": "post_col", "display_graphs": False}).fit()


def test_design_only_show(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_pangeo(_design_only(), save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_realised_with_post(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_pangeo(_realised(), save=False)


def test_save_with_extension(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_pangeo(_design_only(), save="pangeo_out.png")
    assert (tmp_path / "pangeo_out.png").exists()


def test_save_without_extension(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_pangeo(_design_only(), save="pangeo_noext")
    assert (tmp_path / "pangeo_noext.png").exists()


def test_save_default_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_pangeo(_design_only(), save=True)
    assert (tmp_path / "PANGEO_design.png").exists()


def test_empty_designs_returns(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    empty = PangeoResults(arm_designs={}, max_supergeo_size=3, assignment={},
                          time_labels=np.arange(5))
    assert plot_pangeo(empty) is None


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B"),
                                   T=80, seed=0)
    PANGEO({"df": df, "outcome": "sales", "arm": "arm", "unitid": "unit",
            "time": "time", "max_supergeo_size": 3,
            "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
