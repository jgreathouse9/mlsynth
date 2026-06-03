"""Coverage tests for mlsynth.utils.fdid_helpers.plotter.plot_fdid."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mlsynth import FDID
from mlsynth.config_models import FDIDConfig
from mlsynth.exceptions import MlsynthDataError, MlsynthPlottingError
from mlsynth.utils.fdid_helpers import plotter as fdid_plotter
from mlsynth.utils.fdid_helpers.plotter import plot_fdid


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _data():
    return pd.DataFrame({
        "unit": ["T"] * 6 + ["C1"] * 6 + ["C2"] * 6,
        "time": list(range(1, 7)) * 3,
        "y": [10, 12, 14, 16, 22, 24, 8, 9, 10, 11, 12, 13,
              9, 10, 11, 12, 13, 14],
        "treated_indicator": [0, 0, 0, 0, 1, 1] + [0] * 12,
    })


def _fit():
    cfg = FDIDConfig(df=_data(), unitid="unit", time="time", outcome="y",
                     treat="treated_indicator", display_graphs=False)
    return FDID(config=cfg).fit()


_KW = dict(time="time", unitid="unit", outcome="y", treat="treated_indicator",
           treated_color="black", counterfactual_color=["red", "blue"])


def test_plot_success_list_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_fdid(_fit(), save=False, **_KW)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kw = {**_KW, "counterfactual_color": ["red", "blue"]}
    plot_fdid(_fit(), save={"filename": "fdid_plot", "directory": str(tmp_path)}, **kw)
    assert (tmp_path / "fdid_plot.png").exists()


def test_plotting_error_warns(monkeypatch):
    def _raise(*a, **k):
        raise MlsynthPlottingError("boom")

    monkeypatch.setattr(fdid_plotter, "plot_estimates", _raise)
    with pytest.warns(UserWarning, match="Plotting failed"):
        plot_fdid(_fit(), save=False, **_KW)


def test_data_error_warns(monkeypatch):
    def _raise(*a, **k):
        raise MlsynthDataError("bad data")

    monkeypatch.setattr(fdid_plotter, "plot_estimates", _raise)
    with pytest.warns(UserWarning, match="Plotting failed"):
        plot_fdid(_fit(), save=False, **_KW)


def test_unexpected_error_warns(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(fdid_plotter, "plot_estimates", _raise)
    with pytest.warns(UserWarning, match="Unexpected plotting error"):
        plot_fdid(_fit(), save=False, **_KW)


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = FDIDConfig(df=_data(), unitid="unit", time="time", outcome="y",
                     treat="treated_indicator", display_graphs=True)
    FDID(config=cfg).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
