"""Coverage tests for mlsynth.utils.masc_helpers.plotter.plot_masc."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import MASC
from mlsynth.utils.masc_helpers.plotter import plot_masc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _toy_panel(n_donors: int = 5, T: int = 15, T0: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.normal(size=T))
    rows = []
    units = ["u0"] + [f"u{j}" for j in range(1, n_donors + 1)]
    for u in units:
        for t in range(T):
            if u == "u0":
                y = 0.6 * factor[t] + rng.normal(scale=0.1)
            elif u in ("u1", "u2"):
                y = factor[t] + rng.normal(scale=0.1)
            else:
                y = rng.normal()
            if u == "u0" and t >= T0:
                y += -2.0
            rows.append({"unit": u, "t": t, "y": y, "treat": int(u == "u0" and t >= T0)})
    return pd.DataFrame(rows)


def _results():
    cfg = dict(df=_toy_panel(), outcome="y", treat="treat", unitid="unit",
               time="t", display_graphs=False)
    return MASC(cfg).fit()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = dict(df=_toy_panel(), outcome="y", treat="treat", unitid="unit",
               time="t", display_graphs=True)
    MASC(cfg).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_save_default_filename(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_masc(_results(), outcome="y", time="t", save=True)
    assert (tmp_path / "masc.png").exists()


def test_plot_save_string_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "custom_masc.png"
    plot_masc(_results(), outcome="y", time="t", save=str(out))
    assert out.exists()


def test_plot_no_save_list_colors(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_masc(_results(), outcome="y", time="t",
              counterfactual_color=["blue", "green"], save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
