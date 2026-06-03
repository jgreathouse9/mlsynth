"""Coverage tests for mlsynth.utils.snn_helpers.plotter.plot_snn."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SNN
from mlsynth.utils.snn_helpers.plotter import plot_snn


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _low_rank_panel(n_co=25, n_tr=5, T=40, T0=30, r=3, effect=2.0,
                    noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    N = n_co + n_tr
    U = rng.standard_normal((N, r))
    Vt = rng.standard_normal((r, T))
    A = U @ Vt
    Y = A + rng.standard_normal((N, T)) * noise
    Y[n_co:, T0:] += effect
    rows = [{"unit": f"u{i}", "time": t, "y": Y[i, t],
             "D": int(i >= n_co and t >= T0)}
            for i in range(N) for t in range(T)]
    return pd.DataFrame(rows)


def _fit(n_tr=5):
    df = _low_rank_panel(n_tr=n_tr)
    return SNN({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
                "time": "time", "max_rank": 3, "display_graphs": False}).fit()


def test_multi_treated_str_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_snn(_fit(n_tr=5), counterfactual_color="red", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_single_treated_list_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_snn(_fit(n_tr=1), counterfactual_color=["green"], save=False)


def test_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_snn(_fit(n_tr=1), save={"filename": "snn_plot", "directory": str(tmp_path)})
    assert (tmp_path / "snn_plot.png").exists()


def test_save_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_snn(_fit(n_tr=1), save=True)
    assert any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = _low_rank_panel(n_tr=1)
    SNN({"df": df, "outcome": "y", "treat": "D", "unitid": "unit",
         "time": "time", "max_rank": 3, "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
