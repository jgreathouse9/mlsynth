"""Coverage tests for mlsynth.utils.sdid_helpers.plotter.plot_sdid."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.utils.sdid_helpers.plotter import plot_sdid


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _staggered_panel(seed=0):
    rng = np.random.default_rng(seed)
    states = [f"s{i}" for i in range(8)]
    T = 20
    records = []
    for i, s in enumerate(states):
        base = rng.standard_normal() * 5 + 50
        for t in range(T):
            outcome = base + 0.5 * t + rng.standard_normal()
            if i in (0, 1):
                treated = 1 if t >= 10 else 0
                if t >= 10:
                    outcome -= 5.0
            elif i == 2:
                treated = 1 if t >= 12 else 0
                if t >= 12:
                    outcome -= 5.0
            else:
                treated = 0
            records.append({"state": s, "year": 2000 + t, "y": float(outcome),
                            "treated": treated})
    return pd.DataFrame(records)


def _fit():
    return SDID({"df": _staggered_panel(), "outcome": "y", "treat": "treated",
                 "unitid": "state", "time": "year", "B": 20,
                 "display_graphs": False}).fit()


def test_plot_sdid_direct(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sdid(_fit())
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_sdid_with_title(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sdid(_fit(), title="custom title")


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    SDID({"df": _staggered_panel(), "outcome": "y", "treat": "treated",
          "unitid": "state", "time": "year", "B": 20,
          "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
