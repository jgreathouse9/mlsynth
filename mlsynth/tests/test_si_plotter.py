"""Coverage tests for mlsynth.utils.si_helpers.plotter.plot_si."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SI
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.si_helpers.plotter import plot_si

_BASE = dict(outcome="y", unitid="unit", time="time", treat="treat",
             inters=["interA", "interB"], display_graphs=False)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _simulate(seed=0, N=14, T=30, T0=20, rank=2, noise=0.3):
    rng = np.random.default_rng(seed)
    F = np.column_stack([np.cumsum(rng.normal(0, 1, T)) for _ in range(rank)])
    lam = rng.normal(0, 1, (N, rank))
    Y = lam @ F.T + noise * rng.standard_normal((N, T))
    units = [f"u{j:02d}" for j in range(N)]
    rows = []
    for j, u in enumerate(units):
        inter_a = int(1 <= j <= 6)
        inter_b = int(7 <= j <= N - 1)
        for t in range(T):
            rows.append({"unit": u, "time": t, "y": float(Y[j, t]),
                         "treat": int(u == "u00" and t >= T0),
                         "interA": inter_a, "interB": inter_b})
    return pd.DataFrame(rows)


def test_plot_with_ci_label(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = SI({"df": _simulate(), **_BASE}).fit()
    plot_si(res)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_no_ci_label(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = SI({"df": _simulate(), **{**_BASE, "bias_correct": False}}).fit()
    plot_si(res)


def test_plot_empty_arms_raises():
    res = SI({"df": _simulate(), **_BASE}).fit()
    empty = dataclasses.replace(res, arms={})
    with pytest.raises(MlsynthPlottingError):
        plot_si(empty)


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    SI({"df": _simulate(), **{**_BASE, "display_graphs": True}}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
