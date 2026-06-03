"""Coverage tests for mlsynth.utils.sparse_sc_helpers.plotter.plot_sparse_sc."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SparseSC
from mlsynth.utils.sparse_sc_helpers.plotter import plot_sparse_sc

COVS = ["p0", "p1", "p2", "p3"]


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _factor_panel(seed=0, N_donors=6, T=20, T0=14, P=4,
                  true_effect=-2.5, noise=0.3):
    rng = np.random.default_rng(seed)
    n_units = N_donors + 1
    F = rng.standard_normal((T, P))
    Lambda = rng.standard_normal((n_units, P))
    Y = F @ Lambda.T + noise * rng.standard_normal((T, n_units))
    Y[T0:, 0] += true_effect
    records = []
    for u in range(n_units):
        covs = {f"p{p}": float(Lambda[u, p]) for p in range(P)}
        for t in range(T):
            row = {"unit": f"unit_{u}", "year": 2000 + t, "y": float(Y[t, u]),
                   "tr": int(u == 0 and t >= T0)}
            row.update(covs)
            records.append(row)
    return pd.DataFrame(records)


def _fit():
    return SparseSC({
        "df": _factor_panel(), "outcome": "y", "treat": "tr",
        "unitid": "unit", "time": "year", "covariates": COVS,
        "lambda_grid": [0.0, 0.01], "run_inference": False,
        "display_graphs": False,
    }).fit()


def test_plot_str_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sparse_sc(_fit(), counterfactual_color="red", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_list_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sparse_sc(_fit(), counterfactual_color=["green"], save=False)


def test_plot_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sparse_sc(_fit(),
                   save={"filename": "ssc_plot", "directory": str(tmp_path)})
    assert (tmp_path / "ssc_plot.png").exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    SparseSC({
        "df": _factor_panel(), "outcome": "y", "treat": "tr",
        "unitid": "unit", "time": "year", "covariates": COVS,
        "lambda_grid": [0.0, 0.01], "run_inference": False,
        "display_graphs": True,
    }).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
