"""Coverage tests for mlsynth.utils.hsc_helpers.plotter.plot_hsc."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import HSC
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.hsc_helpers.plotter import plot_hsc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _integ_ar1(T, phi, eps):
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps[t]
    return np.cumsum(x)


def _simulate(seed, N0=12, T0=60, Tpost=6, kappa=2.0, rho_u=0.0):
    rng = np.random.default_rng(seed)
    T = T0 + Tpost
    F = np.column_stack([
        np.cumsum(rng.normal(0, 2, T)),
        _integ_ar1(T, 0.5, rng.normal(0, 2, T)),
        np.array([0.0] + list(np.cumsum(rng.normal(0, 1, T - 1)))),
    ])
    Lam = np.clip(rng.normal(0, 0.5, (N0, 3)), -2, 2)
    S = rng.choice(N0, 8, replace=False)
    lam0 = rng.dirichlet(np.ones(8) * 0.5) @ Lam[S]
    units = np.vstack([lam0, Lam])
    L = units @ F.T
    uc = rng.normal(0, np.sqrt(1 - 0.25 ** 2), T)
    E = np.zeros((N0 + 1, T))
    for j in range(N0 + 1):
        ui = rng.normal(0, np.sqrt(1 - 0.25 ** 2), T)
        U = np.sqrt(rho_u) * uc + np.sqrt(1 - rho_u) * ui
        E[j] = _integ_ar1(T, 0.25, U)
    alpha = np.concatenate([[0.0], rng.uniform(5, 15, N0)])
    Y = (L + kappa * E + rng.normal(0, 1, (N0 + 1, T)) + alpha[:, None]
         + rng.normal(0, 1, T)[None, :])
    return Y, T0


def _to_df(Y, T0):
    rows = []
    for j in range(Y.shape[0]):
        for t in range(Y.shape[1]):
            rows.append({"unit": f"u{j:02d}", "time": t, "y": float(Y[j, t]),
                         "treat": int(j == 0 and t >= T0)})
    return pd.DataFrame(rows)


def _fit():
    Y, T0 = _simulate(0)
    df = _to_df(Y, T0)
    return HSC({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
                "treat": "treat", "display_graphs": False}).fit()


def test_plot_hsc(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_hsc(_fit())
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_missing_counterfactual_raises():
    res = _fit()
    broken = res.model_copy(update={"counterfactual_full": None})
    with pytest.raises(MlsynthPlottingError):
        plot_hsc(broken)


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Y, T0 = _simulate(0)
    df = _to_df(Y, T0)
    HSC({"df": df, "outcome": "y", "unitid": "unit", "time": "time",
         "treat": "treat", "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
