"""Coverage tests for mlsynth.utils.ppscm_helpers.plotter.plot_ppscm."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import PPSCM
from mlsynth.utils.ppscm_helpers.plotter import plot_ppscm


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _staggered_panel(seed=0, adoption_offsets=(10, 15, 20), N_donors=8,
                     T=30, true_effect=-3.0, noise=0.4):
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 2))
    loadings_donors = rng.standard_normal((N_donors, 2)) * 0.5
    loadings_treated = loadings_donors.mean(axis=0)
    records = []
    for j, T_j in enumerate(adoption_offsets):
        base_load = loadings_treated + 0.1 * rng.standard_normal(2)
        series = factors @ base_load + rng.standard_normal(T) * noise
        series[T_j:] += true_effect
        for t in range(T):
            records.append({"unit": f"treated_{j}", "year": 2000 + t,
                            "y": float(series[t]), "tr": int(t >= T_j)})
    for dd in range(N_donors):
        series = factors @ loadings_donors[dd] + rng.standard_normal(T) * noise
        for t in range(T):
            records.append({"unit": f"d_{dd}", "year": 2000 + t,
                            "y": float(series[t]), "tr": 0})
    return pd.DataFrame(records)


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="tr", unitid="unit", time="year",
                display_graphs=False, run_inference=True)
    base.update(kw)
    return base


def _fit():
    return PPSCM(_cfg(_staggered_panel())).fit()


def test_plot_no_save(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_ppscm(_fit(), save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_save_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_ppscm(_fit(), save=True, title="custom title")
    assert (tmp_path / "ppscm_event_study.png").exists()


def test_plot_save_string(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "ev.png"
    plot_ppscm(_fit(), save=str(out))
    assert out.exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    PPSCM(_cfg(_staggered_panel(), display_graphs=True)).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
