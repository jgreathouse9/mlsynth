"""Coverage tests for mlsynth.utils.mlsc_helpers.plotter.plot_mlsc."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import MLSC
from mlsynth.config_models import MLSCConfig
from mlsynth.utils.mlsc_helpers.plotter import plot_mlsc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _make_panel(seed=0, S=5, C=4, T=24, T0=18, treated=0, rank=2,
                noise_scale=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    beta_t = rng.standard_normal((T, rank)) * 0.3
    alpha_s = rng.standard_normal((S, rank)) * 1.5
    eta_sc = rng.standard_normal((S, C, rank)) * 0.4
    mu_sc = alpha_s[:, None, :] + eta_sc
    eps = rng.standard_normal((S, C, T)) * noise_scale
    Y_sct = np.einsum("scf,tf->sct", mu_sc, beta_t) + eps
    Y_st = Y_sct.mean(axis=1)
    labels = [f"state_{s}" for s in range(S)]
    agg = [{"state": labels[s], "year": 2000 + t, "y": float(Y_st[s, t]),
            "treated": 1 if (s == treated and t >= T0) else 0}
           for s in range(S) for t in range(T)]
    dis = []
    for s in range(S):
        for c in range(C):
            for t in range(T):
                dis.append({"county": f"county_{s}_{c}", "state": labels[s],
                            "year": 2000 + t, "y": float(Y_sct[s, c, t]),
                            "treated": 1 if (s == treated and t >= T0) else 0,
                            "pop": 1.0})
    return pd.DataFrame(agg), pd.DataFrame(dis)


def _cfg(df_agg, df_disagg, **overrides):
    cfg = dict(df_agg=df_agg, df_disagg=df_disagg, outcome="y", time="year",
               treat="treated", unitid_agg="state", unitid_disagg="county",
               agg_id="state", weight_col=None, lambda_est="heuristic",
               display_graphs=False)
    cfg.update(overrides)
    return cfg


def _fit():
    a, d = _make_panel()
    return MLSC(MLSCConfig(**_cfg(a, d))).fit()


def test_plot_str_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_mlsc(_fit(), counterfactual_color="red", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_list_color(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_mlsc(_fit(), counterfactual_color=["green"], save=False)


def test_plot_save_dict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_mlsc(_fit(), save={"filename": "mlsc_plot", "directory": str(tmp_path)})
    assert (tmp_path / "mlsc_plot.png").exists()


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    a, d = _make_panel()
    MLSC(MLSCConfig(**_cfg(a, d, display_graphs=True))).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
