"""Coverage tests for mlsynth.utils.clustersc_helpers.plotter.plot_clustersc."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import CLUSTERSC
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.clustersc_helpers.plotter import plot_clustersc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _factor_panel(J=12, T_pre=14, T_post=6, r=2, tau_true=1.0, seed=0):
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, r))
    lam = rng.standard_normal((J + 1, r))
    eps = rng.standard_normal((T, J + 1)) * 0.4
    Y = F @ lam.T + eps
    Y[T_pre:, 0] += tau_true
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
             "D": int(j == 0 and t >= T_pre)}
            for j in range(J + 1) for t in range(T)]
    return pd.DataFrame(rows)


def _fit(method, **kw):
    cfg = {"df": _factor_panel(), "outcome": "y", "treat": "D",
           "unitid": "unit", "time": "time", "method": method,
           "k_clusters": 1, "display_graphs": False}
    cfg.update(kw)
    return CLUSTERSC(cfg).fit()


def test_both_methods(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_clustersc(_fit("both", primary="pcr"))
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_pcr_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_clustersc(_fit("pcr"))


def test_rpca_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_clustersc(_fit("rpca"))


def test_time_label_mismatch_fallback(monkeypatch, tmp_path):
    """A time_labels array whose size != T forces the arange fallback."""
    monkeypatch.chdir(tmp_path)
    res = _fit("both", primary="pcr")
    bad_inputs = dataclasses.replace(res.inputs, time_labels=np.array([1, 2, 3]))
    res2 = dataclasses.replace(res, inputs=bad_inputs)
    plot_clustersc(res2)


def test_show_failure_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _boom():
        raise RuntimeError("no display")

    monkeypatch.setattr(plt, "show", _boom)
    with pytest.raises(MlsynthPlottingError):
        plot_clustersc(_fit("pcr"))
