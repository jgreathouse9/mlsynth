"""Coverage tests for mlsynth.utils.fma_helpers.plotter.plot_fma."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import FMA
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.fma_helpers.plotter import plot_fma


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _factor_panel(J=20, T_pre=30, T_post=10, r_true=2, tau_true=1.0, seed=0):
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    F = rng.standard_normal((T, r_true)).cumsum(axis=0)
    lam = rng.standard_normal((J + 1, r_true))
    eps = rng.standard_normal((T, J + 1)) * 0.5
    Y = F @ lam.T + eps
    Y[T_pre:, 0] += tau_true
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
             "D": int(j == 0 and t >= T_pre)}
            for j in range(J + 1) for t in range(T)]
    return pd.DataFrame(rows)


def _fit(inference_methods, **kw):
    cfg = {"df": _factor_panel(), "outcome": "y", "treat": "D",
           "unitid": "unit", "time": "time", "n_factors": 2,
           "inference_methods": inference_methods, "display_graphs": False}
    cfg.update(kw)
    return FMA(cfg).fit()


def test_all_bands(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _fit(["asymptotic", "bootstrap", "placebo"], n_bootstrap=100)
    plot_fma(res)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_no_bands(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_fma(_fit([]))


def test_bootstrap_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_fma(_fit(["bootstrap"], n_bootstrap=100))


def test_placebo_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_fma(_fit(["placebo"]))


def test_time_label_mismatch_fallback(monkeypatch, tmp_path):
    """A time_labels array whose size != T forces the arange fallback."""
    monkeypatch.chdir(tmp_path)
    res = _fit([])
    bad_inputs = dataclasses.replace(res.inputs, time_labels=np.array([1, 2, 3]))
    res2 = res.model_copy(update={"inputs": bad_inputs})
    plot_fma(res2)


def test_show_failure_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _boom():
        raise RuntimeError("no display")

    monkeypatch.setattr(plt, "show", _boom)
    with pytest.raises(MlsynthPlottingError):
        plot_fma(_fit([]))
