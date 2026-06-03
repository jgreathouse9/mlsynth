"""Branch coverage for mlsynth.utils.musc_helpers.plotter.plot_musc."""

import matplotlib

matplotlib.use("Agg")

import builtins
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import MUSC
from mlsynth.exceptions import MlsynthPlottingError
from mlsynth.utils.musc_helpers import SC as SC_NAME
from mlsynth.utils.musc_helpers.plotter import plot_musc


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _panel(N=8, T_pre=14, T_post=3, seed=0):
    rng = np.random.default_rng(seed)
    T = T_pre + T_post
    f = np.zeros(T)
    eta = rng.normal(size=T)
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + eta[t]
    lam = rng.normal(1.0, 0.3, size=N)
    mu = rng.normal(0.0, 0.5, size=N)
    eps = rng.normal(0.0, 1.0, size=(T, N))
    Y = mu[None, :] + f[:, None] * lam[None, :] + eps
    rows = []
    for j in range(N):
        for t in range(T):
            rows.append({"unit": f"u{j:02d}", "time": t, "y": float(Y[t, j]),
                         "treat": int(j == 0 and t >= T_pre)})
    return pd.DataFrame(rows)


def _fit():
    return MUSC({"df": _panel(), "outcome": "y", "treat": "treat",
                 "unitid": "unit", "time": "time",
                 "display_graphs": False, "run_inference": True}).fit()


def test_default_with_sc_baseline():
    res = _fit()
    assert SC_NAME in res.fits
    plot_musc(res)


def test_without_sc_baseline():
    res = _fit()
    plot_musc(res, show_sc_baseline=False, treated_color="navy",
              counterfactual_color="green", outcome="Y", time="T")


def test_sc_absent_but_baseline_requested():
    # show_sc_baseline True but SC not in fits: the SC branch is skipped.
    res = _fit()
    fits = {k: v for k, v in res.fits.items() if k != SC_NAME}
    res2 = dataclasses.replace(res, fits=fits)
    plot_musc(res2, show_sc_baseline=True)


def test_save_str(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit()
    plot_musc(res, save="m.png")
    assert (tmp_path / "m.png").exists()


def test_save_true_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = _fit()
    plot_musc(res, save=True)
    assert (tmp_path / "musc_plot.png").exists()


def test_save_dict_kwargs(tmp_path):
    res = _fit()
    out = tmp_path / "kw.png"
    plot_musc(res, save={"fname": str(out), "dpi": 80})
    assert out.exists()


def test_matplotlib_import_error_raises(monkeypatch):
    # Simulate matplotlib being unavailable: plot_musc imports pyplot inside
    # the function, so a failing import hits the MlsynthPlottingError branch.
    res = _fit()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot" or name.startswith("matplotlib.pyplot"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MlsynthPlottingError):
        plot_musc(res)
