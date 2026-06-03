"""Coverage tests for mlsynth.utils.laxscm_helpers.plotter.plot_rescm."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth.estimators.laxscm import RESCM
from mlsynth.utils.laxscm_helpers import (
    assemble_rescm_results,
    prepare_rescm_inputs,
    run_rescm,
)
from mlsynth.utils.laxscm_helpers.plotter import plot_rescm


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    N, T, T0, r, EFFECT = 16, 40, 30, 3, -3.0
    f = np.cumsum(rng.normal(size=(T, r)), axis=0)
    rows = []
    for i in range(N):
        load = rng.normal(size=r)
        a = rng.normal()
        for t in range(T):
            y = a + load @ f[t] + rng.normal(scale=0.3)
            treat = int(i == 0 and t >= T0)
            if treat:
                y += EFFECT
            rows.append({"unit": f"u{i}", "time": t, "y": y, "treat": treat})
    return pd.DataFrame(rows)


def _results(methods=("SC",)):
    df = _panel()
    inputs = prepare_rescm_inputs(df, unitid="unit", time="time",
                                  outcome="y", treat="treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = run_rescm(inputs, list(methods))
    return assemble_rescm_results(inputs, fits, selected_variant=methods[0])


def test_plot_show_single_method(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_rescm(_results(("SC",)), outcome="y", time="time", save=False)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_multi_method_list_colors(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = _results(("SC", "LINF", "RELAX_L2"))
    plot_rescm(res, outcome="y", time="time",
               counterfactual_color=["red", "blue"], save=False)


def test_plot_save_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_rescm(_results(("SC",)), outcome="y", time="time", save=True)
    assert (tmp_path / "rescm_estimates.png").exists()


def test_plot_save_string(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "rescm_custom.png"
    plot_rescm(_results(("SC",)), outcome="y", time="time", save=str(out))
    assert out.exists()


def test_plot_failure_warns():
    """A malformed results object hits the except->warning branch."""

    class Bad:
        inputs = None  # accessing .inputs.time_index raises inside the plotter

    with pytest.warns(UserWarning, match="RESCM plotting failed"):
        plot_rescm(Bad(), outcome="y", time="time")


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RESCM({"df": _panel(), "outcome": "y", "treat": "treat",
               "unitid": "unit", "time": "time",
               "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
