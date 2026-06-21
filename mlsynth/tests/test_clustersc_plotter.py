"""Coverage tests for mlsynth.utils.clustersc_helpers.plotter.plot_clustersc.

The CLUSTERSC display plot is a single-panel observed-vs-counterfactual chart
rendered through the shared in-house ``Plotter`` (``mlsynth.utils.plotting``),
matching the style every other estimator uses. It deliberately does *not* draw a
separate gap / treatment-effect panel.
"""

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


def _capture(monkeypatch):
    """Snapshot the live figure at ``plt.show()`` time (before it is closed)."""
    captured: dict = {}

    def fake_show():
        fig = plt.gcf()
        axes = fig.axes
        captured["n_axes"] = len(axes)
        captured["ylabels"] = [a.get_ylabel().lower() for a in axes]
        captured["ylabels_raw"] = [a.get_ylabel() for a in axes]
        captured["xlabels_raw"] = [a.get_xlabel() for a in axes]
        captured["titles"] = [a.get_title() for a in axes]
        captured["legend_labels"] = [
            t.get_text()
            for a in axes
            if a.get_legend() is not None
            for t in a.get_legend().get_texts()
        ]

    monkeypatch.setattr(plt, "show", fake_show)
    return captured


def test_single_panel_no_gap(monkeypatch, tmp_path):
    """The plot is one panel; there is no separate gap / effect panel."""
    monkeypatch.chdir(tmp_path)
    captured = _capture(monkeypatch)
    plot_clustersc(_fit("pcr"))
    assert captured["n_axes"] == 1
    # No axis is a gap/treatment-effect panel.
    assert not any("treatment effect" in y or "gap" in y
                   for y in captured["ylabels"])
    assert not any("gap" in t.lower() for t in captured["titles"])


def test_pcr_counterfactual_labelled(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = _capture(monkeypatch)
    plot_clustersc(_fit("pcr"))
    assert any("PCR" in lab for lab in captured["legend_labels"])


def test_both_methods_overlay_single_panel(monkeypatch, tmp_path):
    """Both counterfactuals are overlaid on the one panel; nothing is saved."""
    monkeypatch.chdir(tmp_path)
    captured = _capture(monkeypatch)
    plot_clustersc(_fit("both", primary="pcr"))
    assert captured["n_axes"] == 1
    assert any("PCR" in lab for lab in captured["legend_labels"])
    assert any("RPCA" in lab for lab in captured["legend_labels"])
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_axis_labels_use_column_names(monkeypatch, tmp_path):
    """Axes are labelled with the outcome / time column names, like VanillaSC,
    not generic placeholders."""
    monkeypatch.chdir(tmp_path)
    captured = _capture(monkeypatch)
    plot_clustersc(_fit("pcr"))
    assert captured["ylabels_raw"] == ["y"]
    assert captured["xlabels_raw"] == ["time"]


def test_rpca_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = _capture(monkeypatch)
    plot_clustersc(_fit("rpca"))
    assert captured["n_axes"] == 1
    assert any("RPCA" in lab for lab in captured["legend_labels"])


def test_time_label_mismatch_fallback(monkeypatch, tmp_path):
    """A time_labels array whose size != T forces the arange fallback."""
    monkeypatch.chdir(tmp_path)
    _capture(monkeypatch)
    res = _fit("both", primary="pcr")
    bad_inputs = dataclasses.replace(res.inputs, time_labels=np.array([1, 2, 3]))
    res2 = res.model_copy(update={"inputs": bad_inputs})
    plot_clustersc(res2)


def test_show_failure_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _boom():
        raise RuntimeError("no display")

    monkeypatch.setattr(plt, "show", _boom)
    with pytest.raises(MlsynthPlottingError):
        plot_clustersc(_fit("pcr"))
