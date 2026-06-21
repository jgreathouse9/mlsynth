"""Coverage tests for mlsynth.utils.sdid_helpers.plotter.plot_sdid.

The SDID display plot follows one rule: a single treated cohort gets an
observed-versus-counterfactual chart in the shared in-house style (like every
other single-treated-unit estimator); a staggered design (several cohorts) keeps
the pooled event-study chart, which is the only sensible aggregate view there.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import SDID
from mlsynth.utils.sdid_helpers.plotter import plot_sdid


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _staggered_panel(seed=0):
    """Three treated units adopting at two different times -> two cohorts."""
    rng = np.random.default_rng(seed)
    states = [f"s{i}" for i in range(8)]
    T = 20
    records = []
    for i, s in enumerate(states):
        base = rng.standard_normal() * 5 + 50
        for t in range(T):
            outcome = base + 0.5 * t + rng.standard_normal()
            if i in (0, 1):
                treated = 1 if t >= 10 else 0
                if t >= 10:
                    outcome -= 5.0
            elif i == 2:
                treated = 1 if t >= 12 else 0
                if t >= 12:
                    outcome -= 5.0
            else:
                treated = 0
            records.append({"state": s, "year": 2000 + t, "y": float(outcome),
                            "treated": treated})
    return pd.DataFrame(records)


def _single_panel(seed=0):
    """One treated unit -> a single cohort."""
    rng = np.random.default_rng(seed)
    states = [f"s{i}" for i in range(6)]
    T = 20
    records = []
    for i, s in enumerate(states):
        base = rng.standard_normal() * 5 + 50
        for t in range(T):
            outcome = base + 0.5 * t + rng.standard_normal()
            treated = 1 if (i == 0 and t >= 10) else 0
            if i == 0 and t >= 10:
                outcome -= 5.0
            records.append({"state": s, "year": 2000 + t, "y": float(outcome),
                            "treated": treated})
    return pd.DataFrame(records)


def _fit(**kw):
    cfg = {"df": _staggered_panel(), "outcome": "y", "treat": "treated",
           "unitid": "state", "time": "year", "B": 20, "display_graphs": False}
    cfg.update(kw)
    return SDID(cfg).fit()


def _fit_single(**kw):
    cfg = {"df": _single_panel(), "outcome": "y", "treat": "treated",
           "unitid": "state", "time": "year", "B": 20, "display_graphs": False}
    cfg.update(kw)
    return SDID(cfg).fit()


def _capture(monkeypatch):
    captured: dict = {}

    def fake_show():
        fig = plt.gcf()
        axes = fig.axes
        captured["n_axes"] = len(axes)
        captured["titles"] = [a.get_title() for a in axes]
        captured["ylabels"] = [a.get_ylabel() for a in axes]
        captured["legend"] = [
            t.get_text()
            for a in axes
            if a.get_legend() is not None
            for t in a.get_legend().get_texts()
        ]

    monkeypatch.setattr(plt, "show", fake_show)
    return captured


def test_single_cohort_is_observed_vs_fitted(monkeypatch, tmp_path):
    """One treated unit -> single-panel observed-vs-counterfactual, not an
    event study."""
    monkeypatch.chdir(tmp_path)
    cap = _capture(monkeypatch)
    plot_sdid(_fit_single())
    assert cap["n_axes"] == 1
    assert cap["ylabels"] == ["y"]
    assert any("counterfactual" in lab.lower() for lab in cap["legend"])
    assert not any("event study" in t.lower() for t in cap["titles"])
    assert not any("treatment effect" in y.lower() for y in cap["ylabels"])


def test_multiple_cohorts_keep_event_study(monkeypatch, tmp_path):
    """Staggered adoption -> the pooled event-study chart."""
    monkeypatch.chdir(tmp_path)
    cap = _capture(monkeypatch)
    plot_sdid(_fit())
    assert (any("event study" in t.lower() for t in cap["titles"])
            or any("treatment effect" in y.lower() for y in cap["ylabels"]))


def test_single_cohort_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _fit_single(display_graphs=True)
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_sdid_direct(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sdid(_fit())
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())


def test_plot_sdid_with_title(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plot_sdid(_fit(), title="custom title")


def test_fit_display_graphs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    SDID({"df": _staggered_panel(), "outcome": "y", "treat": "treated",
          "unitid": "state", "time": "year", "B": 20,
          "display_graphs": True}).fit()
    assert not any(p.suffix == ".png" for p in tmp_path.iterdir())
