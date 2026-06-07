"""Smoke tests for the standardized plotting foundation (step 1).

Covers the PlotConfig back-compat resolution, EffectResult.plot() end-to-end on
FDID (observed-vs-counterfactual + gap, config-driven cosmetics), and the
event-study archetype exercised on real SDID output.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

from pathlib import Path

import pandas as pd
import pytest

from mlsynth import FDID, SDID
from mlsynth.config_models import BaseEstimatorConfig, PlotConfig
from mlsynth.utils.plotting import Plotter

_BASE = Path(__file__).resolve().parents[2] / "basedata"


# --- PlotConfig resolution / back-compat -----------------------------------

def _hk():
    return pd.read_csv(_BASE / "HongKong.csv")


def test_resolved_plot_folds_legacy_fields():
    cfg = BaseEstimatorConfig(
        df=_hk(), outcome="GDP", treat="Integration", unitid="Country",
        time="Time", treated_color="navy", counterfactual_color=["crimson"],
    )
    pc = cfg.resolved_plot()
    assert pc.observed_color == "navy"
    assert pc.counterfactual_colors == ["crimson"]


def test_nested_plot_config_wins():
    cfg = BaseEstimatorConfig(
        df=_hk(), outcome="GDP", treat="Integration", unitid="Country",
        time="Time", plot=PlotConfig(observed_color="green", title="custom"),
    )
    pc = cfg.resolved_plot()
    assert pc.observed_color == "green"
    assert pc.title == "custom"


# --- FDID end-to-end through EffectResult.plot() ----------------------------

@pytest.fixture(scope="module")
def fdid_res():
    return FDID({
        "df": _hk(), "outcome": "GDP", "treat": "Integration",
        "unitid": "Country", "time": "Time", "display_graphs": False,
    }).fit()


def test_fdid_plot_counterfactual(fdid_res):
    ax = fdid_res.plot(display=False)
    # observed + counterfactual + intervention reference line
    assert len(ax.lines) >= 3
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "GDP"


def test_fdid_plot_respects_cosmetic_override(fdid_res):
    ax = fdid_res.plot(observed_color="navy", display=False)
    assert ax.lines[0].get_color() == "navy"


def test_fdid_plot_gap(fdid_res):
    ax = fdid_res.plot(kind="gap", display=False)
    assert len(ax.lines) >= 1


# --- event-study archetype on real SDID output -----------------------------

def test_event_study_archetype_on_sdid():
    df = pd.read_csv(_BASE / "smoking_data.csv")
    df["Proposition 99"] = df["Proposition 99"].astype(int)
    res = SDID({
        "df": df, "outcome": "cigsale", "treat": "Proposition 99",
        "unitid": "state", "time": "year", "B": 0, "display_graphs": False,
    }).fit()
    es = res.event_study
    ax = Plotter().event_study(
        es.event_times, es.tau,
        ci_lower=es.ci[:, 0], ci_upper=es.ci[:, 1],
    )
    assert len(ax.lines) >= 1  # effect line + reference lines
