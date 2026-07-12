"""Plotting the pointwise and/or simultaneous SCPI prediction-interval bands.

Both VanillaSC (``inference="scpi"``) and CLUSTERSC (``compute_scpi_pi=True``)
expose ``plot_bands`` in {"pointwise", "simultaneous", "both"} controlling which
SCPI band(s) are shaded on the observed-vs-counterfactual plot. The simultaneous
band is SCPI-only; other inference modes fall back to the pointwise band.

Layered per ``agents/agents_tests.md``:

* Layer 1: the pure band-selection helper resolves the right (interval,
  interval2, label) for each choice, with fallback when no simultaneous band
  exists.
* Layer 2: the Plotter shades one or two bands; CLUSTERSC's plotter draws the
  SCPI band per the choice.
* Layer 3/4: config validates the field; ``.fit()`` renders without error.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import CLUSTERSC, VanillaSC
from mlsynth.config_models import CLUSTERSCConfig, VanillaSCConfig
from mlsynth.utils.plotting import Plotter, select_pi_bands


_PW = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
_SM = (np.array([0.5, 1.5]), np.array([3.5, 4.5]))


# ----------------------------------------------------------------------
# Layer 1: the pure selection helper
# ----------------------------------------------------------------------
def test_select_pointwise():
    iv, iv2, label = select_pi_bands(_PW, _SM, "pointwise")
    assert iv is _PW and iv2 is None
    assert "Prediction" in label


def test_select_simultaneous():
    iv, iv2, label = select_pi_bands(_PW, _SM, "simultaneous")
    assert iv is _SM and iv2 is None
    assert "Simultaneous" in label


def test_select_both():
    iv, iv2, label = select_pi_bands(_PW, _SM, "both")
    assert iv is _PW and iv2 is _SM
    assert "Prediction" in label


def test_select_falls_back_when_no_simultaneous():
    # other inference modes have no simultaneous band -> pointwise regardless
    for choice in ("pointwise", "simultaneous", "both"):
        iv, iv2, label = select_pi_bands(_PW, None, choice)
        assert iv is _PW and iv2 is None


def test_select_none_when_no_bands():
    iv, iv2, label = select_pi_bands(None, None, "both")
    assert iv is None and iv2 is None


# ----------------------------------------------------------------------
# Layer 2: the Plotter shades two bands
# ----------------------------------------------------------------------
def _n_fills(ax):
    from matplotlib.collections import PolyCollection
    return sum(isinstance(c, PolyCollection) for c in ax.collections)


def test_plotter_one_band():
    p = Plotter()
    t = np.arange(2)
    ax = p.observed_vs_counterfactual(t, np.array([2.0, 3.0]), [np.array([2.0, 3.0])],
                                      interval=_PW)
    assert _n_fills(ax) == 1
    plt.close(ax.figure)


def test_plotter_two_bands():
    p = Plotter()
    t = np.arange(2)
    ax = p.observed_vs_counterfactual(t, np.array([2.0, 3.0]), [np.array([2.0, 3.0])],
                                      interval=_PW, interval2=_SM,
                                      interval2_label="Simultaneous interval")
    assert _n_fills(ax) == 2
    labels = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
    plt.close(ax.figure)


# ----------------------------------------------------------------------
# Layer 4: config
# ----------------------------------------------------------------------
def _panel():
    rng = np.random.default_rng(0)
    J, Tpre, Tpost, r = 10, 16, 6, 2
    T = Tpre + Tpost
    F = rng.standard_normal((T, r)); lam = rng.standard_normal((J + 1, r))
    Y = F @ lam.T + rng.standard_normal((T, J + 1)) * 0.4
    Y[Tpre:, 0] += 1.0
    rows = [{"unit": j, "time": t, "y": float(Y[t, j]),
             "D": int(j == 0 and t >= Tpre)}
            for j in range(J + 1) for t in range(T)]
    return pd.DataFrame(rows)


def test_config_defaults_pointwise():
    df = _panel()
    assert VanillaSCConfig(df=df, outcome="y", treat="D", unitid="unit",
                           time="time", display_graphs=False).plot_bands == "pointwise"
    assert CLUSTERSCConfig(df=df, outcome="y", treat="D", unitid="unit",
                           time="time", display_graphs=False).plot_bands == "pointwise"


def test_config_rejects_bad_plot_bands():
    df = _panel()
    with pytest.raises(Exception):
        VanillaSCConfig(df=df, outcome="y", treat="D", unitid="unit", time="time",
                        display_graphs=False, plot_bands="nonsense")


# ----------------------------------------------------------------------
# Layer 3: .fit() renders without error for each choice
# ----------------------------------------------------------------------
def _germany():
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2] / "basedata"
    d = pd.read_csv(base / "scpi_germany.csv")[["country", "year", "gdp"]].dropna()
    d["status"] = ((d.country == "West Germany") & (d.year >= 1991)).astype(int)
    return d


@pytest.mark.parametrize("bands", ["pointwise", "simultaneous", "both"])
def test_vanillasc_scpi_plot_bands_smoke(bands):
    d = _germany()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        VanillaSC({"df": d, "outcome": "gdp", "treat": "status", "unitid": "country",
                   "time": "year", "inference": "scpi", "scpi_sims": 40,
                   "seed": 0, "display_graphs": True, "plot_bands": bands}).fit()
    plt.close("all")


@pytest.mark.parametrize("bands", ["pointwise", "simultaneous", "both"])
def test_clustersc_scpi_plot_bands_smoke(bands):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CLUSTERSC({"df": _panel(), "outcome": "y", "treat": "D", "unitid": "unit",
                   "time": "time", "method": "pcr", "compute_scpi_pi": True,
                   "scpi_constraint": "ridge", "scpi_sims": 40, "random_state": 0,
                   "display_graphs": True, "plot_bands": bands}).fit()
    plt.close("all")


def test_clustersc_plot_draws_scpi_band():
    # plot_clustersc returns the axis; the SCPI band adds a shaded region.
    from mlsynth.utils.clustersc_helpers.plotter import plot_clustersc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CLUSTERSC({"df": _panel(), "outcome": "y", "treat": "D",
                         "unitid": "unit", "time": "time", "method": "pcr",
                         "compute_scpi_pi": True, "scpi_constraint": "ridge",
                         "scpi_sims": 40, "random_state": 0,
                         "display_graphs": False}).fit()
    ax_both = plot_clustersc(res, plot_bands="both")
    assert _n_fills(ax_both) == 2
    plt.close("all")
    ax_pw = plot_clustersc(res, plot_bands="pointwise")
    assert _n_fills(ax_pw) == 1
    plt.close("all")
