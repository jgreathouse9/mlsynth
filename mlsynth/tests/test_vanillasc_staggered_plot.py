"""Staggered-adoption VanillaSC honours ``display_graphs`` / ``save``.

The staggered path (``run_vanillasc_staggered``) followed Cattaneo, Feng, Palomba
and Titiunik but had no plotting, so ``display_graphs=True`` was silently a no-op.
These pin scpi-``scplotMulti``-style output -- a per-unit ``series`` facet, a
per-unit ``treatment`` (gap) facet, a per-unit ATT dot plot (``effect="unit"``),
and the event-time aggregate effect (``effect="time"``) -- rendered through
mlsynth's ``Plotter``, with prediction-interval bands shaded when SCPI inference
is requested.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth import VanillaSC


def _staggered_panel(n=10, T=30, adopt=(("u00", 18), ("u01", 22)), seed=0):
    """Factor panel with several treated units adopting at different times."""
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    for t in range(1, T):
        common[t] = 0.6 * common[t - 1] + rng.standard_normal()
    Y = np.zeros((T, n))
    for i in range(n):
        Y[:, i] = 10 + rng.standard_normal() * common + rng.standard_normal(T) * 0.4
    adopt_map = dict(adopt)
    rows = []
    for i in range(n):
        name = f"u{i:02d}"
        for t in range(T):
            tr = int(name in adopt_map and t >= adopt_map[name])
            rows.append({"dma": name, "start_date": t,
                         "total_sales": Y[t, i], "treated": tr})
    return pd.DataFrame(rows)


def _cfg(**extra):
    base = {"df": _staggered_panel(), "outcome": "total_sales", "treat": "treated",
            "unitid": "dma", "time": "start_date", "display_graphs": False}
    base.update(extra)
    return VanillaSC(base).config


@pytest.fixture(scope="module")
def staggered_fit():
    df = _staggered_panel()
    return VanillaSC({"df": df, "outcome": "total_sales", "treat": "treated",
                      "unitid": "dma", "time": "start_date",
                      "display_graphs": False}).fit()


# ── wiring: the staggered path renders when asked, and not otherwise ──────────

def test_staggered_fit_writes_plot_files(tmp_path):
    df = _staggered_panel()
    VanillaSC({"df": df, "outcome": "total_sales", "treat": "treated",
               "unitid": "dma", "time": "start_date",
               "display_graphs": False, "save": str(tmp_path / "vsc")}).fit()
    written = {p.name for p in tmp_path.glob("*.png")}
    # one file per view (scpi scplotMulti ptype x effect coverage)
    assert {"vsc_series.png", "vsc_treatment.png",
            "vsc_att_by_unit.png", "vsc_event_study.png"} <= written


def test_no_files_when_save_and_display_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _staggered_panel()
    VanillaSC({"df": df, "outcome": "total_sales", "treat": "treated",
               "unitid": "dma", "time": "start_date",
               "display_graphs": False}).fit()
    assert not list(tmp_path.glob("*.png"))


# ── the four scpi views ───────────────────────────────────────────────────────

def _plot(staggered_fit):
    from mlsynth.utils.vanillasc_helpers.staggered_plotter import plot_staggered
    return plot_staggered(
        _cfg(),
        staggered_fit.sub_method_results,
        staggered_fit.additional_outputs["event_study"],
        staggered_fit.additional_outputs.get("event_study_intervals"),
    )


def test_all_four_views_returned(staggered_fit):
    figs = _plot(staggered_fit)
    assert len(figs) == 4  # series, treatment, att-by-unit, event-study
    plt.close("all")


def test_series_facets_one_panel_per_unit(staggered_fit):
    n = len(staggered_fit.sub_method_results)
    series = _plot(staggered_fit)[0]
    visible = [ax for ax in series.axes if ax.get_visible()]
    assert len(visible) == n
    for ax in visible:
        assert len(ax.get_lines()) >= 2  # observed + synthetic
    plt.close("all")


def test_treatment_facets_have_gap_and_zero_line(staggered_fit):
    n = len(staggered_fit.sub_method_results)
    treatment = _plot(staggered_fit)[1]
    visible = [ax for ax in treatment.axes if ax.get_visible()]
    assert len(visible) == n
    for ax in visible:                       # gap line + a horizontal zero line
        ys = [tuple(np.round(ln.get_ydata(), 9)) for ln in ax.get_lines()]
        assert any(set(y) == {0.0} for y in ys), "missing zero reference line"
    plt.close("all")


def test_att_by_unit_has_point_per_unit(staggered_fit):
    n = len(staggered_fit.sub_method_results)
    att_fig = _plot(staggered_fit)[2]
    ax = att_fig.axes[0]
    assert len(ax.get_yticklabels()) == n
    plt.close("all")


def test_event_study_plots_effect_path(staggered_fit):
    event = _plot(staggered_fit)[3]
    assert any(ln.get_ydata().size for ln in event.axes[0].get_lines())
    plt.close("all")


def test_single_unit_facet_degrades_to_one_panel(staggered_fit):
    from mlsynth.utils.vanillasc_helpers.staggered_plotter import plot_staggered
    one_name = next(iter(staggered_fit.sub_method_results))
    one = {one_name: staggered_fit.sub_method_results[one_name]}
    figs = plot_staggered(_cfg(), one, {1: 0.1, 2: 0.2})
    assert len([ax for ax in figs[0].axes if ax.get_visible()]) == 1
    plt.close("all")


# ── prediction intervals appear only under SCPI inference ─────────────────────

def test_no_bands_without_inference(staggered_fit):
    # Default (no inference) -> no shaded PI regions on the series facets.
    series = _plot(staggered_fit)[0]
    assert not any(ax.collections for ax in series.axes)
    plt.close("all")


def test_scpi_inference_shades_bands():
    from mlsynth.utils.vanillasc_helpers.staggered_plotter import plot_staggered
    df = _staggered_panel(n=8, T=22, adopt=(("u00", 13), ("u01", 16)))
    est = VanillaSC({"df": df, "outcome": "total_sales", "treat": "treated",
                     "unitid": "dma", "time": "start_date",
                     "display_graphs": False, "inference": "scpi",
                     "scpi_sims": 100})
    res = est.fit()
    figs = plot_staggered(est.config,
                          res.sub_method_results,
                          res.additional_outputs["event_study"],
                          res.additional_outputs.get("event_study_intervals"))
    series, treatment, _att, event = figs
    # fill_between draws a PolyCollection -> a shaded band per panel
    assert any(ax.collections for ax in series.axes), "no counterfactual PI band"
    assert any(ax.collections for ax in treatment.axes), "no gap PI band"
    assert event.axes[0].collections, "no event-study CI band"
    plt.close("all")


# ── the new Plotter.gap PI band ───────────────────────────────────────────────

def test_plotter_gap_shades_interval_band():
    from mlsynth.utils.plotting import Plotter
    t = np.arange(6.0)
    g = np.array([0.0, 0.1, -0.2, 0.5, 0.8, 1.0])
    lo, hi = g - 0.3, g + 0.3
    ax = Plotter().gap(t, g, interval=(lo, hi))
    assert ax.collections, "gap() did not shade the interval band"
    # backward compatible: no interval -> no band
    ax2 = Plotter().gap(t, g)
    assert not ax2.collections
    plt.close("all")
