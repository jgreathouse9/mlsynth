"""The shared Plotter keeps date x-axes legible.

Weekly ``start_date`` labels over several years arrive as strings, which
Matplotlib plots on a *categorical* axis -- one tick per label -- producing an
unreadable smear of ~130 overlapping dates. The Plotter parses date-like labels
to real datetimes so a date locator auto-thins the ticks, and rotates dense
non-date axes. These pin that behaviour.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.plotting import (
    Plotter, _coerce_time_axis, _coerce_time_scalar,
)


def _weekly(n_weeks=134):
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    return [d.strftime("%Y-%m-%d") for d in weeks]


# ── the coercion helper ───────────────────────────────────────────────────────

def test_coerce_parses_date_strings():
    vals, is_dt = _coerce_time_axis(["2024-01-01", "2024-01-08", "2024-01-15"])
    assert is_dt
    assert np.issubdtype(np.asarray(vals).dtype, np.datetime64)


def test_coerce_leaves_integer_years_alone():
    vals, is_dt = _coerce_time_axis(np.arange(1970, 2001))
    assert not is_dt
    assert np.asarray(vals).dtype.kind in ("i", "u")


def test_coerce_leaves_non_date_strings_alone():
    vals, is_dt = _coerce_time_axis(["north", "south", "east"])
    assert not is_dt


def test_coerce_scalar_matches_axis():
    dt = _coerce_time_scalar("2024-06-03", True)
    assert np.asarray(dt).dtype.kind == "M"
    assert _coerce_time_scalar(1990, False) == 1990          # untouched off a datetime axis


# ── observed vs counterfactual: weekly dates -> thinned date axis ─────────────

def test_weekly_dates_get_date_axis_not_134_ticks():
    times = _weekly(134)
    y = np.linspace(0, 1, len(times))
    ax = Plotter().observed_vs_counterfactual(
        times, y, [y * 0.9], intervention=times[70])
    # a concise *date* formatter, and the plotted x-data are datetimes
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    xd = np.asarray(ax.get_lines()[0].get_xdata())
    assert np.issubdtype(xd.dtype, np.datetime64)
    # the date locator thins 134 weeks down to a legible handful
    ax.figure.canvas.draw()
    assert len(ax.get_xticks()) <= 12
    plt.close("all")


def test_integer_year_axis_stays_numeric():
    times = np.arange(1970, 2001)
    y = np.linspace(0, 1, len(times))
    ax = Plotter().observed_vs_counterfactual(times, y, [y])
    assert not isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    xd = np.asarray(ax.get_lines()[0].get_xdata())
    assert xd.dtype.kind in ("i", "u", "f")
    plt.close("all")


def test_gap_weekly_dates_thinned():
    times = _weekly(120)
    g = np.linspace(-1, 1, len(times))
    ax = Plotter().gap(times, g, intervention=times[60])
    ax.figure.canvas.draw()
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    assert len(ax.get_xticks()) <= 12
    plt.close("all")


# ── integration: the staggered facets inherit the clean date axis ─────────────

def test_staggered_series_facets_have_date_axis():
    from mlsynth import VanillaSC
    from mlsynth.utils.vanillasc_helpers.staggered_plotter import plot_staggered

    weeks = _weekly(60)
    rng = np.random.default_rng(0)
    rows = []
    adopt = {"DMA_00": 35, "DMA_01": 45}
    for i in range(6):
        name = f"DMA_{i:02d}"
        base = 100 + rng.standard_normal(len(weeks)).cumsum() * 0.2
        for t, wk in enumerate(weeks):
            tr = int(name in adopt and t >= adopt[name])
            rows.append({"dma": name, "start_date": wk,
                         "total_sales": base[t] + 5 * tr, "treated": tr})
    df = pd.DataFrame(rows)
    est = VanillaSC({"df": df, "outcome": "total_sales", "treat": "treated",
                     "unitid": "dma", "time": "start_date", "display_graphs": False})
    res = est.fit()
    figs = plot_staggered(est.config, res.sub_method_results,
                          res.additional_outputs["event_study"])
    series = figs[0]
    panels = [ax for ax in series.axes if ax.get_visible()]
    assert panels and all(
        isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
        for ax in panels)
    plt.close("all")
