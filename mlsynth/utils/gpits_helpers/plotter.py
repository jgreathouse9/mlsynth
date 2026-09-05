"""Plotting for GPITS. Returns figures; showing and saving are the caller's.

Mirrors ``plot.gp_its`` from the paper's replication code
(``soonhong-cho/gpits``, ``code/00_its_helpers.R``), which returns four panels:
the model fit, the pointwise effects with the placebo window prepended, the
cumulative effects, and the average effects. :func:`plot_gpits` draws the first,
which is the one an estimator's ``display_graphs`` wants;
:func:`plot_gpits_panels` returns all four under the reference's own names.

One convention of the reference is carried over deliberately. The fit panel
draws the pre-period band in grey and the post-period band in the
counterfactual colour, because the two mean different things: before the
intervention the band is the uncertainty of a fit to observed data, after it
the band is the uncertainty of an extrapolation. Drawing one continuous ribbon
would blur the distinction the method exists to make.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_gpits", "plot_gpits_panels"]

# plot.gp_its defaults: y0_color, y0_alpha, placebo_color.
Y0_ALPHA = 0.6
RIBBON_ALPHA = 0.3
PLACEBO_COLOR = "#4671D5"
GREY = "lightgrey"


def _plottable(x: np.ndarray) -> bool:
    """Whether an object-dtype label array can be used as an axis."""
    return all(isinstance(v, (int, float, np.number)) for v in x)


def _one(color: Union[str, List[str]], fallback: str) -> str:
    """Take a single colour from the config, which may carry a list."""
    if isinstance(color, (list, tuple)):
        return str(color[0]) if color else fallback
    return str(color) if color else fallback


def _axis(results):
    """Period labels for the x axis, and the label for it.

    GPITS predicts inside the observed index, so every point has a real label;
    only labels matplotlib cannot place fall back to positions.
    """
    ts = results.time_series
    n = np.asarray(ts.observed_outcome).size
    x = np.asarray(ts.time_periods)
    if x.size != n or (x.dtype == object and not _plottable(x)):
        return np.arange(n), "Period"
    return x, ("Date" if np.issubdtype(x.dtype, np.datetime64) else "Period")


def _relative_time(results):
    """Periods since treatment: 0 is the first treated period, as in R's
    ``id_time_std = time_id - time_id_treat``."""
    n = np.asarray(results.time_series.observed_outcome).size
    return np.arange(n) - int(results.inputs.T0)


def _tidy(ax):
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_gpits(results, title: Optional[str] = None,
               counterfactual_color: Union[str, List[str]] = "red",
               treated_color: Union[str, List[str]] = "black"):
    """The fit panel: observed points, the pre-period fit, the counterfactual.

    The pre-period band is the fit's own uncertainty and is drawn in grey; the
    post-period band is the extrapolation's and is drawn in the counterfactual
    colour.
    """
    cf_color = _one(counterfactual_color, "red")
    obs_color = _one(treated_color, "black")
    ts = results.time_series
    obs = np.asarray(ts.observed_outcome, dtype=float).ravel()
    cf = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    lo = np.asarray(ts.counterfactual_lower, dtype=float).ravel()
    hi = np.asarray(ts.counterfactual_upper, dtype=float).ravel()
    T0 = int(results.inputs.T0)
    x, x_label = _axis(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x[:T0], lo[:T0], hi[:T0], color=GREY, alpha=0.5, lw=0,
                    label="Fit interval")
    ax.plot(x[:T0], cf[:T0], color=obs_color, lw=1.0, label="Fitted")
    if T0 < obs.size:
        ax.fill_between(x[T0:], lo[T0:], hi[T0:], color=cf_color,
                        alpha=RIBBON_ALPHA, lw=0, label="Counterfactual interval")
        ax.plot(x[T0:], cf[T0:], color=cf_color, ls="--", lw=1.6,
                alpha=Y0_ALPHA, label="Counterfactual Y")
        ax.axvline(x[T0], color=obs_color, ls="--", lw=0.9,
                   label="Treatment start")
    ax.scatter(x, obs, s=9, color=obs_color, zorder=3,
               label=f"Observed: {results.inputs.treated_label}")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Outcome")
    ax.set_title(title or
                 f"GP-ITS with {results.design.kernel.replace('_', '+')} kernel")
    ax.legend(frameon=False, fontsize=9)
    _tidy(ax)
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _effect_panel(ax, t, effect, lower, upper, *, ylabel, title, color="black"):
    """The shared body of the three effect panels."""
    ax.axhline(0.0, color="grey", ls="--", lw=1.0)
    ax.axvline(0.0, color=color, ls="--", lw=1.4)
    ax.fill_between(t, lower, upper, color=GREY, alpha=0.5, lw=0)
    ax.plot(t, effect, color=color, lw=1.0)
    ax.scatter(t, effect, s=11, color=color, zorder=3)
    ax.set_xlabel("Time Since Treatment")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _tidy(ax)


def plot_gpits_panels(results, counterfactual_color: Union[str, List[str]] = "red",
                      treated_color: Union[str, List[str]] = "black",
                      placebo_color: str = PLACEBO_COLOR) -> Dict[str, plt.Figure]:
    """The reference's four panels, keyed by its own names.

    Returns ``{"fit", "pointwise", "cumulative", "average"}``. ``pointwise``
    prepends the placebo window when the fit carried one, shading it separately
    so a reader can see the periods where the effect is known to be zero next
    to the periods where it is being estimated.

    The average-effect panel is derived from the cumulative one: the average
    over the first ``k`` post-periods is the cumulative total divided by ``k``,
    and its standard error scales the same way.
    """
    obs_color = _one(treated_color, "black")
    T0 = int(results.inputs.T0)
    rel = _relative_time(results)
    rel_post = rel[T0:]

    cum = np.asarray(results.cumulative_effect, dtype=float)
    cum_lo = np.array([c[0] for c in results.cumulative_ci], dtype=float)
    cum_hi = np.array([c[1] for c in results.cumulative_ci], dtype=float)
    k = np.arange(1, cum.size + 1, dtype=float)

    # Pointwise: the effect is observed minus counterfactual, so its interval
    # is the counterfactual band reflected about that point. Taking the
    # half-width off the stored band keeps the level exactly the fit's own.
    ts = results.time_series
    gap = np.asarray(ts.estimated_gap, dtype=float)[T0:]
    cf = np.asarray(ts.counterfactual_outcome, dtype=float)[T0:]
    half = np.asarray(ts.counterfactual_upper, dtype=float)[T0:] - cf
    t_pt, e_pt = rel_post, gap
    lo_pt, hi_pt = gap - half, gap + half

    placebo = results.placebo
    if placebo is not None:
        labels = list(np.asarray(ts.time_periods))
        idx = np.array([labels.index(v) for v in np.asarray(placebo.time_labels)])
        rel_pl = idx - T0
        t_pt = np.concatenate([rel_pl, t_pt])
        e_pt = np.concatenate([np.asarray(placebo.tau, dtype=float), e_pt])
        lo_pt = np.concatenate([np.asarray(placebo.ci_lower, dtype=float), lo_pt])
        hi_pt = np.concatenate([np.asarray(placebo.ci_upper, dtype=float), hi_pt])

    figs: Dict[str, plt.Figure] = {}
    figs["fit"] = plot_gpits(results, counterfactual_color=counterfactual_color,
                             treated_color=treated_color)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _effect_panel(ax, t_pt, e_pt, lo_pt, hi_pt, ylabel="Pointwise Effect",
                  title=("Pointwise Treatment Effects (with Placebo)"
                         if placebo is not None else "Pointwise Treatment Effects"),
                  color=obs_color)
    if placebo is not None:
        ax.fill_between(rel_pl, np.asarray(placebo.ci_lower, dtype=float),
                        np.asarray(placebo.ci_upper, dtype=float),
                        color=placebo_color, alpha=RIBBON_ALPHA, lw=0,
                        label="Placebo window")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    figs["pointwise"] = fig

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _effect_panel(ax, rel_post, cum, cum_lo, cum_hi,
                  ylabel="Cumulative Effect", title="Cumulative Treatment Effects",
                  color=obs_color)
    fig.tight_layout()
    figs["cumulative"] = fig

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _effect_panel(ax, rel_post, cum / k, cum_lo / k, cum_hi / k,
                  ylabel="Average Effect", title="Average Treatment Effects",
                  color=obs_color)
    fig.tight_layout()
    figs["average"] = fig

    return figs
