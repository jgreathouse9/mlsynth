"""Plotting for GPITS. Returns the figure; showing and saving are the caller's."""

from __future__ import annotations

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def _plottable(x: np.ndarray) -> bool:
    """Whether an object-dtype label array can be used as an axis."""
    return all(isinstance(v, (int, float, np.number)) for v in x)


def _one(color: Union[str, List[str]], fallback: str) -> str:
    """Take a single colour from the config, which may carry a list."""
    if isinstance(color, (list, tuple)):
        return str(color[0]) if color else fallback
    return str(color) if color else fallback


def plot_gpits(results, title: Optional[str] = None,
               counterfactual_color: Union[str, List[str]] = "red",
               treated_color: Union[str, List[str]] = "black"):
    """Observed series, GP counterfactual, and the band that widens off support."""
    counterfactual_color = _one(counterfactual_color, "red")
    treated_color = _one(treated_color, "black")
    ts = results.time_series
    obs = np.asarray(ts.observed_outcome, dtype=float).ravel()
    cf = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    lo = np.asarray(ts.counterfactual_lower, dtype=float).ravel()
    hi = np.asarray(ts.counterfactual_upper, dtype=float).ravel()
    T0 = int(results.inputs.T0)

    # GPITS predicts inside the observed index, so every point has a real
    # period label; use them and let matplotlib format the axis.
    x = np.asarray(ts.time_periods)
    if x.size != obs.size or x.dtype == object and not _plottable(x):
        x = np.arange(obs.size)
    x_label = "Period"
    if np.issubdtype(x.dtype, np.datetime64):
        x_label = "Date"
    # The intervention starts at the first treated period; mark the boundary
    # between the last pre and the first post period.
    cut = (x[T0] if 0 < T0 < x.size else None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, lo, hi, color=counterfactual_color, alpha=0.15, lw=0,
                    label="Counterfactual interval")
    ax.plot(x, obs, color=treated_color, lw=1.6,
            label=f"Observed: {results.inputs.treated_label}")
    ax.plot(x, cf, color=counterfactual_color, lw=1.8, ls="--",
            label="GP counterfactual")
    if cut is not None:
        ax.axvline(cut, color="#555555", ls=":", lw=1.1, label="Treatment start")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Outcome")
    ax.set_title(title or "Gaussian-process interrupted time series")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if np.issubdtype(x.dtype, np.datetime64):
        fig.autofmt_xdate()
    fig.tight_layout()
    return fig
