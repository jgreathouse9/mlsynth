"""Plotting for GPITS. Returns the figure; showing and saving are the caller's."""

from __future__ import annotations

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


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
    x = np.arange(obs.size)
    T0 = int(results.inputs.T0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, lo, hi, color=counterfactual_color, alpha=0.15, lw=0,
                    label="Counterfactual interval")
    ax.plot(x, obs, color=treated_color, lw=1.6,
            label=f"Observed: {results.inputs.treated_label}")
    ax.plot(x, cf, color=counterfactual_color, lw=1.8, ls="--",
            label="GP counterfactual")
    ax.axvline(T0 - 0.5, color="#555555", ls=":", lw=1.1,
               label="Treatment start")
    ax.set_xlabel("Period")
    ax.set_ylabel("Outcome")
    ax.set_title(title or "Gaussian-process interrupted time series")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
