"""Plotting for TWSF. Returns the figure; showing and saving are the caller's."""

from __future__ import annotations

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def _one(color: Union[str, List[str]], fallback: str) -> str:
    """Take a single colour from the config, which may carry a list."""
    if isinstance(color, (list, tuple)):
        return str(color[0]) if color else fallback
    return str(color) if color else fallback


def plot_twsf(results, title: Optional[str] = None,
              counterfactual_color: Union[str, List[str]] = "red",
              treated_color: Union[str, List[str]] = "black"):
    """Draw the observed path, the forecast past the panel, and its band."""
    counterfactual_color = _one(counterfactual_color, "red")
    treated_color = _one(treated_color, "black")
    ts = results.time_series
    obs = np.asarray(ts.observed_outcome, dtype=float).ravel()
    fc = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    lo = np.asarray(ts.counterfactual_lower, dtype=float).ravel()
    hi = np.asarray(ts.counterfactual_upper, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(obs.size), obs, color=treated_color, lw=1.6,
            label="Observed (control)")
    fx = np.arange(obs.size, obs.size + fc.size)
    ax.fill_between(fx, lo, hi, color=counterfactual_color, alpha=0.15, lw=0,
                    label="Pointwise interval")
    ax.plot(fx, fc, color=counterfactual_color, lw=1.8, ls="-.",
            label="TWSF forecast (treated)")
    ax.axvline(obs.size - 0.5, color="#555555", ls=":", lw=1.1)
    ax.set_xlabel("Period")
    ax.set_ylabel("Outcome")
    ax.set_title(title or "Two-Way Synthetic Forecasting")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
