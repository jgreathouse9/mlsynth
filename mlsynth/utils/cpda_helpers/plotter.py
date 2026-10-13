"""Presentation for CPDA. Returns its Figure; the caller decides the rest."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np


def plot_cpda(results, *, outcome: str = "Outcome", time: str = "Time",
              treated_color: str = "black",
              counterfactual_color: Union[str, List[str]] = "#3b78e7",
              save: Union[bool, str, dict] = False):
    """Observed against the CPDA counterfactual, with the treatment marked.

    Returns the Figure and neither shows nor saves it unless ``save`` names a
    path, which keeps computing and displaying separate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = results.time_series
    t = np.asarray(ts.time_periods)
    obs = np.asarray(ts.observed_outcome, dtype=float)
    cf = np.asarray(ts.counterfactual_outcome, dtype=float)
    colour = (counterfactual_color[0] if isinstance(counterfactual_color, list)
              else counterfactual_color)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(t, obs, color=treated_color, lw=2.0, label="observed")
    ax.plot(t, cf, color=colour, lw=2.0, label="CPDA counterfactual")
    if ts.intervention_time is not None:
        ax.axvline(ts.intervention_time, color="#6b6b6b", ls="--", lw=1.2)
    ax.set_xlabel(time)
    ax.set_ylabel(outcome)
    ax.set_title(f"CPDA: ATT {results.effects.att:.3f}", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if isinstance(save, str):
        fig.savefig(save, dpi=200, bbox_inches="tight")
    elif isinstance(save, dict) and save.get("path"):
        fig.savefig(save["path"], dpi=save.get("dpi", 200), bbox_inches="tight")
    return fig
