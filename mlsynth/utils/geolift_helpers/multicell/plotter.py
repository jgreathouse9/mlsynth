"""Minimal multi-cell plot: each cell's observed vs synthetic, gap below."""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_multicell(result, *, show: bool = True, theme: Optional[dict] = None):
    """Plot every cell's observed vs synthetic path (one row per cell).

    Black = observed, red = synthetic counterfactual; a dashed line marks the
    intervention. Uses the shared ``mlsynth`` house style.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from mlsynth.utils.plotting import mlsynth_style

    cells = result.cells or {}
    if not cells:
        return None

    labels = sorted(cells)
    with mlsynth_style(theme):
        fig, axes = plt.subplots(len(labels), 1, figsize=(8, 2.6 * len(labels)),
                                 squeeze=False)
        for ax, label in zip(axes[:, 0], labels):
            ts = cells[label].time_series
            t = np.asarray(ts.time_periods)
            ax.plot(t, np.asarray(ts.observed_outcome), color="black", label="observed")
            ax.plot(t, np.asarray(ts.counterfactual_outcome), color="red",
                    linestyle="--", label="synthetic")
            if ts.intervention_time is not None:
                ax.axvline(ts.intervention_time, color="grey", linestyle=":")
            att = cells[label].effects.att
            ax.set_title(f"Cell {label}"
                         + (f"  (ATT {att:.1f})" if att is not None else ""))
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
    if show:
        matplotlib.pyplot.show()
    return fig
