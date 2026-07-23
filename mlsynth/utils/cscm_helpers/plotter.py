"""Observed-vs-synthetic plot for CSCM."""

from __future__ import annotations

import numpy as np


def plot_cscm(config, y, counterfactual, time_labels, pre_periods, resolved
              ) -> None:
    """Render the observed vs synthetic (CSCM) counterfactual."""
    import matplotlib.pyplot as plt

    from ..plotting import Plotter, mlsynth_style

    y = np.asarray(y, dtype=float).ravel()
    counterfactual = np.asarray(counterfactual, dtype=float).ravel()
    T = len(y)
    if time_labels is None:
        time_labels = np.arange(T)
    intervention = time_labels[pre_periods] if 0 <= pre_periods < T else None

    with mlsynth_style():
        plotter = Plotter.from_config(getattr(config, "plot", None))
        ax = plotter.observed_vs_counterfactual(
            times=time_labels, observed=y, counterfactuals=[counterfactual],
            labels=["Synthetic (CSCM)"], treated_label="Treated",
            intervention=intervention, outcome=config.outcome, time=config.time,
            title="Flexible count synthetic control (CSCM)",
        )
        fig = ax.figure
        if resolved.save:
            fname = resolved.save if isinstance(resolved.save, str) else "CSCM.png"
            fig.savefig(fname, bbox_inches="tight")
        if resolved.display:
            plt.show()
        plt.close(fig)
