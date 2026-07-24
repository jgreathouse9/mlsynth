"""Plotting for DROSC: treated vs. robust synthetic counterfactual, delegated to
the shared :class:`~mlsynth.utils.plotting.Plotter`."""
from __future__ import annotations

from typing import List, Union

import numpy as np

from ..plotting import Plotter, mlsynth_style


def plot_drosc(
    results,
    *,
    outcome: str,
    time: str,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str] = False,
) -> None:
    """Observed vs. robust counterfactual, with the treatment line and the
    robustness radius in the label."""
    import matplotlib.pyplot as plt

    ts = results.time_series
    times = np.asarray(ts.time_periods)
    observed = np.asarray(ts.observed_outcome, dtype=float)
    counterfactual = np.asarray(ts.counterfactual_outcome, dtype=float)
    lam = float(results.additional_outputs.get("robustness_lambda", 0.0))

    with mlsynth_style():
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        plotter = Plotter(treated_color=treated_color,
                          counterfactual_colors=counterfactual_color)
        plotter.observed_vs_counterfactual(
            times, observed, counterfactual,
            labels=[f"DROSC (λ = {lam:g})"],
            treated_label=str(results.additional_outputs.get("treated_name", "Treated")),
            intervention=ts.intervention_time,
            outcome=outcome, time=time,
            title="Treated vs. robust synthetic counterfactual",
            ax=ax,
        )
        fig.tight_layout()
        if save:
            fname = save if isinstance(save, str) else "drosc.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
