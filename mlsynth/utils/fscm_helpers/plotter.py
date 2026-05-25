"""Plotting for FSCM: treated-vs-counterfactual and the donor-count CV curve.

The observed-vs-counterfactual panel is delegated to the shared
:class:`~mlsynth.utils.plotting.Plotter`; the rolling-origin CV curve is FSCM's
own bespoke panel and stays local.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from ..plotting import Plotter, mlsynth_style
from .structures import FSCMResults


def plot_fscm(
    results: FSCMResults,
    *,
    outcome: str,
    time: str,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str] = False,
) -> None:
    """Outcome paths (shared archetype) plus the donor-count CV curve."""
    import matplotlib.pyplot as plt

    inputs = results.inputs
    times = np.asarray(inputs.time_index.labels)
    intervention = inputs.metadata.get("intervention_time")
    path = results.selection_path

    with mlsynth_style():
        # Two panels when forward selection ran (outcome + CV curve); one otherwise.
        if path is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

        plotter = Plotter(treated_color=treated_color,
                          counterfactual_colors=counterfactual_color)
        plotter.observed_vs_counterfactual(
            times, inputs.y, results.counterfactual,
            labels=[f"FSCM ({results.n_selected} donors)"],
            treated_label=inputs.treated_label,
            intervention=intervention,
            outcome=outcome, time=time,
            title="Treated vs. synthetic counterfactual",
            ax=ax1,
        )

        if path is not None:
            ax2.plot(path.sizes, path.test_rmspe, color="darkorange", marker="o", markersize=3)
            ax2.axvline(path.optimal_size, color="grey", linestyle="--", linewidth=1,
                        label=f"optimal = {path.optimal_size}")
            ax2.set_xlabel("Number of donors")
            ax2.set_ylabel("CV RMSPE")
            ax2.set_title("Rolling-origin cross-validation")
            ax2.legend()

        fig.tight_layout()
        if save:
            fname = save if isinstance(save, str) else "fscm.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
