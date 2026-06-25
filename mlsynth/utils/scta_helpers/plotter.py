"""Plotting for SCTA: observed-vs-counterfactual and the imbalance frontier."""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .structures import SCTAResults


def plot_scta(results: SCTAResults, *, outcome: str = "Outcome",
              time: str = "Time", treated_color: str = "black",
              counterfactual_color: Union[str, List[str]] = "red",
              save: Union[bool, str] = False) -> None:
    """Plot the treated series against its SCTA counterfactual.

    When the fit carries an imbalance frontier, a second panel traces the
    disaggregated-vs-aggregated pre-treatment RMSE across ``nu`` (the paper's
    Figure 1).
    """
    import matplotlib.pyplot as plt

    cf_color = (counterfactual_color[0] if isinstance(counterfactual_color, list)
                and counterfactual_color else counterfactual_color)
    fit = results.fit
    labels = np.asarray(results.inputs.time_labels)
    T0 = results.inputs.T0

    has_frontier = results.frontier is not None
    fig, axes = plt.subplots(1, 2 if has_frontier else 1,
                             figsize=(12 if has_frontier else 7, 4.5))
    ax = axes[0] if has_frontier else axes

    ax.plot(labels, results.inputs.y, color=treated_color, label="Treated")
    ax.plot(labels, fit.counterfactual, color=cf_color, linestyle="--",
            label="SCTA counterfactual")
    if T0 < len(labels):
        ax.axvline(labels[T0], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(time)
    ax.set_ylabel(outcome)
    ax.set_title(f"SCTA (nu={fit.nu:g})")
    ax.legend(frameon=False)

    if has_frontier:
        axf = axes[1]
        xs = [pt["rmse_dis"] for pt in results.frontier]
        ys = [pt["rmse_agg"] for pt in results.frontier]
        axf.plot(xs, ys, marker="o", color=cf_color)
        for pt in results.frontier:
            axf.annotate(f"{pt['nu']:g}", (pt["rmse_dis"], pt["rmse_agg"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=8)
        axf.set_xlabel("Disaggregated pre-period RMSE")
        axf.set_ylabel("Aggregated pre-period RMSE")
        axf.set_title("Imbalance frontier")

    fig.tight_layout()
    if isinstance(save, str):
        fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
