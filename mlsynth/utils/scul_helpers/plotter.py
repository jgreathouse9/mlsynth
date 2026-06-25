"""Plotting for SCUL: observed treated series vs the lasso synthetic control."""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .structures import SCULResults


def plot_scul(results: SCULResults, *, outcome: str = "Outcome",
              time: str = "Time", treated_color: str = "black",
              counterfactual_color: Union[str, List[str]] = "red",
              save: Union[bool, str] = False) -> None:
    """Plot the treated series against its SCUL (lasso) synthetic control."""
    import matplotlib.pyplot as plt

    cf_color = (counterfactual_color[0] if isinstance(counterfactual_color, list)
                and counterfactual_color else counterfactual_color)
    fit = results.fit
    labels = np.asarray(results.inputs.time_labels)
    T0 = results.inputs.T0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(labels, results.inputs.y, color=treated_color, label="Treated")
    ax.plot(labels, fit.counterfactual, color=cf_color, linestyle="--",
            label="SCUL synthetic control")
    if T0 < len(labels):
        ax.axvline(labels[T0], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(time)
    ax.set_ylabel(outcome)
    ax.set_title(f"SCUL (lambda={fit.ridge_lambda:.3g}, Cohen's D={fit.cohens_d:.3f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    if isinstance(save, str):
        fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
