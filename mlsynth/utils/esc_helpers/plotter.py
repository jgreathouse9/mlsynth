"""Plotting for ESC: observed treated series vs the ensemble synthetic control
counterfactual with its pointwise prediction interval."""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .structures import ESCResults


def plot_esc(results: ESCResults, *, outcome: str = "Outcome", time: str = "Time",
             treated_color: str = "black",
             counterfactual_color: Union[str, List[str]] = "red",
             save: Union[bool, str] = False) -> None:
    """Plot the treated series vs the ESC counterfactual and its prediction band."""
    import matplotlib.pyplot as plt

    cf_color = (counterfactual_color[0] if isinstance(counterfactual_color, list)
                and counterfactual_color else counterfactual_color)
    fit = results.fit
    labels = np.asarray(results.inputs.time_labels)
    T0 = results.inputs.T0

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.fill_between(labels, np.asarray(fit.lower, float), np.asarray(fit.upper, float),
                    color=cf_color, alpha=0.15, lw=0,
                    label=f"{int(round(fit.level*100))}% prediction interval")
    ax.plot(labels, fit.counterfactual, color=cf_color, linestyle="--",
            label="ESC counterfactual")
    ax.plot(labels, results.inputs.y, color=treated_color, label="Treated")
    if T0 < len(labels):
        ax.axvline(labels[T0], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(time)
    ax.set_ylabel(outcome)
    ax.set_title(f"ESC (ATT={fit.att:.3g}, pre-RMSE={fit.pre_rmse:.3g}, "
                 f"{fit.params.get('n_learners')} learners)")
    ax.legend(frameon=False)
    fig.tight_layout()
    if isinstance(save, str):
        fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
