"""Observed-vs-counterfactual overlay plot for SCMO (self-contained, NumPy)."""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np

from .structures import SCMOResults


def plot_scmo(
    results: SCMOResults,
    *,
    outcome: str,
    time: str,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
) -> None:
    """Plot the treated outcome against each scheme's counterfactual.

    Plotting failures are downgraded to warnings so a rendering problem never
    masks a successful estimation.
    """
    try:
        import matplotlib.pyplot as plt

        inputs = results.inputs
        years = np.asarray(inputs.time_index.labels)
        T0 = inputs.T0
        colors = ([counterfactual_color] if isinstance(counterfactual_color, str)
                  else list(counterfactual_color))

        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.plot(years, inputs.y_treated, color=treated_color, lw=2.2,
                label=f"{inputs.treated_label} (observed)")
        for i, (name, fit) in enumerate(results.fits.items()):
            ax.plot(years, fit.counterfactual, ls="--", lw=1.8,
                    color=colors[i % len(colors)], label=f"SCMO [{name}]")
        if T0 < len(years):
            ax.axvline(years[T0], color="gray", ls=":")
        ax.set_xlabel(time); ax.set_ylabel(outcome); ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        if save:
            path = save if isinstance(save, str) else "scmo_estimates.png"
            fig.savefig(path, dpi=130)
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"SCMO plotting failed: {e}", UserWarning)
