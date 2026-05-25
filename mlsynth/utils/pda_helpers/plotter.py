"""Observed-vs-counterfactual overlay plot for PDA (self-contained, NumPy)."""

from __future__ import annotations

import warnings
from typing import List, Union

import numpy as np

from .structures import PDAResults


def plot_pda(
    results: PDAResults, *, outcome: str, time: str,
    treated_color: str = "black", counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
) -> None:
    """Plot the treated outcome against each PDA variant's counterfactual."""
    try:
        import matplotlib.pyplot as plt

        inp = results.inputs
        years = np.asarray(inp.time_index.labels)
        colors = ([counterfactual_color] if isinstance(counterfactual_color, str)
                  else list(counterfactual_color))
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.plot(years, inp.y, color=treated_color, lw=2.2, label=f"{inp.treated_label} (observed)")
        for i, (name, fit) in enumerate(results.fits.items()):
            ax.plot(years, fit.counterfactual, ls="--", lw=1.8,
                    color=colors[i % len(colors)], label=f"PDA [{name}]")
        if inp.T0 < len(years):
            ax.axvline(years[inp.T0], color="gray", ls=":")
        ax.set_xlabel(time); ax.set_ylabel(outcome); ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        if save:
            fig.savefig(save if isinstance(save, str) else "pda_estimates.png", dpi=130)
        else:
            plt.show()
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"PDA plotting failed: {e}", UserWarning)
