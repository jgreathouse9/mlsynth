"""Plotting for MASC: treated-vs-counterfactual and the CV grid.

The observed-vs-counterfactual panel is delegated to the shared
:class:`~mlsynth.utils.plotting.Plotter`; the CV-error-vs-``m`` curve is
MASC's own bespoke panel and stays local.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from ..plotting import Plotter, mlsynth_style
from .structures import MASCResults


def plot_masc(
    results: MASCResults,
    *,
    outcome: str,
    time: str,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str] = False,
) -> None:
    """Outcome paths (shared archetype) plus the CV-grid panel.

    The left panel overlays the treated trajectory and the MASC
    counterfactual (``φ * matching + (1 − φ) * SC``). The right panel
    plots the cross-validation error against the candidate ``m`` grid,
    annotating the CV-selected ``(m̂, φ̂)``.
    """
    import matplotlib.pyplot as plt

    inputs = results.inputs
    times = np.asarray(inputs.time_index)
    intervention = inputs.intervention_time
    cv_grid = results.fit.cv_grid

    with mlsynth_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        plotter = Plotter(
            treated_color=treated_color,
            counterfactual_colors=counterfactual_color,
        )
        plotter.observed_vs_counterfactual(
            times,
            inputs.Y_treated,
            results.counterfactual,
            labels=[f"MASC (m={results.m_hat}, φ={results.phi_hat:.2f})"],
            treated_label=inputs.treated_label,
            intervention=intervention,
            outcome=outcome,
            time=time,
            title="Treated vs. synthetic counterfactual",
            ax=ax1,
        )

        ax2.plot(
            cv_grid[:, 0], cv_grid[:, 2],
            color="darkorange", marker="o", markersize=4,
        )
        ax2.axvline(
            results.m_hat, color="grey", linestyle="--", linewidth=1,
            label=f"m̂ = {results.m_hat}, φ̂ = {results.phi_hat:.3f}",
        )
        ax2.set_xlabel("Number of nearest neighbours (m)")
        ax2.set_ylabel("Rolling-origin CV error")
        ax2.set_title("MASC cross-validation curve")
        ax2.legend()

        fig.tight_layout()
        if save:
            fname = save if isinstance(save, str) else "masc.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
