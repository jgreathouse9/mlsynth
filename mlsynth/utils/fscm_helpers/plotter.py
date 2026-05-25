"""Plotting for FSCM: treated-vs-counterfactual and the donor-count CV curve."""

from __future__ import annotations

from typing import List, Union

import numpy as np

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
    """Two-panel figure: outcome paths and the test-RMSPE selection curve."""
    import matplotlib.pyplot as plt

    inputs = results.inputs
    times = np.asarray(inputs.time_index.labels)
    intervention = inputs.metadata.get("intervention_time")
    cf_color = counterfactual_color[0] if isinstance(counterfactual_color, list) else counterfactual_color
    path = results.selection_path

    # Two panels when forward selection ran (outcome + CV curve); one otherwise.
    if path is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    ax1.plot(times, inputs.y, color=treated_color, label=f"Treated ({inputs.treated_label})")
    ax1.plot(times, results.counterfactual, color=cf_color, linestyle="--",
             label=f"FSCM ({results.n_selected} donors)")
    if intervention is not None:
        ax1.axvline(intervention, color="grey", linestyle=":", linewidth=1)
    ax1.set_xlabel(time)
    ax1.set_ylabel(outcome)
    ax1.set_title("Treated vs. synthetic counterfactual")
    ax1.legend(frameon=False)

    if path is not None:
        ax2.plot(path.sizes, path.test_rmspe, color="darkorange", marker="o", markersize=3)
        ax2.axvline(path.optimal_size, color="grey", linestyle="--", linewidth=1,
                    label=f"optimal = {path.optimal_size}")
        ax2.set_xlabel("Number of donors")
        ax2.set_ylabel("CV RMSPE")
        ax2.set_title("Rolling-origin cross-validation")
        ax2.legend(frameon=False)

    fig.tight_layout()
    if save:
        fname = save if isinstance(save, str) else "fscm.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
