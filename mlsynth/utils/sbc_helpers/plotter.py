"""Plotting helper for SBC results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .structures import SBCResults


def plot_sbc(results: SBCResults, title: str | None = None) -> None:
    """Render the SBC counterfactual against the observed treated series.

    Plots ``y_target`` over the entire window plus the SBC post-treatment
    counterfactual (trend + cycle), with a treatment-start indicator.
    """

    inputs = results.inputs
    design = results.design
    T0 = inputs.T0
    time = inputs.time_labels

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(time, inputs.y_target, lw=2.5, color="black",
            label=f"Observed: {inputs.treated_unit_name}")
    hz = design.counterfactual_post.size
    if hz > 0:
        # The counterfactual spans only the h-step forecast horizon.
        post_time = time[T0:T0 + hz]
        ax.plot(
            post_time, design.counterfactual_post,
            lw=2, color="tab:red", linestyle="--", label="SBC counterfactual"
        )
        # Also overlay the trend forecast separately for interpretability.
        ax.plot(
            post_time, design.trend_forecast,
            lw=1.2, color="tab:blue", linestyle=":", alpha=0.8,
            label="Forecast trend (treated AR)",
        )

    if T0 < len(time):
        ax.axvline(x=time[T0], color="gray", linestyle="--", alpha=0.7,
                   label="Treatment start")

    if title is None:
        title = (
            f"SBC counterfactual ({design.weights_mode}, "
            f"h={design.treated_hamilton.h}, p={design.treated_hamilton.p})"
        )
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
