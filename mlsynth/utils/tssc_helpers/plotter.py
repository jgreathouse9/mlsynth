"""Observed vs recommended-counterfactual plot for TSSC."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from .structures import TSSCResults


def plot_tssc(results: TSSCResults, title: Optional[str] = None) -> None:
    """Plot the treated series against the recommended variant's synthetic."""

    inputs = results.inputs
    variant = results.recommended
    time = inputs.time_labels
    T0 = inputs.T0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        time, inputs.y, color="black", linewidth=2,
        label=f"Observed: {inputs.treated_unit_name}",
    )
    ax.plot(
        time, variant.counterfactual, color="tab:red", linestyle="--",
        linewidth=1.8, label=f"TSSC synthetic ({variant.method})",
    )
    if T0 < len(time):
        ax.axvline(x=time[T0], color="gray", linestyle=":", label="Treatment start")

    if title is None:
        title = (
            f"TSSC ({variant.method}): {inputs.treated_unit_name}"
            f", ATT={variant.att:.3f}"
        )
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
