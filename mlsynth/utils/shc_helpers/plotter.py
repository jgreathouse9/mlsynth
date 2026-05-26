"""Plotting helper for SHC results."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .structures import SHCResults


def plot_shc(
    results: SHCResults,
    *,
    treated_color: str = "black",
    counterfactual_color: str = "red",
    title: Optional[str] = None,
) -> None:
    """Render the SHC counterfactual against the observed treated series.

    Plots the observed series and the SHC counterfactual over the
    ``m + n`` block window, the post-intervention conformal band (if
    inference was run), and a treatment-start indicator.
    """
    inputs = results.inputs
    m = inputs.m
    time = np.asarray(results.time_labels)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(time, results.observed, lw=2.5, color=treated_color,
            label=f"Observed: {inputs.treated_label}")
    ax.plot(time, results.counterfactual, lw=2, color=counterfactual_color,
            linestyle="--", label="Synthetic Historical Control")

    if results.inference is not None and m < len(time):
        post_time = time[m:]
        lo = results.inference.conformal_lower
        hi = results.inference.conformal_upper
        if lo is not None and hi is not None and len(lo) == len(post_time):
            ax.fill_between(
                post_time, lo, hi, color=counterfactual_color, alpha=0.15,
                label=f"{int(results.inference.confidence_level * 100)}% conformal band",
            )

    if m < len(time):
        ax.axvline(x=time[m], color="gray", linestyle="--", alpha=0.7,
                   label="Treatment start")

    if title is None:
        variant = "ASHC" if results.design.use_augmented else "SHC"
        title = f"{variant} counterfactual (m={inputs.m}, n={inputs.n})"
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
