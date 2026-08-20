"""Observed vs counterfactual plot for BLTVP, with posterior credible band."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .structures import BLTVPResults


def plot_bltvp(results: BLTVPResults, title: Optional[str] = None) -> None:
    """Plot observed series, BLTVP counterfactual, and pointwise credible band.

    One solid black series for the observed outcome, a dashed blue
    posterior-mean counterfactual, a shaded credible band, and a vertical line
    at ``T_0``. The band widens after ``T_0`` because the coefficient path is
    unobserved there and diffuses.
    """
    inputs = results.inputs
    inference = results.inference_detail
    time = np.asarray(inputs.time_labels)
    T0 = inputs.T0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, inputs.y_target, color="black", linewidth=2,
            label=f"Observed: {inputs.treated_unit_name}")
    ax.plot(time, inference.counterfactual_mean, color="blue", linestyle="--",
            linewidth=2, label="BLTVP counterfactual")
    ax.fill_between(time, inference.counterfactual_lower,
                    inference.counterfactual_upper, color="blue", alpha=0.15,
                    label=f"{int(round(100 * (1 - inference.ci_alpha)))}% credible band")
    if T0 < inputs.T:
        ax.axvline(time[T0], color="grey", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.set_title(title or f"BLTVP: {inputs.treated_unit_name}")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
