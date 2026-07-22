"""Observed vs counterfactual plot for MVBBSC, with posterior credible band."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .structures import MVBBSCResults


def plot_mvbbsc(results: MVBBSCResults, title: str | None = None) -> None:
    """Plot observed series, MVBBSC counterfactual, and pointwise credible band.

    One solid black series for the observed outcome, a dashed blue
    posterior-mean counterfactual, a shaded credible band, and a vertical line
    at ``T_0``.
    """

    inputs = results.inputs
    inference = results.inference_detail

    time = inputs.time_labels
    T0 = inputs.T0

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        time, inputs.y_target,
        color="black", linewidth=2, label=f"Observed: {inputs.treated_unit_name}",
    )
    ax.plot(
        time, inference.counterfactual_mean,
        color="tab:blue", linestyle="--", linewidth=1.8,
        label="MVBBSC counterfactual",
    )
    ax.fill_between(
        time,
        inference.counterfactual_lower,
        inference.counterfactual_upper,
        color="tab:blue", alpha=0.2,
        label=f"{int(round((1 - inference.ci_alpha) * 100))}% credible interval",
    )
    if T0 < len(time):
        ax.axvline(x=time[T0], color="red", linestyle=":", label="Treatment start")

    if title is None:
        att_str = ("" if np.isnan(inference.att_mean)
                   else f", ATT={inference.att_mean:.3f}")
        title = f"MVBBSC counterfactual: {inputs.treated_unit_name}{att_str}"
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
