"""Observed vs BPSCS counterfactual plot with the posterior credible band."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .structures import BPSCSResults


def plot_bpscs(results: BPSCSResults, title: str | None = None) -> None:
    """Plot the observed series, the BPSCS counterfactual, and its credible band."""
    inputs = results.inputs
    inf = results.inference_detail
    time = np.asarray(inputs.time_labels)
    T0 = inputs.T0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, inputs.y_target, color="black", linewidth=2,
            label=f"Observed: {inputs.treated_unit_name}")
    ax.plot(time, inf.counterfactual_median, color="tab:red", linestyle="--",
            linewidth=1.8, label=f"BPSCS counterfactual ({inputs.prior})")
    ax.fill_between(time, inf.counterfactual_lower, inf.counterfactual_upper,
                    color="tab:red", alpha=0.2,
                    label=f"{int(round((1 - inf.ci_alpha) * 100))}% credible interval")
    if T0 < len(time):
        ax.axvline(x=time[T0], color="gray", linestyle=":", label="Treatment start")
    if title is None:
        att = inf.att_median
        att_str = "" if np.isnan(att) else f", ATT={att:.3f}"
        title = f"BPSCS counterfactual: {inputs.treated_unit_name}{att_str}"
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("Time"); ax.set_ylabel("Outcome")
    ax.legend(loc="best", frameon=True, fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.show(); plt.close(fig)
