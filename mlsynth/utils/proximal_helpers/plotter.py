"""Diagnostic plot for PROXIMAL results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import PROXIMALResults

_STYLES = {
    "PI": ("r--", "Proximal Inference"),
    "PIS": ("b-.", "Proximal Surrogates"),
    "PIPost": ("g:", "Proximal Post"),
}


def plot_proximal(results: PROXIMALResults) -> None:
    """Two-panel plot: trajectories + gap, with one overlay per method run."""

    inputs = results.inputs
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:
        t = np.arange(inputs.T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(t, inputs.y, "k-", lw=2, label=str(inputs.treated_unit_name))
    for name, fit in results.methods.items():
        style, label = _STYLES.get(name, ("--", name))
        ax.plot(t, fit.counterfactual, style, lw=2, label=f"{label} counterfactual")
    if 0 <= inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("outcome")
    ax.set_title("PROXIMAL trajectories")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.axhline(0.0, color="grey", lw=0.8)
    for name, fit in results.methods.items():
        style, label = _STYLES.get(name, ("-", name))
        ax.plot(t, fit.gap, style, lw=2, label=f"{label} gap")
    if 0 <= inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("treatment effect")
    ax.set_title("PROXIMAL gap")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        raise MlsynthPlottingError(f"PROXIMAL plotting failed: {exc}") from exc
