"""Diagnostic plot for CLUSTERSC results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import CLUSTERSCResults


def plot_clustersc(results: CLUSTERSCResults) -> None:
    """Two-panel plot: trajectories + gap, both with PCR / RPCA overlays."""

    inputs = results.inputs
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:
        t = np.arange(inputs.T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(t, inputs.treated_outcome, "k-", lw=2,
            label=str(inputs.treated_unit_name))
    if results.pcr is not None:
        ax.plot(t, results.pcr.counterfactual, "r--", lw=2,
                label="PCR counterfactual")
    if results.rpca is not None:
        ax.plot(t, results.rpca.counterfactual, "b-.", lw=2,
                label="RPCA counterfactual")
    if 0 <= inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("outcome")
    ax.set_title("CLUSTERSC trajectories")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.axhline(0.0, color="grey", lw=0.8)
    if results.pcr is not None:
        ax.plot(t, results.pcr.gap, "r-", lw=2, label="PCR gap")
    if results.rpca is not None:
        ax.plot(t, results.rpca.gap, "b-", lw=2, label="RPCA gap")
    if 0 <= inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("treatment effect")
    ax.set_title("CLUSTERSC gap")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        raise MlsynthPlottingError(f"CLUSTERSC plotting failed: {exc}") from exc
