"""Diagnostic plot for NSC results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import NSCResults


def plot_nsc(results: NSCResults) -> None:
    """Two-panel plot: trajectories + gap with CI.

    The left panel overlays the treated and synthetic-control series
    over the full panel; the right panel shows the per-period gap
    plus the Doudchenko-Imbens confidence band when available.
    """

    inputs = results.inputs
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:
        t = np.arange(inputs.T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---------- trajectories ----------
    ax = axes[0]
    ax.plot(t, inputs.treated_outcome, "k-", lw=2, label=str(inputs.treated_unit_name))
    ax.plot(t, results.counterfactual, "r--", lw=2, label="NSC counterfactual")
    if inputs.T0 - 1 >= 0 and inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("outcome")
    ax.set_title("NSC trajectories")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    # ---------- gap ----------
    ax = axes[1]
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.plot(t, results.gap, "b-", lw=2, label="Y_treated - Y_SC")
    inf = results.inference_detail
    if inf.gap_lower.size:
        ax.fill_between(
            t, inf.gap_lower, inf.gap_upper,
            color="blue", alpha=0.15,
            label=f"{int(round((1 - inf.alpha) * 100))}% CI",
        )
    if inputs.T0 - 1 >= 0 and inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("treatment effect")
    ax.set_title("NSC gap")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        raise MlsynthPlottingError(f"NSC plotting failed: {exc}") from exc
