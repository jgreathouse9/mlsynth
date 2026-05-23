"""Plot helper for PANGEO.

Visualises the design's payoff: for each arm, the **treatment vs control
aggregate pre-period trajectories** (mean over the assigned geos). Parallel
lines = a good design. One subplot per arm, annotated with the achieved
parallel-trends R^2.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np

from .structures import PangeoResults


def plot_pangeo(
    results: PangeoResults,
    treated_color: str = "red",
    control_color: str = "black",
    save: Union[bool, str] = False,
    outcome_label: str = "Outcome",
) -> None:
    """Plot treatment vs control aggregate pre-period trajectories per arm."""
    import matplotlib.pyplot as plt

    designs = results.arm_designs
    n = len(designs)
    if n == 0:
        return
    t = results.time_labels
    fig, axes = plt.subplots(1, n, figsize=(min(5 * n, 15), 4), squeeze=False)
    for ax, (arm, d) in zip(axes[0], designs.items()):
        # Aggregate (mean) trajectory of the treatment and control geos.
        tr = np.mean([p.treatment_mean for p in d.pairs], axis=0)
        co = np.mean([p.control_mean for p in d.pairs], axis=0)
        ax.plot(t, tr, color=treated_color, lw=1.6, label="Treatment agg.")
        ax.plot(t, co, color=control_color, lw=1.6, label="Control agg.")
        ax.set_title(f"Arm {arm}  (R²={d.mean_parallelism_r2:.3f})",
                     fontsize=10)
        ax.set_xlabel("Pre-period")
        ax.grid(ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylabel(outcome_label)
    fig.suptitle("PANGEO design: pre-period parallelism by arm", fontsize=12)
    fig.tight_layout()

    if save:
        fname = save if isinstance(save, str) else "PANGEO_design.png"
        if not os.path.splitext(fname)[1]:
            fname += ".png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(fname)}")
        plt.close(fig)
    else:
        plt.show()
