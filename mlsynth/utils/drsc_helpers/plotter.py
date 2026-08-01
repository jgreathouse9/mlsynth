"""Plot the conditional distributional treatment effect."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def plot_drsc(results, outcome_label: str = "outcome",
              focus_region=None, save: Union[bool, str] = False):
    """One panel per evaluation point: ``Delta_t(y | x)`` over the grid."""
    import matplotlib.pyplot as plt

    eff = results.conditional_effects
    grid = np.asarray(results.outcome_grid, dtype=float)
    n = len(eff)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax, (label, e) in zip(axes.ravel(), eff.items()):
        if focus_region is not None:
            ax.axvspan(focus_region[0], focus_region[1], color="#f5d76e",
                       alpha=0.35, lw=0)
        ax.plot(grid, e.delta, color="#1f4e79", lw=1.8)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title(f"{label}   $\\hat f$={e.f_hat:.2e}, p={e.p_value:.3f}",
                     fontsize=9)
        ax.set_xlabel(outcome_label)
        ax.set_ylabel(r"$\hat\Delta_t(y \mid x)$")
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(f"{save}.png" if isinstance(save, str) else "drsc.png",
                    dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
