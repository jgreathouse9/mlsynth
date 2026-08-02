"""Plot FSC output: the observed and counterfactual objects, and the effect.

A scalar synthetic control plots one line against time. Here every period holds
a whole object, so the natural display is a small multiple: one panel per
period, the argument on the horizontal axis, the observed and counterfactual
objects overlaid, and the conformal band shaded on the post-treatment panels.
That is the layout the paper's own figures use.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def plot_fsc(results, argument_label: str = "argument",
             outcome_label: str = "outcome", max_panels: int = 12,
             save: Union[bool, str] = False, display: bool = True) -> None:
    """Small multiple of observed vs counterfactual objects, one panel a period.

    Shows the last ``max_panels`` periods so the post-treatment ones are always
    visible; a long pre-period is summarised by the fit statistics rather than
    by dozens of near-identical panels.
    """
    import matplotlib.pyplot as plt

    curves = list(results.curves)[-int(max_panels):]
    grid = np.asarray(results.grid, dtype=float)
    n = len(curves)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow),
                             squeeze=False, sharex=True)

    for ax, curve in zip(axes.ravel(), curves):
        if curve.lower is not None and curve.is_post:
            ax.fill_between(grid, curve.lower, curve.upper, color="#c8d8e8",
                            alpha=0.7, lw=0)
        ax.plot(grid, curve.observed, color="black", lw=1.6, label="observed")
        ax.plot(grid, curve.counterfactual, color="#c0392b", lw=1.5, ls="--",
                label="synthetic")
        ax.set_title(f"{curve.time}{'' if curve.is_post else ' (pre)'}",
                     fontsize=9)
        ax.grid(alpha=0.22)
        ax.set_xlabel(argument_label)
        ax.set_ylabel(outcome_label)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()

    if save:
        fig.savefig(f"{save}.png" if isinstance(save, str) else "fsc.png",
                    dpi=150, bbox_inches="tight")
    if display:
        plt.show()
    plt.close(fig)
