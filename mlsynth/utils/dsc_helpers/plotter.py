"""Plot helper for DSC (Distributional Synthetic Control).

DSC compares the treated unit's *outcome distribution* to a synthetic
control distribution. The natural visualization is two-panel:

1. the post-period-averaged observed vs synthetic **quantile functions**
   (the distributions being compared); and
2. the **quantile treatment effect (QTE) curve** -- the gap at each
   quantile -- with the per-period curves overlaid faintly and the
   average effect (ATT) marked.

This surfaces distributional heterogeneity (e.g. an effect concentrated
in the upper tail) that a single ATT would hide.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np

from .structures import DSCResults


def plot_dsc(
    results: DSCResults,
    treated_color: str = "black",
    counterfactual_color: str = "red",
    save: Union[bool, str, dict] = False,
    outcome_label: str = "Outcome",
) -> None:
    """Quantile functions (observed vs synthetic) and the QTE curve."""
    import matplotlib.pyplot as plt

    curves = results.qte_curves
    if not curves:
        return
    q = curves[0].quantiles
    observed = np.mean([c.observed for c in curves], axis=0)
    counterfactual = np.mean([c.counterfactual for c in curves], axis=0)
    avg_qte = results.average_qte

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: averaged quantile functions.
    ax1.plot(q, observed, color=treated_color, lw=2, label="Observed")
    ax1.plot(q, counterfactual, color=counterfactual_color, lw=2, ls="--",
             label="Synthetic")
    ax1.set_xlabel("Quantile")
    ax1.set_ylabel(outcome_label)
    ax1.set_title("Post-period quantile functions")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(ls="--", alpha=0.4)

    # Panel 2: QTE curve (+ per-period curves, faint).
    for c in curves:
        ax2.plot(q, c.qte, color=treated_color, lw=0.8, alpha=0.18)
    ax2.plot(q, avg_qte, color=counterfactual_color, lw=2.5,
             label="Average QTE")
    ax2.axhline(results.att, color=treated_color, lw=1.5, ls="-.",
                label=f"ATT = {results.att:+.3f}")
    ax2.axhline(0.0, color="grey", lw=1, ls=":")
    ax2.set_xlabel("Quantile")
    ax2.set_ylabel(f"Quantile treatment effect on {outcome_label}")
    ax2.set_title("Quantile treatment effects")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(ls="--", alpha=0.4)

    fig.suptitle(
        f"DSC: {results.inputs.treated_unit_name}", fontsize=12
    )
    fig.tight_layout()

    if save:
        fname = save if isinstance(save, str) else "DSC_qte.png"
        if not os.path.splitext(fname)[1]:
            fname += ".png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(fname)}")
        plt.close(fig)
    else:
        plt.show()
