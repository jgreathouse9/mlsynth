"""Plot helper for CTSC.

CTSC estimates a unit-specific treatment slope :math:`\\alpha_i` for each
treatment variable and reports the population-weighted average. Following
the paper's Figure 1 (state-specific elasticity estimates), this plots a
sorted bar chart of the per-unit slopes for each treatment variable, with
a horizontal line at the average effect (and its confidence band when
inference is available) -- surfacing the treatment-effect heterogeneity
that motivates the estimator.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np

from .structures import CTSCResults


def plot_ctsc(
    results: CTSCResults,
    treated_color: str = "black",
    counterfactual_color: str = "red",
    save: Union[bool, str, dict] = False,
    effect_label: str = "Unit-specific effect",
) -> None:
    """Sorted bar chart of per-unit slopes with the average effect line."""
    import matplotlib.pyplot as plt

    alpha = results.unit_effects                 # (n, K)
    K = alpha.shape[1]
    names = results.inputs.treatment_names
    inf = results.inference

    fig, axes = plt.subplots(1, K, figsize=(min(5 * K, 14), 4), squeeze=False)
    for k in range(K):
        ax = axes[0, k]
        vals = np.sort(alpha[:, k])
        ax.bar(np.arange(vals.size), vals, color=treated_color, alpha=0.75,
               width=0.9)
        ae = float(results.average_effect[k])
        if inf is not None and np.isfinite(inf.se[k]):
            lo, hi = ae - 1.96 * inf.se[k], ae + 1.96 * inf.se[k]
            ax.axhspan(lo, hi, color=counterfactual_color, alpha=0.15)
        ax.axhline(ae, color=counterfactual_color, lw=2,
                   label=f"avg = {ae:+.3f}")
        ax.axhline(0.0, color="grey", lw=1, ls=":")
        title = names[k] if k < len(names) else f"variable {k}"
        if inf is not None and np.isfinite(inf.p_value[k]):
            title += f"  (p={inf.p_value[k]:.3f})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Units (sorted by effect)")
        if k == 0:
            ax.set_ylabel(effect_label)
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.4)
    fig.suptitle("CTSC unit-specific treatment effects", fontsize=12)
    fig.tight_layout()

    if save:
        fname = save if isinstance(save, str) else "CTSC_unit_effects.png"
        if not os.path.splitext(fname)[1]:
            fname += ".png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(fname)}")
        plt.close(fig)
    else:
        plt.show()
