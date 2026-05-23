"""Plot helper for ISCM.

ISCM identifies the treatment effect by pooling per-unit estimates
:math:`\\widehat\\alpha_i` across the units that carry identifying
variation (those whose synthetic control places weight on a treated
unit), each weighted by its contribution :math:`v_i`. A treated unit's
*own* synthetic control may fit poorly -- by design, since ISCM does not
rely on it -- so the canonical observed-vs-counterfactual trajectory
would be misleading. Instead this plots the paper's native summary
(cf. Table 1 / state-specific estimates): each contributing unit's
estimate, sized by its contribution, with the aggregate ATT (and
confidence interval, if available) marked.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np

from .structures import ISCMResults


def plot_iscm(
    results: ISCMResults,
    treated_color: str = "black",
    counterfactual_color: str = "red",
    save: Union[bool, str, dict] = False,
    unit_label: str = "Unit",
    effect_label: str = "Treatment effect estimate",
) -> None:
    """Per-unit effect dot plot with the aggregate ATT and CI."""
    import matplotlib.pyplot as plt

    inputs = results.inputs
    contributing = np.where(
        np.isfinite(results.unit_att) & (results.contribution > 1e-12)
    )[0]
    if contributing.size == 0:
        return

    order = contributing[np.argsort(-results.contribution[contributing])]
    est = results.unit_att[order]
    contrib = results.contribution[order]
    names = [str(inputs.unit_names[i]) for i in order]
    # Marker sizes scale with contribution share.
    sizes = 40 + 360 * (contrib / contrib.max())

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * len(order) + 1)))
    y = np.arange(len(order))[::-1]   # highest contribution at top

    if results.inference is not None and np.isfinite(results.inference.ci[0]):
        lo, hi = results.inference.ci
        ax.axvspan(lo, hi, color=counterfactual_color, alpha=0.12,
                   label="ATT 95% CI")
    ax.axvline(results.att, color=counterfactual_color, lw=2,
               label=f"ATT = {results.att:+.3f}")
    ax.axvline(0.0, color="grey", lw=1, ls=":")
    ax.scatter(est, y, s=sizes, color=treated_color, alpha=0.8,
               edgecolor="white", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(effect_label)
    ax.set_ylabel(unit_label)
    ax.set_title("ISCM per-unit estimates (marker size = contribution)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", ls="--", alpha=0.4)
    fig.tight_layout()

    if save:
        fname = save if isinstance(save, str) else "ISCM_unit_effects.png"
        if not os.path.splitext(fname)[1]:
            fname += ".png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(fname)}")
        plt.close(fig)
    else:
        plt.show()
