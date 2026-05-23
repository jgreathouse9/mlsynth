"""Plot helper for PANGEO.

Visualises the design through its estimator: for each arm, the **observed
treated supergeo aggregate** against the **augmented-DiD counterfactual
prediction** (Li & Van den Bulte 2022). When the panel carries post-period
data the counterfactual extends past the treatment date, so the gap in the
post window is the per-period treatment effect; otherwise the in-sample
pre-period fit is shown. One subplot per arm.
"""

from __future__ import annotations

import os
from typing import Union

import numpy as np

from .effects import adid_counterfactual
from .structures import PangeoResults


def plot_pangeo(
    results: PangeoResults,
    treated_color: str = "red",
    counterfactual_color: str = "black",
    save: Union[bool, str] = False,
    outcome_label: str = "Outcome",
) -> None:
    """Plot observed treated vs augmented-DiD counterfactual per arm."""
    import matplotlib.pyplot as plt

    designs = results.arm_designs
    n = len(designs)
    if n == 0:
        return
    augment = bool(results.metadata.get("att_augment", True))
    trend = bool(results.metadata.get("att_trend", True))
    effects = results.effects

    fig, axes = plt.subplots(1, n, figsize=(min(5 * n, 15), 4), squeeze=False)
    for ax, (arm, d) in zip(axes[0], designs.items()):
        est = effects.arms.get(arm) if effects is not None else None
        if est is not None and np.asarray(est.observed).size:
            # Realised: observed treated vs counterfactual over pre + post.
            observed = np.asarray(est.observed)
            counterfactual = np.asarray(est.counterfactual)
            n_pre = observed.size - est.n_post
            title = f"Arm {arm}  (ATT={est.att_pct:.1f}%, p={est.p_value:.3f})"
        else:
            # Design only: treated-size-weighted pre aggregates + in-sample fit.
            w = np.array([len(p.treatment) for p in d.pairs], dtype=float)
            w = w / w.sum()
            observed = w @ np.vstack([p.treatment_mean for p in d.pairs])
            control = w @ np.vstack([p.control_mean for p in d.pairs])
            counterfactual = adid_counterfactual(
                observed, control, observed.size, augment, trend)
            n_pre = observed.size
            title = f"Arm {arm}  (R²={d.mean_parallelism_r2:.3f})"

        x = np.arange(observed.size)
        ax.plot(x, observed, color=treated_color, lw=1.6,
                label="Treated (observed)")
        ax.plot(x, counterfactual, color=counterfactual_color, lw=1.6,
                ls="--", label="Augmented-DiD counterfactual")
        if n_pre < observed.size:
            ax.axvline(n_pre - 0.5, color="grey", ls=":", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Period")
        ax.grid(ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylabel(outcome_label)
    title = ("PANGEO: observed treated vs augmented-DiD counterfactual"
             if effects is not None
             else "PANGEO design: treated vs counterfactual (pre-period fit)")
    fig.suptitle(title, fontsize=12)
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
