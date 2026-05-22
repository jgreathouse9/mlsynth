"""Plot helpers for MicroSynth.

Two diagnostics:

* **Love plot**: per-covariate SMD before and after weighting.
  The standard balance diagnostic that marketing-science folks
  recognize from propensity-score work.
* **Lift trajectory**: per-post-period gap with a bootstrap band
  (only meaningful when ``T_post > 1``).
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

from .structures import MicroSynthResults


def plot_microsynth(
    results: MicroSynthResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
) -> None:
    """Render the love plot + (if applicable) the lift trajectory.

    Uses matplotlib lazily so that the module imports cleanly even
    when matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-dep
        raise ImportError(
            "matplotlib is required for MicroSynth plotting"
        ) from exc

    design = results.design
    inputs = results.inputs

    fig, axes = plt.subplots(
        1, 2 if inputs.T_post > 1 else 1,
        figsize=(12 if inputs.T_post > 1 else 6, 5),
    )
    ax_love = axes[0] if inputs.T_post > 1 else axes

    # ---- Love plot ----
    n = len(inputs.covariate_names)
    y_pos = np.arange(n)
    ax_love.scatter(
        design.smd_before, y_pos,
        marker="o", color="lightgray", s=60, label="Before weighting",
    )
    ax_love.scatter(
        design.smd_after, y_pos,
        marker="s", color=treated_color, s=40, label="After weighting",
    )
    ax_love.axvline(0.0, color="black", linewidth=0.5)
    ax_love.axvline(0.1, color="red", linestyle=":", linewidth=0.5)
    ax_love.axvline(-0.1, color="red", linestyle=":", linewidth=0.5)
    ax_love.set_yticks(y_pos)
    ax_love.set_yticklabels(inputs.covariate_names)
    ax_love.set_xlabel("Standardized mean difference")
    ax_love.set_title("Balance: treated vs. (weighted) controls")
    ax_love.legend(loc="best", fontsize=9)
    ax_love.grid(axis="x", alpha=0.3)

    # ---- Lift trajectory ----
    if inputs.T_post > 1:
        ax_traj = axes[1]
        cf_color = (
            counterfactual_color if isinstance(counterfactual_color, str)
            else counterfactual_color[0]
        )
        horizons = np.arange(inputs.T_post)
        ax_traj.plot(horizons, results.gap_trajectory,
                     marker="o", color=cf_color, linewidth=1.5,
                     label="ATT per post-period")
        ax_traj.axhline(0.0, color="black", linewidth=0.5)
        ax_traj.axhline(results.att, color="gray", linestyle="--",
                        linewidth=0.7, label=f"Mean ATT = {results.att:.4f}")
        ax_traj.set_xlabel("Post-treatment period (index)")
        ax_traj.set_ylabel("Treated - weighted control")
        ax_traj.set_title("Lift trajectory")
        ax_traj.legend(loc="best", fontsize=9)
        ax_traj.grid(alpha=0.3)

    fig.tight_layout()

    if isinstance(save, str):
        fig.savefig(save, dpi=150, bbox_inches="tight")
    elif isinstance(save, dict):
        path = save.get("path", "microsynth.png")
        dpi = save.get("dpi", 150)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
