"""Two-panel diagnostic plot for DSC: counterfactual + per-period gap.

Mirrors Figures 3-5 of Zheng & Chen (2024).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import DSCARResults


def plot_dsc(
    results: DSCARResults,
    *,
    treated_color: str = "tab:blue",
    counterfactual_color: str = "tab:red",
    save: Optional[str] = None,
) -> None:
    try:
        inputs = results.inputs
        fit = results.fit
        t = np.arange(1, inputs.T + 1)
        T0 = inputs.T0

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # (a) trajectories
        ax = axes[0]
        ax.plot(t, fit.Y_treated_mean, "-", color=treated_color, lw=2,
                label="Observed (treated mean)")
        ax.plot(t, fit.Y0_hat, "--", color=counterfactual_color, lw=2,
                label="DSC counterfactual")
        ax.axvline(T0 + 0.5, color="grey", ls=":", alpha=0.8)
        ax.set_xlabel("period")
        ax.set_ylabel(inputs.y_name)
        ax.set_title(f"DSC: observed vs counterfactual (n_treated={inputs.n_treated})")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

        # (b) per-period gap
        ax = axes[1]
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.plot(t, fit.gap, "-o", color=treated_color, lw=1.5, ms=3)
        ax.axvline(T0 + 0.5, color="grey", ls=":", alpha=0.8)
        ax.set_xlabel("period")
        ax.set_ylabel("treatment effect")
        title = f"ATT = {fit.att:+.3f}"
        if fit.se is not None:
            title += f" (SE {fit.se:.3f})"
        if fit.att_relative == fit.att_relative:                  # not NaN
            title += f"   |  rel. = {100 * fit.att_relative:+.1f}%"
        ax.set_title(title)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        if save:
            fig.savefig(save, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
    except Exception as exc:                                # pragma: no cover
        raise MlsynthPlottingError(f"DSC plotter failed: {exc}") from exc
