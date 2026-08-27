"""MOSC's diagnostic figure.

The plotter builds and returns a ``Figure``: that is mechanism. Showing it,
saving it and choosing a filename are policy, and belong to whoever is driving
the estimator. The standard observed-against-counterfactual panel is already on
the result as ``result.plot()``; this adds the posterior band, which is the part
the contract's shared plotter cannot know about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from matplotlib.figure import Figure

    from .structures import MOSCResults


def plot_mosc_posterior(
    results: "MOSCResults",
    title: Optional[str] = None,
    treated_color: str = "black",
    counterfactual_color: str = "tab:blue",
) -> "Figure":
    """Observed path against the posterior counterfactual and its credible band.

    Returns
    -------
    matplotlib.figure.Figure
        The caller displays or saves it.
    """
    import matplotlib.pyplot as plt

    detail = results.inference_detail
    labels = np.asarray(results.inputs.time_labels)
    observed = np.asarray(results.inputs.y_target, dtype=float)
    pre = results.inputs.pre_periods

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels, observed, color=treated_color, lw=2, label="Observed")
    ax.plot(labels, detail.counterfactual_mean, color=counterfactual_color,
            lw=2, ls="--", label="Posterior mean counterfactual")
    ax.fill_between(labels, detail.counterfactual_lower, detail.counterfactual_upper,
                    color=counterfactual_color, alpha=0.15,
                    label=f"{100 * (1 - detail.ci_alpha):.0f}% credible interval")
    if pre < len(labels):
        ax.axvline(labels[pre], color="grey", lw=1)

    ax.set_xlabel("Time")
    ax.set_ylabel(results.inputs.treated_unit_name)
    ax.set_title(title or f"MOSC ({results.posterior.factor_model})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
