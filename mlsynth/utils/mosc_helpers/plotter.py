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


_INTERVAL_NAMES = {
    "unit_bootstrap": "bootstrap interval",
    "posterior_mean_band": "credible band (conditional mean)",
}


def plot_mosc_posterior(
    results: "MOSCResults",
    title: Optional[str] = None,
    treated_color: str = "black",
    counterfactual_color: str = "tab:blue",
) -> "Figure":
    """Observed path against the counterfactual, with the interval the fit produced.

    The legend names the interval by what it is. Under the default unit
    bootstrap it is a percentile confidence interval; under
    ``inference="posterior"`` it is a credible band on the conditional mean.
    Labelling one as the other would misdescribe every figure.

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
                    label=f"{100 * (1 - detail.ci_alpha):.0f}% {_INTERVAL_NAMES[detail.method]}")
    if pre < len(labels):
        ax.axvline(labels[pre], color="grey", lw=1)

    _thin_time_ticks(ax, labels)
    ax.set_xlabel("Time")
    ax.set_ylabel(results.inputs.treated_unit_name)
    ax.set_title(title or f"MOSC ({results.posterior.factor_model})")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _thin_time_ticks(ax, labels: np.ndarray, most: int = 8) -> None:
    """Show at most ``most`` time labels, so a long daily panel stays readable.

    A categorical axis draws one tick per period, which for a panel of a few
    hundred days is a solid bar of overlapping text. Matplotlib's date locators
    do not apply here because the labels are whatever the caller's time column
    held, which need not be a date at all.
    """
    if len(labels) <= most:
        return
    step = int(np.ceil(len(labels) / most))
    positions = labels[::step]
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], rotation=45, ha="right")
