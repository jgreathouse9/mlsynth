"""Plot helper for ATEL.

Two panels, because the localization is the point. The upper panel is the usual
observed-versus-counterfactual path with the pointwise band over the
post-period; the lower panel is the localization weight on each post-period, so
a reader can see which periods the reported estimate is actually made of.

The helper builds a figure and returns it. Displaying, saving and closing belong
to the caller.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .structures import ATELResults

__all__ = ["plot_atel"]


def plot_atel(
    results: ATELResults,
    observed_color: str = "black",
    counterfactual_color: str = "red",
    outcome_label: str = "Outcome",
    time_axis_label: str = "Time",
    figsize: Optional[tuple] = None,
):
    """Observed versus counterfactual, with the localization weights beneath.

    Parameters
    ----------
    results : ATELResults
        A fitted result.
    observed_color, counterfactual_color : str
        Line colors.
    outcome_label, time_axis_label : str
        Axis labels.
    figsize : tuple, optional
        Figure size; defaults to ``(9, 7)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    inputs = results.inputs
    observed = np.asarray(results.time_series.observed_outcome, dtype=float).ravel()
    counterfactual = np.asarray(
        results.time_series.counterfactual_outcome, dtype=float
    ).ravel()
    n_post = int(results.kernel_weights.size)
    n_pre = observed.size - n_post
    labels = (
        np.asarray(inputs.time_labels)
        if inputs is not None
        else np.arange(observed.size)
    )

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=figsize or (9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    top.plot(labels, observed, color=observed_color, linewidth=2.0, label="Observed")
    top.plot(
        labels, counterfactual, color=counterfactual_color, linestyle="--",
        linewidth=2.0, label="ATEL counterfactual",
    )
    band = np.asarray(results.pointwise_standard_errors, dtype=float).ravel()
    if band.size == n_post:
        top.fill_between(
            labels[n_pre:],
            counterfactual[n_pre:] - 1.96 * band,
            counterfactual[n_pre:] + 1.96 * band,
            color=counterfactual_color, alpha=0.18, linewidth=0,
            label="95% pointwise band",
        )
    if n_pre < labels.size:
        top.axvline(labels[n_pre], color="grey", linestyle=":", linewidth=1.2)
    top.set_ylabel(outcome_label)
    top.set_title(
        f"ATEL = {results.atel:.4f} "
        f"(SE {results.inference.standard_error:.4f}, h = {results.bandwidth:g})"
    )
    top.legend(frameon=False, loc="best")

    bottom.bar(
        labels[n_pre:], np.asarray(results.kernel_weights, dtype=float),
        color=counterfactual_color, alpha=0.55,
    )
    bottom.set_ylabel("Kernel weight")
    bottom.set_xlabel(time_axis_label)

    fig.tight_layout()
    return fig
