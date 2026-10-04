"""Figures for TBR. Helpers return the Figure; the caller displays or saves it."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_tbr(result, *, title: Optional[str] = None):
    """Three panels: the two series, the per-period gap, the cumulative band.

    Section 3.3's figure. The cumulative panel is the one the method exists for,
    so it carries the posterior band.
    """
    est = getattr(result, "report", result)
    ts = est.time_series
    observed = np.asarray(ts.observed_outcome, dtype=float).ravel()
    counterfactual = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()
    periods = np.asarray(ts.time_periods).ravel()
    cum = est.cumulative
    post = np.asarray(cum.periods).ravel()

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(periods, observed, label="treatment group")
    axes[0].plot(periods, counterfactual, linestyle="--",
                 label="counterfactual")
    axes[0].set_ylabel("outcome")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, linewidth=0.8, color="black")
    axes[1].plot(periods, observed - counterfactual)
    axes[1].set_ylabel("pointwise difference")

    axes[2].axhline(0.0, linewidth=0.8, color="black")
    axes[2].plot(post, np.asarray(cum.estimate, dtype=float))
    axes[2].fill_between(post, np.asarray(cum.lower, dtype=float),
                         np.asarray(cum.upper, dtype=float), alpha=0.25)
    axes[2].set_ylabel(f"cumulative effect ({cum.level:.0%})")
    axes[2].set_xlabel("period")

    if est.cooldown_periods:
        for ax in axes:
            ax.axvline(post[est.intervention_periods - 1], linewidth=0.8,
                       linestyle=":", color="grey")

    axes[0].set_title(title or "TBR")
    fig.tight_layout()
    return fig
