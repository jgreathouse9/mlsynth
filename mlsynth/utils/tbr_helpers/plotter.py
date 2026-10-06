"""Figures for TBR. Helpers return the Figure; the caller displays or saves it."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_tbr(result, *, title: Optional[str] = None):
    """Three panels: the two series, the per-period effect, the cumulative one.

    Section 3.3's figure. Both effect panels carry their posterior band, which
    is what the figure shows: a per-period effect drawn without one cannot be
    read, since an effect indistinguishable from zero looks like one that is
    not. The per-period band is on ``result.pointwise``; a result predating
    that field draws the gap alone.
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
    pw = getattr(est, "pointwise", None)
    if pw is not None:
        axes[1].fill_between(np.asarray(pw.periods).ravel(),
                             np.asarray(pw.lower, dtype=float),
                             np.asarray(pw.upper, dtype=float), alpha=0.2)
        axes[1].plot(np.asarray(pw.periods).ravel(),
                     np.asarray(pw.estimate, dtype=float))
        axes[1].set_ylabel(f"pointwise effect ({pw.level:.0%})")
    else:                               # pragma: no cover - a result without the field
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
