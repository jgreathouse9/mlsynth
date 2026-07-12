"""Event-study plot for the SCD estimator.

Draws the per-period effect path :math:`\\hat\\theta_t` with the confidence
band (weight confidence set plus the pointwise term), a treatment reference
line, and a zero line -- the natural SCD read.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...config_models import BaseEstimatorResults


def plot_scd(
    results: BaseEstimatorResults,
    *,
    treated_color: str = "black",
    counterfactual_color: str = "C0",
    save: Any = False,
    outcome_label: str = "outcome",
    ax: Optional[Any] = None,
):
    """Render the SCD event study from a fitted :class:`BaseEstimatorResults`."""
    import matplotlib.pyplot as plt

    ts = results.time_series
    periods = np.asarray(ts.time_periods)
    theta = np.asarray(ts.estimated_gap)
    intervention = ts.intervention_time

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10, 5.5))

    ax.axhline(0.0, color="#888", lw=1.0, zorder=1)
    if intervention is not None:
        ax.axvline(intervention, color="#c0392b", lw=1.3, ls="--", zorder=2,
                   label=f"treatment ({intervention})")

    inf = results.inference
    if inf is not None and inf.details is not None and "lower" in inf.details:
        lo = np.asarray(inf.details["lower"])
        hi = np.asarray(inf.details["upper"])
        level = int(round((results.inference.confidence_level or 0.9) * 100))
        ax.fill_between(periods, lo, hi, color=counterfactual_color, alpha=0.22,
                        zorder=2, label=f"{level}% band")

    ax.plot(periods, theta, color=treated_color, lw=1.8, marker="o", ms=3,
            zorder=3, label=r"$\hat\theta_t$")
    ax.set_xlabel("period")
    ax.set_ylabel(f"effect on {outcome_label}")
    ax.set_title("SCD event study")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    if save:
        fname = save if isinstance(save, str) else "scd_event_study.png"
        ax.figure.savefig(fname, dpi=140, bbox_inches="tight")
    if created:
        plt.tight_layout()
    return ax
