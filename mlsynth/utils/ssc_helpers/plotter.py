"""Event-study plot for the SSC estimator.

SSC's natural visual is the event-time ATT path: the average treatment effect
by periods since adoption, with its end-of-sample band.
"""

from __future__ import annotations

import os
from typing import List, Union

import numpy as np

from .structures import SSCResults


def plot_ssc(
    results: SSCResults,
    treated_color: str = "black",
    counterfactual_color: Union[str, List[str]] = "red",
    save: Union[bool, str, dict] = False,
    time_axis_label: str = "Event time",
    treatment_label: str = "Treatment",
    unit_label: str = "Unit",
    outcome_label: str = "Outcome",
) -> None:
    """Draw the event-time ATT path with its Andrews band."""
    import matplotlib.pyplot as plt

    es = results.event_att
    if not es:
        return
    events = sorted(es)
    point = np.array([es[e] for e in events], dtype=float)
    cf = counterfactual_color if isinstance(counterfactual_color, str) \
        else (counterfactual_color[0] if counterfactual_color else "red")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="grey", lw=1, ls=":")
    if results.event_bands:
        lo = np.array([results.event_bands[e].lower for e in events])
        hi = np.array([results.event_bands[e].upper for e in events])
        ax.fill_between(events, lo, hi, color=cf, alpha=0.20,
                        label="end-of-sample band")
    ax.plot(events, point, "-o", color=cf, ms=4, lw=1.8, label="event-time ATT")
    ax.set_xlabel(time_axis_label)
    ax.set_ylabel(f"Effect on {outcome_label}")
    title = f"SSC event study (overall ATT = {results.att:+.3f}"
    if results.att_band is not None:
        title += f", p = {results.att_band.p_value:.3f}"
    ax.set_title(title + ")")
    ax.legend(loc="best", fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()

    if save:
        fname = save if isinstance(save, str) else "SSC_event_study.png"
        if not os.path.splitext(fname)[1]:
            fname += ".png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(fname)}")
        plt.close(fig)
    else:
        plt.show()
