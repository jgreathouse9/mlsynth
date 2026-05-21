"""Plotting helper for PPSCM."""

from __future__ import annotations

from typing import Union

import numpy as np

from .structures import PPSCMResults


def plot_ppscm(
    results: PPSCMResults,
    title: str = "Partially Pooled SCM event study",
    save: Union[bool, str, dict] = False,
) -> None:
    """Render the per-horizon ATT trajectory with the jackknife CI band."""
    import matplotlib.pyplot as plt

    es = results.event_study
    fig, ax = plt.subplots()
    ax.fill_between(es.horizons, es.ci[:, 0], es.ci[:, 1], alpha=0.2)
    ax.plot(es.horizons, es.tau, marker="o")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Event-time horizon $k$")
    ax.set_ylabel(r"$\widehat{\mathrm{ATT}}_k$")
    ax.set_title(title)
    if save:
        filename = save if isinstance(save, str) else "ppscm_event_study.png"
        fig.savefig(filename, bbox_inches="tight")
    plt.show()
