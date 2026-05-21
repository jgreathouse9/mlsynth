"""Event-study chart for Sequential SDiD.

Plots ``tau_hat_k^SSDiD`` against the event-time horizon ``k``, with the
bootstrap Wald band as a shaded region. Reuses matplotlib directly to
stay independent of the existing SDID_plot helper, which is wired for the
canonical-SDID dict shape.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .structures import SeqSDIDResults


def plot_seq_sdid(
    results: SeqSDIDResults,
    title: str = "Sequential SDiD event study",
    save: Union[bool, str, dict] = False,
) -> None:
    """Render the event-study trajectory with the bootstrap CI band."""

    import matplotlib.pyplot as plt

    es = results.event_study
    fig, ax = plt.subplots()
    ax.fill_between(es.horizons, es.ci[:, 0], es.ci[:, 1], alpha=0.2)
    ax.plot(es.horizons, es.tau, marker="o")
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Event-time horizon $k$")
    ax.set_ylabel(r"$\hat\tau_k^{\,SSDiD}$")
    ax.set_title(title)

    if save:
        filename = save if isinstance(save, str) else "seq_sdid_event_study.png"
        fig.savefig(filename, bbox_inches="tight")
    plt.show()
