"""Plotting helper for PPSCM: the relative-time event study with CI band."""

from __future__ import annotations

from typing import Union

from ..plotting import mlsynth_style
from .structures import PPSCMResults


def plot_ppscm(
    results: PPSCMResults,
    title: str = "Partially Pooled SCM event study",
    save: Union[bool, str, dict] = False,
) -> None:
    """Render the per-horizon (time-since-treatment) ATT with jackknife CIs."""
    import matplotlib.pyplot as plt

    es = results.event_study
    with mlsynth_style():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.fill_between(es.horizons, es.ci[:, 0], es.ci[:, 1], alpha=0.2,
                        color="#1428A0", label="95% jackknife CI")
        ax.plot(es.horizons, es.tau, marker="o", color="#1428A0",
                label=f"ATT (avg {results.att:.3f})")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time since treatment")
        ax.set_ylabel(r"$\widehat{\mathrm{ATT}}_k$")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        if save:
            fname = save if isinstance(save, str) else "ppscm_event_study.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
