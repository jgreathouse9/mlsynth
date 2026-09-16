"""Plotting helper for PPSCM: the relative-time event study with CI band."""

from __future__ import annotations

from typing import Union

from ..plotting import mlsynth_style
from .structures import PPSCMResults

#: Display name per ``inference_method``. The band's label is read from the
#: fit and not written into the plotter, because ``inference_method`` chooses
#: between three intervals and ``method='callaway_santanna'`` switches it to
#: the influence function without the caller naming it -- so a fixed label
#: reports whichever interval the plotter was written for.
CI_LABELS = {
    "jackknife": "jackknife",
    "bootstrap": "bootstrap",
    "influence_function": "influence-function",
}


def plot_ppscm(
    results: PPSCMResults,
    title: str = "Partially Pooled SCM event study",
    save: Union[bool, str, dict] = False,
) -> None:
    """Render the per-horizon (time-since-treatment) ATT, with its interval."""
    import matplotlib.pyplot as plt

    es = results.event_study
    conventions = getattr(results.design, "conventions", None) or {}
    inference_method = conventions.get("inference_method")
    with mlsynth_style():
        fig, ax = plt.subplots(figsize=(7, 5))
        # With inference off, ``ci`` is the point path twice over; drawing that
        # zero-width band would put a confidence statement on the figure that
        # no procedure produced.
        if inference_method is not None:
            name = CI_LABELS.get(inference_method, inference_method)
            ax.fill_between(es.horizons, es.ci[:, 0], es.ci[:, 1], alpha=0.2,
                            color="#1428A0", label=f"95% {name} CI")
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
