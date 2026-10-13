"""Plotting for EASC: the counterfactuals, the silent-factor diagnostic, and the frontier.

:func:`plot_easc` builds and returns its figure. Displaying and saving are the
caller's: the estimator applies ``display_graphs`` and ``save``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...config_models import PlotConfig
from ...exceptions import MlsynthPlottingError
from .structures import EASCResults

_LABELS = {"easc": "EASC", "ebsc_min": "EBSC-min", "easc_tuned": "EASC*"}


def plot_easc(results: EASCResults, plot_config: Optional[PlotConfig] = None):
    """Three panels: observed against the SC and headline counterfactuals; the
    partial leverage ``ell_t`` of the shock with the +/-2 band; and the
    exposure-fit frontier with its EBSC-min, EASC* and SC points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a dependency
        raise MlsynthPlottingError(f"EASC: plotting unavailable: {exc}") from exc

    pc = plot_config or results.plot_config or PlotConfig()
    inp = results.inputs
    x = np.asarray(inp.time_labels)
    t0 = x[inp.T0] if inp.T0 < inp.T else None
    head = results.fits[results.method]
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))

    ax0.plot(x, inp.y, color=pc.observed_color, lw=pc.observed_linewidth, label="Observed")
    ax0.plot(x, results.fits["sc"].counterfactual, color="grey", ls=":", lw=1.2, label="Synthetic control")
    ax0.plot(x, head.counterfactual, color=pc.counterfactual_colors[0], ls=pc.counterfactual_linestyle,
             lw=pc.counterfactual_linewidth, label=_LABELS[results.method])
    ax1.bar(x, results.diagnostics.partial_leverage, color="tab:blue")
    for level in (-2.0, 2.0):
        ax1.axhline(level, color="grey", ls="--", lw=0.8)
    for ax in (ax0, ax1):
        if t0 is not None and pc.show_intervention_line:
            ax.axvline(t0, color=pc.intervention_color, ls=":", lw=1.0)
        ax.set_xlabel(pc.xlabel or "Time")
    ax0.set_ylabel(pc.ylabel or inp.metadata.get("outcome", "Outcome"))
    ax0.set_title(pc.title or "Observed and counterfactuals")
    ax0.legend(frameon=False)
    ax1.set_title(f"Partial leverage of the shock (pre-period R$^2$ = {results.diagnostics.shock_r2:.2f})")

    fr = results.frontier
    ei = np.abs(fr.exposure_imbalance)
    ax2.plot(ei, fr.rmspe, marker=".", color="tab:blue")
    for idx, label in ((0, "EBSC-min"), (fr.selected, "EASC*"), (-1, "SC")):
        ax2.annotate(label, (ei[idx], fr.rmspe[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax2.set_xlabel("Exposure imbalance |EI(w)|")
    ax2.set_ylabel("Pre-treatment RMSPE")
    ax2.set_title("Exposure-fit frontier")
    fig.tight_layout()
    return fig
