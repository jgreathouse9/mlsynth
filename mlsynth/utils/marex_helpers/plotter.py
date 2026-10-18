"""Plotting for MAREX: synthetic treated vs control (or the treatment effect)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import mlsynth_style
from .structures import MAREXResults


def plot_marex(
    results: MAREXResults,
    clusters: Optional[List[str]] = None,
    plot_type: str = "treatment",
    global_result: bool = True,
    figsize: tuple = (12, 6),
    donor_cloud: bool = False,
) -> None:
    """Plot MAREX treatment effects (or predictions), one panel per cluster + global.

    Parameters
    ----------
    results : MAREXResults
        Output of :class:`mlsynth.estimators.MAREX`.
    clusters : list of str, optional
        Cluster labels to plot (default: all; a lone ``"0"`` cluster is skipped).
    plot_type : {"treatment", "prediction"}
        Plot the treated-minus-control effect, or both synthetic series.
    global_result : bool
        Include the aggregate (global) panel.
    donor_cloud : bool
        On the global ``"prediction"`` panel, overlay one faint line per unit
        (the rows of ``results.globres.Y_full``) behind the series -- the
        "observed data" cloud of Abadie & Zhao's Figure 4. Ignored on the
        ``"treatment"`` (effect) plot and on cluster panels, which carry no
        per-unit outcome matrix. The population mean (a thick solid line) is
        drawn whenever the per-unit matrix is available, with or without the
        cloud; the synthetic treated and control are thinner dashed red / blue
        lines so they stand out against it.
    """
    with mlsynth_style({"axes.axisbelow": True}):
        if clusters is None:
            clusters = list(results.clusters.keys())
        if len(clusters) == 1 and clusters[0] == "0":
            clusters = []
        n = len(clusters) + (1 if global_result else 0)
        if n == 0:
            return
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
        if n == 1:
            axes = [axes]

        def _panel(ax, syn_t, syn_c, prefix, inf, cloud=None):
            if plot_type == "treatment":
                y = syn_t - syn_c
                ax.plot(y, label=prefix)
                if inf is not None and inf.ci is not None:
                    ci = inf.ci
                    ax.fill_between(np.arange(len(y)), ci[:, 0], ci[:, 1], alpha=0.2)
                ax.set_ylabel("Treatment effect")
            else:
                if cloud is not None and cloud.shape[0] > 0:
                    if donor_cloud:
                        for series in cloud:
                            ax.plot(series, color="0.8", lw=0.6, zorder=1)
                    ax.plot(cloud.mean(axis=0), color="black", lw=2.4, ls="-",
                            label="Population mean", zorder=2)
                ax.plot(syn_t, color="#d62728", ls="--", lw=1.6,
                        label=f"{prefix} treated", zorder=4)
                ax.plot(syn_c, color="#1f77b4", ls="--", lw=1.6,
                        label=f"{prefix} control", zorder=4)
                ax.set_ylabel("Outcome")
            ax.legend()

        i = 0
        for cid in clusters:
            c = results.clusters[cid]
            _panel(axes[i], c.synthetic_treated, c.synthetic_control, f"Cluster {cid}", c.inference)
            i += 1
        if global_result:
            g = results.globres
            _panel(axes[i], g.synthetic_treated, g.synthetic_control, "Synthetic", g.inference,
                   cloud=g.Y_full)

        T0, bp = results.study.T0, results.study.blank_periods
        if bp > 0:
            for ax in axes:
                ax.axvspan(T0 - bp, T0, color="gray", alpha=0.2)
        axes[-1].set_xlabel("Time")
        plt.suptitle("MAREX " + ("Treatment Effects" if plot_type == "treatment" else "Predictions"))
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
