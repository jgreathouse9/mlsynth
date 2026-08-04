"""Diagnostic plot for GSYNTH results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import GSYNTHResults


def plot_gsynth(results: GSYNTHResults) -> None:
    """Two panels: the treated-average paths, and the effect by horizon.

    With staggered adoption the left panel is in calendar time and the right in
    time since adoption, which is where the treated units line up.
    """
    inputs = results.inputs
    design = results.design
    inf = results.inference_detail
    es = results.event_study
    ts = results.time_series

    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:  # pragma: no cover - setup always aligns the labels
        t = np.arange(inputs.T)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        ax.plot(t, ts.observed_outcome, "k-", lw=2, label="treated average")
        ax.plot(t, ts.counterfactual_outcome, "r--", lw=2,
                label="imputed untreated")
        for a in np.unique(inputs.adoption_index):
            ax.axvline(t[int(a)], color="grey", ls=":", alpha=0.6)
        ax.set_xlabel("time")
        ax.set_ylabel("outcome")
        ax.set_title(
            f"GSYNTH (r = {design.r_selected}, source = {design.r_source})")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

        ax = axes[1]
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.axvline(-0.5, color="grey", ls=":", alpha=0.7)
        ax.plot(es.horizons, es.tau, "b-", lw=2, label="ATT by horizon")
        if inf.tau_lower.size == es.tau.size:
            ax.fill_between(es.horizons, inf.tau_lower, inf.tau_upper,
                            color="blue", alpha=0.15,
                            label=f"{int(round((1 - inf.alpha) * 100))}% CI")
        ax.set_xlabel("periods since adoption")
        ax.set_ylabel("effect")
        ax.set_title(f"ATT = {results.effects.att:.3f}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - defensive backend translation
        raise MlsynthPlottingError(f"GSYNTH plotting failed: {exc}") from exc
