"""Plot a SPILLCORRECT result: the effect path before and after the correction."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_spillcorrect(results, *, ax=None):
    """The uncorrected and corrected effect over the design's blank and post
    periods, with the restricted re-solve when present.

    The blank periods are a placebo window: no effect reaches them, so a path
    away from zero there is error. Returns the ``Figure``; displaying or saving
    it is the caller's.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure
    arms = [("uncorrected", results.uncorrected, "--"),
            (f"corrected ({results.method})", results.corrected, "-")]
    if results.restricted is not None and results.method != "restricted":
        arms.append(("restricted", results.restricted, ":"))
    n_blank = results.corrected.placebo.shape[0]
    x = np.arange(-n_blank, results.corrected.effect.shape[0])
    for label, arm, style in arms:
        ax.plot(x, np.concatenate([arm.placebo, arm.effect]), style, label=label)
    ax.axvline(-0.5, color="grey", lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_xlabel("periods from treatment (negative: blank placebo window)")
    ax.set_ylabel("treated less control composite")
    ax.legend()
    return fig
