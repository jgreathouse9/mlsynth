"""Diagnostic plot for FMA results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import FMAResults


def plot_fma(results: FMAResults) -> None:
    """Two-panel plot: trajectories + per-period gap with CI bands."""

    inputs = results.inputs
    inf = results.inference
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:
        t = np.arange(inputs.T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(t, inputs.treated_outcome, "k-", lw=2,
            label=str(inputs.treated_unit_name))
    ax.plot(t, results.counterfactual, "r--", lw=2,
            label="FMA counterfactual")
    if 0 <= inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("outcome")
    ax.set_title(
        f"FMA trajectories (r = {results.design.n_factors}, "
        f"source = {results.design.n_factors_source})"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.plot(t, results.gap, "b-", lw=2, label="Y_treated - Y_FMA")

    if inf.bootstrap_att_t_lower.size:
        n_post = inf.bootstrap_att_t_lower.size
        t_post = t[-n_post:]
        ax.fill_between(
            t_post,
            inf.bootstrap_att_t_lower,
            inf.bootstrap_att_t_upper,
            color="blue", alpha=0.15,
            label=f"{int(round((1 - inf.alpha) * 100))}% bootstrap CI",
        )

    if inf.placebo_quantile_lower.size:
        ax.fill_between(
            t,
            inf.placebo_quantile_lower,
            inf.placebo_quantile_upper,
            color="gray", alpha=0.2,
            label=f"{int(round((1 - inf.alpha) * 100))}% placebo band",
        )

    if 0 <= inputs.T0 - 1 < t.size:
        ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("treatment effect")
    ax.set_title("FMA gap with inference bands")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    try:
        plt.show()
    except Exception as exc:
        raise MlsynthPlottingError(f"FMA plotting failed: {exc}") from exc
