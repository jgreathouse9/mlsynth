"""Diagnostic plot for CFM results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import CFMResults


def plot_cfm(results: CFMResults) -> None:
    """Two-panel plot: trajectories + systematic causal effect with CI band."""

    inputs = results.inputs
    design = results.design
    inf = results.inference_detail
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:  # pragma: no cover - setup always yields aligned time labels
        t = np.arange(inputs.T)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        ax.plot(t, inputs.treated_outcome, "k-", lw=2,
                label=str(inputs.treated_unit_name))
        ax.plot(t, design.counterfactual, "r--", lw=2,
                label="systematic untreated")
        if 0 <= inputs.T0 - 1 < t.size:
            ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("outcome")
        ax.set_title(
            f"CFM (r = {design.n_factors}, source = {design.n_factors_source})"
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

        ax = axes[1]
        ax.axhline(0.0, color="grey", lw=0.8)
        t_post = t[inputs.T0:]
        ax.plot(t_post, design.tau, "b-", lw=2, label="systematic effect tau*")
        if inf.se_t.size == design.tau.size:
            ax.fill_between(
                t_post, inf.ci_lower_t, inf.ci_upper_t,
                color="blue", alpha=0.15,
                label=f"{int(round((1 - inf.alpha) * 100))}% CI",
            )
        if 0 <= inputs.T0 - 1 < t.size:
            ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("systematic causal effect")
        ax.set_title("CFM systematic effect with CI")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - defensive backend-error translation
        raise MlsynthPlottingError(f"CFM plotting failed: {exc}") from exc
