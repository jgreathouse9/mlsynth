"""Diagnostic plot for CSCIPCA results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import CSCIPCAResults


def plot_cscipca(results: CSCIPCAResults) -> None:
    """Two-panel plot: observed vs imputed path + effect with conformal band."""

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
        ax.plot(t, design.counterfactual, "r--", lw=2, label="CSC-IPCA counterfactual")
        if 0 <= inputs.T0 - 1 < t.size:
            ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("outcome")
        ax.set_title(f"CSC-IPCA (K = {design.n_factors}, L = {inputs.L})")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

        ax = axes[1]
        ax.axhline(0.0, color="grey", lw=0.8)
        t_post = t[inputs.T0:]
        ax.plot(t_post, design.tau, "b-", lw=2, marker=".", label="effect")
        if inf.ci_lower_t.size == design.tau.size and inf.ci_lower_t.size:
            ax.fill_between(
                t_post, inf.ci_lower_t, inf.ci_upper_t,
                color="blue", alpha=0.15,
                label=f"{int(round((1 - inf.alpha) * 100))}% conformal band",
            )
        if 0 <= inputs.T0 - 1 < t.size:
            ax.axvline(t[inputs.T0 - 1], color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("effect")
        ax.set_title("CSC-IPCA effect over time")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - defensive backend-error translation
        raise MlsynthPlottingError(f"CSCIPCA plotting failed: {exc}") from exc
