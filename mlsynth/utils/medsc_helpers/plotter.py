"""Diagnostic plot for MEDSC results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import MEDSCResults


def plot_medsc(results: MEDSCResults) -> None:
    """Two-panel plot: observed vs both counterfactuals + effect decomposition."""

    inputs = results.inputs
    dec = results.decomposition
    t = np.asarray(inputs.time_labels)
    if t.size != inputs.T:  # pragma: no cover - setup always yields aligned labels
        t = np.arange(inputs.T)
    cut = t[inputs.T0 - 1] if 0 <= inputs.T0 - 1 < t.size else None

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        ax = axes[0]
        ax.plot(t, inputs.treated_outcome, "k-", lw=2,
                label=str(inputs.treated_name))
        ax.plot(t, dec.counterfactual_total, "r--", lw=2,
                label=r"synthetic $\hat{Y}^{0,M0}$ (total)")
        ax.plot(t, dec.counterfactual_direct, "b-.", lw=2,
                label=r"cross-world $\hat{Y}^{0,M1}$ (direct)")
        if cut is not None:
            ax.axvline(cut, color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("outcome")
        ax.set_title(f"MEDSC ({inputs.treated_name})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

        ax = axes[1]
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.plot(t, dec.total, "r-", lw=2, label="total")
        ax.plot(t, dec.direct, "b-", lw=2, marker=".", label="direct")
        ax.plot(t, dec.indirect, "g-", lw=2, marker=".", label="indirect")
        if cut is not None:
            ax.axvline(cut, color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("effect")
        ax.set_title("MEDSC effect decomposition")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - defensive backend-error translation
        raise MlsynthPlottingError(f"MEDSC plotting failed: {exc}") from exc
