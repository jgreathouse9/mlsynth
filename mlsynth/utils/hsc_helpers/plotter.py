"""Plotting for HSC: observed treated series vs the HSC counterfactual."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import HSCResults


def plot_hsc(results: HSCResults) -> None:
    """Plot the treated outcome against the HSC counterfactual.

    Parameters
    ----------
    results : HSCResults
        Output of :class:`mlsynth.estimators.HSC`.

    Raises
    ------
    MlsynthPlottingError
        If required arrays are missing from the result.
    """

    if results.inputs is None or results.counterfactual_full is None:
        raise MlsynthPlottingError(
            "HSC plotting requires inputs and a counterfactual on the result."
        )

    inputs = results.inputs
    design = results.design
    x = np.arange(inputs.T)

    plt.figure(figsize=(12, 5))
    plt.plot(x, inputs.y_target, lw=2.5, label=f"Treated ({inputs.treated_unit_name})")
    plt.plot(
        x,
        results.counterfactual_full,
        lw=2,
        ls="--",
        label=f"HSC counterfactual (rho={design.selected_rho:.2f}, q={design.q})",
    )
    plt.axvline(x=inputs.T0 - 0.5, color="red", alpha=0.4, label="Treatment start")
    plt.title(
        f"Harmonic Synthetic Control (q={design.q}, forecaster={design.forecaster})"
    )
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
