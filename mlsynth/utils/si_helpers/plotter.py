"""Plotting for SI: the focal unit's observed series vs its counterfactual
under each alternative intervention.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import SIResults


def plot_si(results: SIResults) -> None:
    """Plot the focal unit against its SI counterfactuals (one per intervention).

    Parameters
    ----------
    results : SIResults
        Output of :class:`mlsynth.estimators.SI`.

    Raises
    ------
    MlsynthPlottingError
        If the result carries no fitted arms.
    """
    if not results.arms:
        raise MlsynthPlottingError("SI plotting requires at least one fitted arm.")

    inputs = results.inputs
    x = np.arange(inputs.T)

    plt.figure(figsize=(12, 5))
    plt.plot(
        x, inputs.y_target, lw=2.5, color="black",
        label=f"Observed ({inputs.treated_unit_name})",
    )
    for name, arm in results.arms.items():
        label = f"SI under {name} (k={arm.selected_rank})"
        if arm.cf_mean_ci is not None:
            label += f", ATT={arm.att:.1f}"
        plt.plot(x, arm.counterfactual, lw=2, ls="--", label=label)
    plt.axvline(x=inputs.T0 - 0.5, color="red", alpha=0.4, label="Treatment start")
    plt.title(f"Synthetic Interventions ({inputs.treated_unit_name})")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
