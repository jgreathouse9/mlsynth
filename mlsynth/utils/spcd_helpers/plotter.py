"""Plotting helpers for SPCD results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import SPCDResults


def plot_spcd_design(results: SPCDResults) -> None:
    """Plot synthetic treated and synthetic control trajectories for SPCD.

    Mirrors the layout of ``plot_relaxed_design`` in
    ``syndes_helpers.plotter`` so SPCD and SYNDES plots feel consistent.

    Parameters
    ----------
    results : SPCDResults
        Output of :class:`mlsynth.estimators.SPCD`. Must have ``inputs``
        attached so that pre/post matrices can be stacked.

    Raises
    ------
    MlsynthPlottingError
        If the result has no attached ``inputs``.
    """

    if results.inputs is None:
        raise MlsynthPlottingError(
            "SPCD plotting requires inputs to be attached to the result."
        )

    design = results.design
    Y_full = _stack_pre_post(results)
    n_pre = results.inputs.Y_pre.shape[0]

    treated_series = Y_full @ design.treated_weights
    control_series = Y_full @ design.control_weights

    title = (
        f"SPCD Design (variant={design.variant}, "
        f"weights={design.weights_mode}, K={design.n_treated})"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(treated_series, lw=3, label="Synthetic Treated")
    plt.plot(control_series, lw=2, ls="--", label="Synthetic Control")
    plt.axvline(x=n_pre - 0.5, color="red", alpha=0.4, label="Treatment Start")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def _stack_pre_post(results: SPCDResults) -> np.ndarray:
    if results.inputs.Y_post is None:
        return results.inputs.Y_pre
    return np.vstack([results.inputs.Y_pre, results.inputs.Y_post])
