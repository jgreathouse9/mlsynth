"""Plotting helpers for SYNDES results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .relaxed_structures import RelaxedSolverResults
from .structures import SYNDESResults


def plot_syndes_design(results: SYNDESResults | RelaxedSolverResults) -> None:
    """Dispatch to the appropriate SYNDES / SYNDES plot for a result object."""

    if results.mode == "two_way_global_annealed":
        plot_relaxed_design(results)
    elif results.mode in {
        "global_2way", "global_equal_weights",
        "two_way_global", "one_way_global",  # SYNDES paper-aligned names
    }:
        plot_global_design(results)
    elif results.mode == "per_unit":
        plot_per_unit_design(results)
    else:
        raise MlsynthPlottingError(f"Unknown SYNDES plot mode: {results.mode}")


def plot_global_design(results: SYNDESResults) -> None:
    """Plot synthetic treated/control aggregate series for a global SYNDES design."""

    design = results.design

    if design.treated_weights is None or design.control_weights is None:
        raise MlsynthPlottingError("Missing treated/control weights.")

    Y_full = _stack_pre_post(results)
    n_pre = results.inputs.Y_pre.shape[0]

    treated_series = Y_full @ design.treated_weights
    control_series = Y_full @ design.control_weights

    plt.figure(figsize=(12, 5))
    plt.plot(treated_series, lw=3, label="Synthetic Treated")
    plt.plot(control_series, lw=2, ls="--", label="Synthetic Control")
    plt.axvline(x=n_pre - 0.5, color="red", alpha=0.4, label="Treatment Start")
    plt.title(f"SYNDES Global Design ({results.mode})")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_per_unit_design(results: SYNDESResults) -> None:
    """Plot actual and synthetic twin series for each selected unit."""

    design = results.design
    if design.q is None:
        raise MlsynthPlottingError("Per-unit SYNDES plotting requires q weights.")

    Y_full = _stack_pre_post(results)
    n_pre = results.inputs.Y_pre.shape[0]
    treated_idx = design.selected_unit_indices

    fig, axes = plt.subplots(len(treated_idx), 1, figsize=(10, 3 * len(treated_idx)))
    if len(treated_idx) == 1:
        axes = [axes]

    for ax, i in zip(axes, treated_idx):
        synthetic = Y_full @ design.q[i, :]
        unit_label = results.inputs.unit_index.labels[i]
        ax.plot(Y_full[:, i], lw=2, label=f"Unit {unit_label} Actual")
        ax.plot(synthetic, ls="--", lw=2, label="Synthetic Twin")
        ax.axvline(x=n_pre - 0.5, color="red", alpha=0.3)
        ax.legend()
        ax.set_title(f"SYNDES Per-Unit Synthetic Fit: Unit {unit_label}")

    plt.tight_layout()
    plt.show()


def plot_relaxed_design(results: RelaxedSolverResults) -> None:
    """Plot synthetic treated/control aggregate series for a relaxed SYNDES design.

    Parameters
    ----------
    results : RelaxedSolverResults
        Output of the relaxed annealing solver. Must have ``inputs``
        attached so that pre/post matrices can be stacked.

    Raises
    ------
    MlsynthPlottingError
        If the result has no attached ``inputs``.
    """

    if results.inputs is None:
        raise MlsynthPlottingError(
            "Relaxed SYNDES plotting requires inputs to be attached to the result."
        )

    design = results.design
    Y_full = _stack_pre_post(results)
    n_pre = results.inputs.Y_pre.shape[0]

    treated_series = Y_full @ design.treated_weights
    control_series = Y_full @ design.control_weights

    plt.figure(figsize=(12, 5))
    plt.plot(treated_series, lw=3, label="Synthetic Treated")
    plt.plot(control_series, lw=2, ls="--", label="Synthetic Control")
    plt.axvline(x=n_pre - 0.5, color="red", alpha=0.4, label="Treatment Start")
    plt.title("SYNDES Global Design (two_way_global_annealed)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def _stack_pre_post(results: SYNDESResults | RelaxedSolverResults) -> np.ndarray:
    if results.inputs.Y_post is None:
        return results.inputs.Y_pre
    return np.vstack([results.inputs.Y_pre, results.inputs.Y_post])
