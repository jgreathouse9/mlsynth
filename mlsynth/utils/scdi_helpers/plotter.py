"""Plotting helpers for SCDI results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import SCDIResults


def plot_scdi_design(results: SCDIResults) -> None:
    """Dispatch to the appropriate SCDI plot for a result object."""

    if results.mode in {"global_2way", "global_equal_weights"}:
        plot_global_design(results)
    elif results.mode == "per_unit":
        plot_per_unit_design(results)
    else:
        raise MlsynthPlottingError(f"Unknown SCDI plot mode: {results.mode}")





def plot_global_design(results: SCDIResults) -> None:
    """Plot synthetic treated/control aggregate series for a global SCDI design."""

    design = results.design

    # ----------------------------
    # unified requirement
    # ----------------------------
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
    plt.title(f"SCDI Global Design ({results.mode})")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()






def plot_per_unit_design(results: SCDIResults) -> None:
    """Plot actual and synthetic twin series for each selected unit."""

    design = results.design
    if design.q is None:
        raise MlsynthPlottingError("Per-unit SCDI plotting requires q weights.")

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
        ax.set_title(f"SCDI Per-Unit Synthetic Fit: Unit {unit_label}")

    plt.tight_layout()
    plt.show()


def _stack_pre_post(results: SCDIResults) -> np.ndarray:
    if results.inputs.Y_post is None:
        return results.inputs.Y_pre
    return np.vstack([results.inputs.Y_pre, results.inputs.Y_post])
