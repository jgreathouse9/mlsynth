"""Plotting helpers for SPCD results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Union

from ...exceptions import MlsynthPlottingError
from .structures import SPCDMultiArmResults, SPCDResults


def _resolve_arms(
    results: Union[SPCDResults, SPCDMultiArmResults]
) -> Tuple[dict, object]:
    """Return ``({label: SPCDResults}, pooled_power_or_None)`` for either a
    single :class:`SPCDResults` or a multi-arm :class:`SPCDMultiArmResults`.
    """

    if isinstance(results, SPCDMultiArmResults):
        return dict(results.arm_designs), results.pooled_power
    return {"design": results}, None


def plot_mde_bars(
    results: Union[SPCDResults, SPCDMultiArmResults],
    ax: Optional["plt.Axes"] = None,
):
    """Bar chart of the percent MDE per arm, with the pooled whole-study
    MDE alongside (when present).

    Parameters
    ----------
    results : SPCDResults or SPCDMultiArmResults
        A fitted SPCD result. Power analysis must have run
        (``enable_inference=True``).
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """

    arms, pooled = _resolve_arms(results)
    labels = [k for k in arms if arms[k].mde_pct is not None]
    if not labels and pooled is None:
        raise MlsynthPlottingError(
            "No MDE available to plot; fit with enable_inference=True."
        )

    fig = ax.figure if ax is not None else plt.figure(figsize=(7, 4.5))
    ax = ax or fig.add_subplot(111)

    values = [float(arms[k].mde_pct) for k in labels]
    ax.bar([str(k) for k in labels], values, color="#1428A0", label="per-arm")
    if pooled is not None and pooled.mde_pct is not None:
        ax.bar(["Pooled"], [float(pooled.mde_pct)], color="#E04E39",
               label="whole study")
        ax.axhline(float(pooled.mde_pct), ls="--", lw=1, color="#E04E39", alpha=0.6)
        values = values + [float(pooled.mde_pct)]
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f}%", (i, v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Minimum detectable effect (% of baseline)")
    ax.set_title("SPCD MDE by arm vs. pooled")
    ax.legend()
    return fig


def plot_power_curves(
    results: Union[SPCDResults, SPCDMultiArmResults],
    ax: Optional["plt.Axes"] = None,
):
    """Plot the stored power-vs-effect curves for each arm and the pooled
    whole-study contrast. The MDE is where each curve crosses the
    ``power_target`` line.

    Returns
    -------
    matplotlib.figure.Figure
    """

    arms, pooled = _resolve_arms(results)
    fig = ax.figure if ax is not None else plt.figure(figsize=(7.5, 4.8))
    ax = ax or fig.add_subplot(111)

    target = None
    plotted = False
    for label, r in arms.items():
        power = r.power
        if power is None or power.power_curve is None:
            continue
        target = power.power_target
        ax.plot(power.effect_grid_pct, power.power_curve, lw=1.6,
                label=f"arm {label} (MDE {r.mde_pct:.2f}%)")
        plotted = True
    if pooled is not None and pooled.power_curve is not None:
        target = pooled.power_target
        ax.plot(pooled.effect_grid_pct, pooled.power_curve, lw=2.6,
                color="#E04E39", label=f"POOLED (MDE {pooled.mde_pct:.2f}%)")
        plotted = True
    if not plotted:
        raise MlsynthPlottingError(
            "No power curves available; fit with enable_inference=True."
        )

    if target is not None:
        ax.axhline(target, ls=":", color="grey",
                   label=f"target power = {target:g}")
    ax.set_xlabel("Effect size (% of baseline)")
    ax.set_ylabel("Power")
    ax.set_title("SPCD power curves")
    ax.legend(fontsize=9)
    return fig


def plot_detectability(
    results: Union[SPCDResults, SPCDMultiArmResults],
    ax: Optional["plt.Axes"] = None,
):
    """Plot the MDE as a function of post-treatment horizon ("MDE at time
    point t") for each arm and the pooled whole-study contrast -- the
    "how long must the study run?" view.

    Returns
    -------
    matplotlib.figure.Figure
    """

    arms, pooled = _resolve_arms(results)
    fig = ax.figure if ax is not None else plt.figure(figsize=(7.5, 4.8))
    ax = ax or fig.add_subplot(111)

    plotted = False
    for label, r in arms.items():
        power = r.power
        if power is None or not power.detectability:
            continue
        horizons = sorted(power.detectability)
        ax.plot(horizons, [power.detectability[h] for h in horizons],
                marker="o", ms=3, lw=1.4, label=f"arm {label}")
        plotted = True
    if pooled is not None and pooled.detectability:
        horizons = sorted(pooled.detectability)
        ax.plot(horizons, [pooled.detectability[h] for h in horizons],
                marker="o", ms=4, lw=2.6, color="#E04E39", label="POOLED")
        plotted = True
    if not plotted:
        raise MlsynthPlottingError(
            "No detectability curve available; fit with enable_inference=True."
        )

    ax.set_xlabel("Post-treatment horizon (time points)")
    ax.set_ylabel("Minimum detectable effect (% of baseline)")
    ax.set_title("SPCD detectability: MDE vs. study length")
    ax.legend(fontsize=9)
    return fig


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
