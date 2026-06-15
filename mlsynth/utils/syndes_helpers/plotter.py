"""Plotting helpers for SYNDES results.

Every plot is rendered through the in-house mlsynth style
(:func:`mlsynth.utils.plotting.mlsynth_style`) and the shared
:class:`mlsynth.utils.plotting.Plotter` archetypes, so SYNDES figures match the
rest of the library (LEXSCM, MAREX, ...) rather than raw Matplotlib defaults.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from ..plotting import Plotter, mlsynth_style
from .relaxed_structures import RelaxedSolverResults
from .structures import SYNDESResults


def plot_syndes_design(results: SYNDESResults | RelaxedSolverResults) -> None:
    """Dispatch to the appropriate SYNDES plot for a result object.

    When the fit produced a solution pool (``top_K > 1``), the design is shown
    as a two-panel figure: the recommended design's synthetic treated/control
    trajectory on top, and the fit-vs-power Pareto frontier (with the
    recommended design starred) on the bottom. A single-design fit keeps the
    one-panel trajectory plot.
    """

    if (getattr(results, "recommendation", None) is not None
            and getattr(results, "pool", None)):
        _plot_design_with_pareto(results)
        return

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

    n_pre = results.inputs.Y_pre.shape[0]
    treated_series, control_series = _treated_control_series(results, design)
    with mlsynth_style():
        ax = Plotter(figsize=(12, 5)).observed_vs_counterfactual(
            times=np.arange(treated_series.shape[0]),
            observed=treated_series,
            counterfactuals=control_series,
            labels=["Synthetic Control"],
            treated_label="synthetic",
            intervention=n_pre - 0.5,
            outcome=results.inputs.outcome,
            time="period",
            title=f"SYNDES Global Design ({results.mode})",
        )
        ax.figure.tight_layout()
        plt.show()


def plot_per_unit_design(results: SYNDESResults) -> None:
    """Plot actual and synthetic twin series for each selected unit."""

    design = results.design
    if design.q is None:
        raise MlsynthPlottingError("Per-unit SYNDES plotting requires q weights.")

    Y_full = _stack_pre_post(results)
    n_pre = results.inputs.Y_pre.shape[0]
    treated_idx = design.selected_unit_indices

    with mlsynth_style():
        plotter = Plotter(figsize=(10, 3 * len(treated_idx)))
        fig, axes = plt.subplots(len(treated_idx), 1,
                                 figsize=(10, 3 * len(treated_idx)))
        if len(treated_idx) == 1:
            axes = [axes]
        for ax, i in zip(axes, treated_idx):
            synthetic = Y_full @ design.q[i, :]
            unit_label = results.inputs.unit_index.labels[i]
            plotter.observed_vs_counterfactual(
                times=np.arange(Y_full.shape[0]),
                observed=Y_full[:, i],
                counterfactuals=synthetic,
                labels=["Synthetic Twin"],
                treated_label=f"unit {unit_label}",
                intervention=n_pre - 0.5,
                outcome=results.inputs.outcome,
                time="period",
                title=f"SYNDES Per-Unit Synthetic Fit: Unit {unit_label}",
                ax=ax,
            )
        fig.tight_layout()
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

    n_pre = results.inputs.Y_pre.shape[0]
    treated_series, control_series = _treated_control_series(results, results.design)
    with mlsynth_style():
        ax = Plotter(figsize=(12, 5)).observed_vs_counterfactual(
            times=np.arange(treated_series.shape[0]),
            observed=treated_series,
            counterfactuals=control_series,
            labels=["Synthetic Control"],
            treated_label="synthetic",
            intervention=n_pre - 0.5,
            outcome=results.inputs.outcome,
            time="period",
            title="SYNDES Global Design (two_way_global_annealed)",
        )
        ax.figure.tight_layout()
        plt.show()


def _stack_pre_post(results: SYNDESResults | RelaxedSolverResults) -> np.ndarray:
    if results.inputs.Y_post is None:
        return results.inputs.Y_pre
    return np.vstack([results.inputs.Y_pre, results.inputs.Y_post])


def _treated_control_series(results, design):
    """Synthetic treated / control trajectories for any SYNDES design.

    Mirrors the contrast bookkeeping in ``_syndes_post_fit``: ``per_unit`` keeps
    a ``(K, N)`` treated-weight matrix (control side = its column sum / K), while
    the global / annealed modes carry plain ``(N,)`` weight vectors.
    """
    Y_full = _stack_pre_post(results)
    N = Y_full.shape[1]
    assignment = np.asarray(getattr(design, "assignment", np.zeros(N)),
                            dtype=float).flatten()
    K = int(assignment.sum()) or 1
    tw_raw = getattr(design, "treated_weights", None)
    cw_raw = getattr(design, "control_weights", None)
    if tw_raw is not None and np.asarray(tw_raw).ndim == 2:
        q = np.asarray(tw_raw, dtype=float)
        tw = assignment / K
        cw = q.sum(axis=0) / K
    else:
        tw = (np.asarray(tw_raw, dtype=float).flatten()
              if tw_raw is not None else np.zeros(N))
        cw = (np.asarray(cw_raw, dtype=float).flatten()
              if cw_raw is not None else np.zeros(N))
    return Y_full @ tw, Y_full @ cw


def _draw_pareto(ax, rec) -> "object":
    """Draw the fit-vs-power Pareto scatter onto ``ax`` (house palette)."""
    table = rec.table
    fit = np.array([r["fit_rmse"] for r in table], dtype=float)
    mde = np.array([r["mde_pct"] if np.isfinite(r["mde_pct"]) else np.nan
                    for r in table], dtype=float)
    pareto = np.array([r["pareto"] for r in table], dtype=bool)
    winner = np.array([r["winner"] for r in table], dtype=bool)

    dominated = ~pareto
    if dominated.any():
        ax.scatter(fit[dominated], mde[dominated], color="grey", s=55,
                   alpha=0.7, label="dominated", zorder=2)
    if pareto.any():
        order = np.argsort(fit[pareto])
        ax.plot(fit[pareto][order], mde[pareto][order], "-o", color="blue",
                linewidth=1.4, label="Pareto frontier", zorder=3)
    if winner.any():
        ax.scatter(fit[winner], mde[winner], marker="*", s=320, color="red",
                   edgecolor="white", linewidth=0.8, label="recommended",
                   zorder=4)
    for r in table:
        if np.isfinite(r["mde_pct"]):
            ax.annotate(r["design_id"], (r["fit_rmse"], r["mde_pct"]),
                        fontsize=10, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("pre-period RMSE (treated vs weighted control)")
    ax.set_ylabel("MDE %  (lower is better)")
    ax.set_title("SYNDES pool: fit vs power (Pareto frontier)")
    ax.legend()
    return ax


def plot_syndes_pareto(results: SYNDESResults, ax=None):
    """Plot the SYNDES pool's fit-vs-power Pareto frontier (in-house style).

    Scatters every pooled design as (pre-period RMSE between treated and the
    weighted control, MDE%); highlights the non-dominated frontier and stars the
    recommended design. Requires a fit with a solution pool (``top_K > 1``).
    Pass ``ax`` to compose into an existing figure; otherwise a styled standalone
    figure is created. Returns the axis drawn on.
    """
    rec = getattr(results, "recommendation", None)
    if rec is None or not getattr(results, "pool", None):
        raise MlsynthPlottingError(
            "The Pareto plot requires a SYNDES solution pool (set top_K > 1)."
        )
    if ax is not None:
        return _draw_pareto(ax, rec)
    with mlsynth_style():
        _, ax = plt.subplots(figsize=(8, 5))
        _draw_pareto(ax, rec)
        ax.figure.tight_layout()
        plt.show()
    return ax


def _plot_design_with_pareto(results: SYNDESResults) -> None:
    """Two-panel plot: recommended design trajectory + the Pareto frontier."""
    rec = results.recommendation
    winner_design = (rec.winner.design if rec.winner is not None
                     and rec.winner.design is not None else results.design)
    n_pre = results.inputs.Y_pre.shape[0]
    treated, control = _treated_control_series(results, winner_design)
    wid = rec.winner.design_id if rec.winner is not None else "?"

    with mlsynth_style():
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9))
        Plotter().observed_vs_counterfactual(
            times=np.arange(treated.shape[0]),
            observed=treated,
            counterfactuals=control,
            labels=["Synthetic Control"],
            treated_label=f"design {wid}",
            intervention=n_pre - 0.5,
            outcome=results.inputs.outcome,
            time="period",
            title=f"SYNDES recommended design {wid} ({results.mode})",
            ax=ax_top,
        )
        _draw_pareto(ax_bot, rec)
        fig.tight_layout()
        plt.show()
