"""Diagnostic plots for SPILLSYNTH results.

Single-treated panels reproduce Figure 4 of Cao & Dowd (2023):

* (a) treated trajectory + vanilla-SCM and SP-adjusted counterfactuals.
* (b) per-period treatment effects for SCM and SP, with the post-period
  highlighted.
* (c) per-affected-unit spillover trajectories, restricted to the
  post-period.

Multi-treated panels (Section S.1.2) switch to an event-study style:

* (a) per-treated-unit, per-post-period treatment effects with shaded
  95% confidence intervals on top of a flat zero-line.
* (b) per-affected-unit spillover trajectories (when any affected
  units are declared).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import MlsynthPlottingError
from .structures import SpillSynthResults


def plot_spillsynth(
    results: SpillSynthResults,
    *,
    treated_color: str = "black",
    counterfactual_color: str = "red",
    sp_color: str = "tab:blue",
    save: Optional[str] = None,
) -> None:
    """Diagnostic plot for SPILLSYNTH.

    Parameters
    ----------
    results : SpillSynthResults
        Fit container returned by :meth:`SPILLSYNTH.fit`.
    treated_color : str
        Color of the treated unit's observed-outcome line (single-
        treated layout only; ignored in multi-treated event-study mode
        where each treated unit gets its own colour from the default
        cycle).
    counterfactual_color : str
        Color for the vanilla SCM counterfactual (single-treated only).
    sp_color : str
        Color for the SP-adjusted counterfactual and SP gap line
        (single-treated only).
    save : str, optional
        Path to write the figure to. If ``None`` (default), the figure
        is shown interactively.
    """
    try:
        inputs = results.inputs
        if inputs.n_treated > 1:
            _plot_event_study(results, save=save)
        else:
            _plot_single_treated(
                results,
                treated_color=treated_color,
                counterfactual_color=counterfactual_color,
                sp_color=sp_color,
                save=save,
            )
    except Exception as exc:                                # pragma: no cover
        raise MlsynthPlottingError(
            f"SPILLSYNTH plotter failed: {exc}"
        ) from exc


def _plot_single_treated(
    results: SpillSynthResults,
    *,
    treated_color: str,
    counterfactual_color: str,
    sp_color: str,
    save: Optional[str],
) -> None:
    """Three-panel diagnostic (Figure 4 of Cao-Dowd) for a single treated unit."""
    inputs = results.inputs
    cd = results._active

    t = np.asarray(inputs.time_labels)
    t_post = np.asarray(inputs.post_time)
    t_treat_cutoff = t[inputs.T0 - 1] if inputs.T0 >= 1 else None

    y_treated = inputs.Y[0]
    sc_full_pre = cd.a[0] + cd.B[0] @ inputs.Y_pre
    sc_full = np.concatenate([sc_full_pre, cd.counterfactual_scm])
    sp_full = np.concatenate([sc_full_pre, cd.counterfactual_sp])

    if inputs.p > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel (a): trajectories ---
    ax = axes[0]
    ax.plot(t, y_treated, color=treated_color, lw=2,
            label=str(inputs.treated_label))
    ax.plot(t, sc_full, "--", color=counterfactual_color, lw=2,
            label="Vanilla SCM")
    ax.plot(t, sp_full, "-", color=sp_color, lw=2,
            label="Spillover-adjusted (SP)")
    if t_treat_cutoff is not None:
        ax.axvline(t_treat_cutoff, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("outcome")
    ax.set_title(f"{inputs.treated_label}: actual vs counterfactuals")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    # --- Panel (b): per-period treatment effects ---
    ax = axes[1]
    ax.axhline(0.0, color="grey", lw=0.8)
    pre_resid_scm = y_treated[:inputs.T0] - sc_full_pre
    ax.plot(t[:inputs.T0], pre_resid_scm, "--", color=counterfactual_color,
            alpha=0.6, lw=1.5, label="Vanilla SCM (pre fit)")
    ax.plot(t_post, cd.gap_scm, "--", color=counterfactual_color, lw=2,
            label="Vanilla SCM (post)")
    ax.plot(t_post, cd.gap_sp, "-", color=sp_color, lw=2, label="SP (post)")
    if t_treat_cutoff is not None:
        ax.axvline(t_treat_cutoff, color="grey", ls=":", alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("treatment effect")
    ax.set_title("Per-period gap")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    # --- Panel (c): spillover trajectories ---
    if inputs.p > 0:
        ax = axes[2]
        ax.axhline(0.0, color="grey", lw=0.8)
        for label, traj in cd.spillover_panel.items():
            ax.plot(t_post, traj, lw=1.5, label=str(label))
        if t_treat_cutoff is not None:
            ax.axvline(t_treat_cutoff, color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("spillover effect")
        ax.set_title("Per-affected-unit spillover")
        ax.legend(loc="best", fontsize=8, ncol=max(1, inputs.p // 6))
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"SPILLSYNTH (method='{results.method}', "
        f"ATT_SP={cd.att_sp:+.3f}, ATT_SCM={cd.att_scm:+.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _plot_event_study(
    results: SpillSynthResults, *, save: Optional[str],
) -> None:
    """Event-study layout for multiple treated units (Cao-Dowd v3 S.1.2).

    Panel (a): one line per treated unit -- the SP gap over the post-
    period -- with a shaded 95% confidence interval. A flat zero-line
    is overlaid for reference. Panel (b) holds the spillover
    trajectories when any affected units were declared (otherwise the
    figure is single-panel).
    """
    inputs = results.inputs
    cd = results._active

    t_post = np.asarray(inputs.post_time)
    t_treat_cutoff = (
        np.asarray(inputs.time_labels)[inputs.T0 - 1]
        if inputs.T0 >= 1 else None
    )

    if inputs.p > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax_es, ax_sp = axes
    else:
        fig, ax_es = plt.subplots(1, 1, figsize=(8, 4.8))
        ax_sp = None

    # --- Event-study panel ---
    ax_es.axhline(0.0, color="grey", lw=0.8)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, label in enumerate(inputs.treated_labels):
        c = colours[i % len(colours)]
        gap = cd.gaps_sp_by_unit[label]
        ci = cd.treatment_cis_95[label]
        ax_es.plot(t_post, gap, "-o", color=c, lw=2, ms=4,
                   label=str(label))
        ax_es.fill_between(t_post, ci[:, 0], ci[:, 1],
                           color=c, alpha=0.2)
    if t_treat_cutoff is not None and t_treat_cutoff < t_post[0]:
        # Visual reminder of where the pre-period ends (only useful when
        # we plot some pre-period context — here we only plot post, so
        # the cutoff vertical is suppressed by the conditional).
        pass
    ax_es.set_xlabel("time")
    ax_es.set_ylabel("treatment effect")
    ax_es.set_title(
        f"Event study: SP gap by treated unit "
        f"(n_treated = {inputs.n_treated})"
    )
    ax_es.legend(loc="best",
                 ncol=max(1, inputs.n_treated // 6),
                 fontsize=8 if inputs.n_treated > 6 else 10)
    ax_es.grid(alpha=0.2)

    # --- Spillover panel ---
    if ax_sp is not None:
        ax_sp.axhline(0.0, color="grey", lw=0.8)
        for label, traj in cd.spillover_panel.items():
            ax_sp.plot(t_post, traj, lw=1.5, label=str(label))
        ax_sp.set_xlabel("time")
        ax_sp.set_ylabel("spillover effect")
        ax_sp.set_title("Per-affected-unit spillover")
        ax_sp.legend(loc="best", fontsize=8,
                     ncol=max(1, inputs.p // 6))
        ax_sp.grid(alpha=0.2)

    avg_att = float(np.mean(list(cd.atts_sp_by_unit.values())))
    fig.suptitle(
        f"SPILLSYNTH (method='{results.method}', "
        f"avg ATT_SP across treated = {avg_att:+.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
