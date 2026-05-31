"""Diagnostic plots for SPILLSYNTH results.

Three panels recreate Figure 4 of Cao & Dowd (2023):

* (a) treated trajectory + vanilla-SCM and SP-adjusted counterfactuals.
* (b) per-period treatment effects for SCM and SP, with the post-period
  highlighted.
* (c) per-affected-unit spillover trajectories, restricted to the
  post-period.
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
    """Three-panel diagnostic plot for SPILLSYNTH.

    Parameters
    ----------
    results : SpillSynthResults
        Fit container returned by :meth:`SPILLSYNTH.fit`.
    treated_color : str
        Color of the treated unit's observed-outcome line.
    counterfactual_color : str
        Color for the vanilla SCM counterfactual.
    sp_color : str
        Color for the spillover-adjusted counterfactual and SP gap line.
    save : str, optional
        Path to write the figure to. If ``None`` (default), the figure
        is shown interactively.
    """
    try:
        inputs = results.inputs
        cd = results._active

        t = np.asarray(inputs.time_labels)
        t_post = np.asarray(inputs.post_time)
        t_treat_cutoff = t[inputs.T0 - 1] if inputs.T0 >= 1 else None

        # Full-period treated path: actual.
        y_treated = inputs.Y[0]

        # Full-period vanilla SCM counterfactual (pre + post).
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
        # Pre-period: SCM/SP residuals are by construction the leave-one-out
        # SCM fit residual, identical for both estimators.
        pre_resid_scm = y_treated[:inputs.T0] - sc_full_pre
        ax.plot(t[:inputs.T0], pre_resid_scm, "--", color=counterfactual_color,
                alpha=0.6, lw=1.5, label="Vanilla SCM (pre fit)")
        ax.plot(t_post, cd.gap_scm, "--", color=counterfactual_color, lw=2,
                label="Vanilla SCM (post)")
        ax.plot(t_post, cd.gap_sp, "-", color=sp_color, lw=2,
                label="SP (post)")
        if t_treat_cutoff is not None:
            ax.axvline(t_treat_cutoff, color="grey", ls=":", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("treatment effect")
        ax.set_title("Per-period gap")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

        # --- Panel (c): spillover trajectories (if any affected units) ---
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
    except Exception as exc:                                # pragma: no cover
        raise MlsynthPlottingError(
            f"SPILLSYNTH plotter failed: {exc}"
        ) from exc
