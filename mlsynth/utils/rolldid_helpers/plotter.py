"""Event-study / effect plot for ROLLDID."""

from __future__ import annotations

from typing import Optional


def plot_rolldid(result, *, show: bool = True, theme: Optional[dict] = None):
    """Plot the rolling-DiD effect.

    Common timing: the per-period ATTs (event study) with their confidence band
    and a zero reference line. Staggered: the per-cohort ATTs. Uses the shared
    ``mlsynth`` house style.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from mlsynth.utils.plotting import mlsynth_style

    with mlsynth_style(theme):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if result.per_period is not None:
            pp = result.per_period
            x = np.asarray(pp["time"])
            ax.plot(x, pp["att"], color="black", marker="o", label="ATT")
            ax.fill_between(x, pp["ci_lower"], pp["ci_upper"], color="grey",
                            alpha=0.3, label="CI")
            ax.set_xlabel("period")
        elif result.per_cohort is not None:
            pc = result.per_cohort
            x = np.arange(len(pc))
            ax.errorbar(x, pc["att"], yerr=[pc["att"] - pc["ci_lower"],
                        pc["ci_upper"] - pc["att"]], fmt="o", color="black",
                        capsize=4, label="cohort ATT")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in pc["cohort"]])
            ax.set_xlabel("cohort")
        ax.axhline(0.0, color="red", linestyle=":")
        att = result.effects.att if result.effects else None
        ax.set_title(f"ROLLDID ({result.transformation})"
                     + (f"  aggregated ATT {att:.3f}" if att is not None else ""))
        ax.set_ylabel("effect")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
    if show:
        plt.show()
    return fig
