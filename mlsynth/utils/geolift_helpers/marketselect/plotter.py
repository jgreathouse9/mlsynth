"""Plot the recommended GeoLift design — design phase and post (realized) phase.

- **Design phase** (no realized report): the winning test-market aggregate vs its
  synthetic counterfactual over the pre-period (the fit you are buying).
- **Post phase** (a realized report present): observed vs counterfactual over
  pre+post with the conformal prediction band, plus the effect (gap) with its
  per-period conformal intervals and the intervention line.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_design(result, report=None, *, figsize=(11, 7), save_path=None, show=False):
    """Plot the recommended design (design phase, or realized post phase)."""
    report = report if report is not None else getattr(result, "report", None)
    if report is not None:
        return _plot_post(result, report, figsize=figsize, save_path=save_path, show=show)
    return _plot_design_phase(result, figsize=figsize, save_path=save_path, show=show)


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _plot_design_phase(result, *, figsize, save_path, show):
    winner = result.search.winner
    if winner is None:
        raise ValueError(
            "no winning design to plot (no candidate cleared the power threshold)."
        )
    ts = winner.time_series
    obs = np.asarray(ts.observed_outcome, dtype=float)
    cf = np.asarray(ts.counterfactual_outcome, dtype=float)
    x = np.arange(obs.shape[0])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, obs, color="black", lw=1.6, label="Observed (test markets)")
    ax.plot(x, cf, color="C3", ls="--", lw=1.6, label="Synthetic counterfactual")
    ax.set_ylabel("Outcome")
    ax.set_xlabel("Time")
    ax.legend(frameon=False)

    units = "+".join(result.selected_units) if result.selected_units else "design"
    title = f"Design phase — {units}"
    if winner.mde is not None:
        sl2 = winner.fit_diagnostics.additional_metrics.get("scaled_l2")
        title += f"  (MDE {winner.mde:.2f}"
        title += f", scaled L2 {sl2:.3f})" if sl2 is not None else ")"
    fig.suptitle(title)
    return _finish(fig, save_path, show)


def _plot_post(result, report, *, figsize, save_path, show):
    ts = report.time_series
    obs = np.asarray(ts.observed_outcome, dtype=float)
    cf = np.asarray(ts.counterfactual_outcome, dtype=float)
    gap = np.asarray(ts.estimated_gap, dtype=float)
    x = np.arange(obs.shape[0])

    det = report.inference.details
    post = np.asarray(det["periods"], dtype=int)
    lower = np.asarray(det["lower"], dtype=float)
    upper = np.asarray(det["upper"], dtype=float)
    intervention_x = int(post[0]) if post.size else obs.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Panel 1: observed vs synthetic, with the conformal band on the counterfactual.
    ax1.plot(x, obs, color="black", lw=1.6, label="Observed (test markets)")
    ax1.plot(x, cf, color="C3", ls="--", lw=1.6, label="Synthetic counterfactual")
    if post.size:
        # effect CI [lower, upper] -> counterfactual band obs - [upper, lower]
        ax1.fill_between(post, obs[post] - upper, obs[post] - lower,
                         color="C3", alpha=0.2, label="Conformal interval")
    ax1.axvline(intervention_x, color="gray", ls=":", lw=1.0)
    ax1.set_ylabel("Outcome")
    ax1.legend(frameon=False)

    # Panel 2: the effect (gap) with its per-period conformal intervals.
    ax2.plot(x, gap, color="black", lw=1.4)
    ax2.axhline(0.0, color="gray", lw=0.8)
    if post.size:
        ax2.fill_between(post, lower, upper, color="C3", alpha=0.2)
    ax2.axvline(intervention_x, color="gray", ls=":", lw=1.0)
    ax2.set_ylabel("Effect (obs − synthetic)")
    ax2.set_xlabel("Time")

    units = "+".join(result.selected_units) if getattr(result, "selected_units", None) else "design"
    p = report.inference.p_value
    fig.suptitle(f"Realized design — {units}   (conformal joint p = {p:.3f})")
    return _finish(fig, save_path, show)
