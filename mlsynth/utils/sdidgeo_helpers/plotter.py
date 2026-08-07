"""Plots for an SDIDGEO design.

Two views, matching what GeoLift shows for a market selection:

- the power curve, power against injected effect size, with the detection
  threshold and the minimum detectable effect marked, and every other candidate
  drawn faintly behind so the winner is placed among the alternatives;
- the fit being bought, the test region's aggregate against its synthetic
  counterfactual over the pre-period.

Both render in the mlsynth house style (:func:`mlsynth_style`); pass ``theme``
to override.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import mlsynth_style


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:  # pragma: no cover - interactive
        plt.show()
    return fig


def _label(candidate) -> str:
    return " + ".join(sorted(map(str, candidate)))


def _power_panel(ax, result, power_threshold):
    """Power against effect size: the winner in front, the field behind."""
    table = result.search.power_table
    winner_units = result.selected_units
    winner_key = None
    for candidate in table["candidate"].unique():
        if sorted(map(str, candidate)) == list(winner_units or ()):
            winner_key = candidate
            break

    for candidate, group in table.groupby("candidate", sort=False):
        if candidate is winner_key:
            continue
        curve = group.groupby("effect_size")["power"].mean().sort_index()
        ax.plot(curve.index, curve.values, color="0.8", lw=0.7, alpha=0.45,
                zorder=1)

    if winner_key is not None:
        best = table[table["candidate"] == winner_key]
        curve = best.groupby("effect_size")["power"].mean().sort_index()
        ax.plot(curve.index, curve.values, color="black", lw=2.0, zorder=3,
                marker="o", ms=3.5, label=_label(winner_key))

    ax.axhline(power_threshold, color="red", ls="--", lw=1.2, zorder=2,
               label=f"power threshold ({power_threshold:g})")
    ax.axvline(0.0, color="0.5", lw=0.8, zorder=0)

    winner_design = result.search.winner
    if winner_design is not None and winner_design.mde is not None:
        mde = float(winner_design.mde)
        ax.axvline(mde, color="red", ls=":", lw=1.4, zorder=2)
        # Anchored above the curve so it clears the legend in the lower right.
        ax.annotate(f"MDE {mde:+.2f}", xy=(mde, 1.0),
                    xytext=(5, -12), textcoords="offset points",
                    color="red", fontsize=9, va="top", ha="left")

    ax.set_xlabel("Injected effect size (proportional lift)")
    ax.set_ylabel("Power (detection rate across placements)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Simulated power across lookback placements")
    ax.legend(frameon=False, loc="lower center", fontsize=8)


def _fit_panel(ax, result, index=None):
    """The test region's aggregate against its synthetic counterfactual."""
    winner = result.search.winner
    if winner is None:  # pragma: no cover - guarded by the caller
        return
    cf = np.asarray(winner.counterfactual, dtype=float)
    observed = getattr(winner, "observed", None)
    x = index if index is not None else np.arange(cf.shape[0])

    if observed is not None:
        ax.plot(x, np.asarray(observed, dtype=float), color="black", lw=1.5,
                label="Test markets (observed)")
    ax.plot(x, cf, color="red", ls="--", lw=1.5, label="Synthetic control")

    n_pre = getattr(winner, "n_pre", None)
    if n_pre is not None and 0 < n_pre < cf.shape[0]:
        ax.axvline(x[n_pre], color="0.4", lw=1.0, ls=":")

    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome")
    ax.set_title(f"Design fit (pre-period scaled L2 {winner.scaled_l2:.3f})")
    ax.legend(frameon=False, fontsize=8)


def plot_sdidgeo_design(
    result, *, power_threshold: float = 0.8, theme=None,
    figsize=(12, 4.8), save_path: Optional[str] = None, show: bool = False,
):
    """Plot an SDIDGEO design: the power curve and the fit it rests on.

    Parameters
    ----------
    result : SDIDGEOResults
        The design returned by ``SDIDGEO(...).fit()``.
    power_threshold : float, default 0.8
        Threshold line to draw; match the config used for the design.
    theme : dict or str, optional
        rcParams dict or named Matplotlib style overriding the house style.
    figsize : tuple, default (12, 4.8)
    save_path : str, optional
        Write the figure here.
    show : bool, default False
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no candidate cleared the power threshold, so there is no design.
    """
    if result.search is None or result.search.winner is None:
        raise ValueError(
            "no winning design to plot: no candidate cleared the power "
            "threshold. Widen effect_sizes or lower power_threshold.")

    with mlsynth_style(theme):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _power_panel(axes[0], result, power_threshold)
        _fit_panel(axes[1], result)
        units = " + ".join(result.selected_units or ())
        mde = result.metadata.get("winner_mde")
        title = f"SDIDGEO design — {units}"
        if mde is not None:
            title += f"   (MDE {mde:+.2f})"
        fig.suptitle(title, fontsize=12)
        return _finish(fig, save_path, show)


def plot_mde_ranking(
    result, *, top: int = 12, theme=None, figsize=(8, 5),
    save_path: Optional[str] = None, show: bool = False,
):
    """Rank the shortlisted test regions by minimum detectable effect.

    The market-selection view: which regions can detect the smallest lift, with
    the chosen one highlighted.
    """
    shortlist = result.power
    if shortlist is None or shortlist.empty:
        raise ValueError("the shortlist is empty: no candidate cleared the "
                         "power threshold.")

    best = (shortlist.sort_values("rank")
            .drop_duplicates(subset="candidate")
            .head(int(top))[::-1])
    labels = [_label(c) for c in best["candidate"]]
    values = best["mde"].abs().to_numpy()
    winner_units = list(result.selected_units or ())
    colours = ["red" if sorted(map(str, c)) == winner_units else "0.65"
               for c in best["candidate"]]

    with mlsynth_style(theme):
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, values, color=colours)
        ax.set_xlabel("Minimum detectable effect (absolute)")
        ax.set_title("Candidate test regions, best first")
        ax.tick_params(axis="y", labelsize=8)
        return _finish(fig, save_path, show)
