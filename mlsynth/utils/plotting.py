"""Shared plotting primitives for mlsynth.

A small, style-carrying :class:`Plotter` with one method per common plot
archetype. Module-specific plotter helpers build their figures by calling these
archetype methods -- passing an existing Matplotlib ``Axes`` so they can
compose multi-panel layouts -- and then add any bespoke panels of their own.

The most common archetype, by far, is **observed vs. counterfactual(s)**; gap
plots and event-study plots are intended to follow as further methods.
Estimators with unusual outputs (DSC, ISCM, CTSC, ...) are free to ignore this
class and plot however they need.

The class carries the *style* (colors, intervention-line look, figure size) so
that every module routed through it renders consistently, while leaving figure
lifecycle (creation, saving, showing) to the caller.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np

_DEFAULT_CF_COLORS = ("red", "blue", "green", "purple", "orange", "brown")

CounterfactualLike = Union[np.ndarray, Sequence[np.ndarray]]

#: Preferred display font; falls back through the ``font.sans-serif`` chain
#: below if it is not installed. Override by reassigning before plotting.
MLSYNTH_FONT = "Inter"

#: The default mlsynth plot style (Matplotlib rcParams). Applied per-plot via
#: :func:`mlsynth_style` (scoped, no global side effects) or globally via
#: :func:`apply_mlsynth_style`.
MLSYNTH_RC = {
    # Figure
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "figure.facecolor": "white",
    # Grid
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.40,
    "grid.color": "#1428A0",
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": [MLSYNTH_FONT, "Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "medium",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "medium",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "#DDDDDD",
    "legend.fontsize": 12,
    # Lines
    "lines.linewidth": 1,
    "lines.antialiased": True,
    # Clean axes (modern slide look)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
}


def mlsynth_style():
    """Context manager applying :data:`MLSYNTH_RC` to plots created within it.

    Scoped via ``matplotlib.rc_context`` so the user's global rcParams are
    untouched. Wrap the *entire* plotting block (figure creation included), as
    rcParams are read when artists are created::

        with mlsynth_style():
            fig, ax = plt.subplots()
            ...
    """
    import matplotlib.pyplot as plt

    return plt.rc_context(MLSYNTH_RC)


def apply_mlsynth_style() -> None:
    """Apply the mlsynth style globally (mutates ``matplotlib.rcParams``)."""
    import matplotlib as mpl

    mpl.rcParams.update(MLSYNTH_RC)



class Plotter:
    """Render common synthetic-control plot archetypes with a consistent style.

    Parameters
    ----------
    treated_color : str, default "black"
        Color of the observed (treated) series.
    counterfactual_colors : str or sequence of str, optional
        Color cycle for counterfactual lines. A single string is accepted.
        Defaults to a six-color cycle.
    intervention_color : str, default "grey"
        Color of the vertical intervention marker.
    figsize : tuple, default (7, 5)
        Figure size used only when a method must create its own figure.
    """

    def __init__(
        self,
        *,
        treated_color: str = "black",
        counterfactual_colors: Optional[Union[str, Sequence[str]]] = None,
        intervention_color: str = "grey",
        figsize: tuple = (7, 5),
    ) -> None:
        self.treated_color = treated_color
        if counterfactual_colors is None:
            counterfactual_colors = list(_DEFAULT_CF_COLORS)
        elif isinstance(counterfactual_colors, str):
            counterfactual_colors = [counterfactual_colors]
        self.counterfactual_colors = list(counterfactual_colors)
        self.intervention_color = intervention_color
        self.figsize = figsize

    @staticmethod
    def _as_counterfactual_list(counterfactuals: CounterfactualLike) -> List[np.ndarray]:
        """Normalize a single series, a 2D array of columns, or a list to a list."""
        if isinstance(counterfactuals, np.ndarray):
            if counterfactuals.ndim == 1:
                return [counterfactuals]
            return [counterfactuals[:, i] for i in range(counterfactuals.shape[1])]
        return [np.asarray(cf) for cf in counterfactuals]

    def observed_vs_counterfactual(
        self,
        times: Sequence[Any],
        observed: np.ndarray,
        counterfactuals: CounterfactualLike,
        *,
        labels: Optional[Sequence[str]] = None,
        treated_label: Any = "Treated",
        intervention: Optional[Any] = None,
        outcome: str = "",
        time: str = "",
        title: str = "Observed vs. counterfactual",
        ax: Optional["object"] = None,
    ) -> "object":
        """Plot the observed series against one or more counterfactual series.

        The workhorse archetype: a treated/observed line plus any number of
        dashed counterfactual lines, with an optional vertical intervention
        marker. Pass ``ax`` to draw into an existing subplot (for multi-panel
        figures); otherwise a standalone figure is created.

        Parameters
        ----------
        times : sequence
            X-axis values (time labels), length ``T``.
        observed : np.ndarray
            Observed treated outcome, shape ``(T,)``.
        counterfactuals : np.ndarray or sequence of np.ndarray
            One counterfactual ``(T,)``, a 2D array of column-counterfactuals,
            or a list of ``(T,)`` arrays.
        labels : sequence of str, optional
            Legend labels for the counterfactuals (one per series).
        treated_label : Any, default "Treated"
            Identifier shown for the observed series.
        intervention : Any, optional
            X position of the intervention marker; omitted if None.
        outcome, time : str
            Axis labels.
        title : str
            Panel title.
        ax : matplotlib Axes, optional
            Existing axis to draw on. If None, a new figure/axis is created.

        Returns
        -------
        matplotlib.axes.Axes
            The axis drawn on (``ax.figure`` gives the figure).
        """
        import matplotlib.pyplot as plt

        cfs = self._as_counterfactual_list(counterfactuals)
        if labels is None:
            labels = (["Synthetic"] if len(cfs) == 1
                      else [f"Counterfactual {i + 1}" for i in range(len(cfs))])

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        times = np.asarray(times)
        ax.plot(times, np.asarray(observed).reshape(-1),
                color=self.treated_color, label=f"Treated ({treated_label})")
        for i, cf in enumerate(cfs):
            color = self.counterfactual_colors[i % len(self.counterfactual_colors)]
            ax.plot(times, np.asarray(cf).reshape(-1), linestyle="--",
                    color=color, label=labels[i])
        if intervention is not None:
            ax.axvline(intervention, color=self.intervention_color,
                       linestyle=":", linewidth=1)
        ax.set_xlabel(time)
        ax.set_ylabel(outcome)
        ax.set_title(title)
        ax.legend()  # frame styling comes from the active rcParams
        return ax
