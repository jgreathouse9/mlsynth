"""Pictures for the placebo-inverted confidence set (Firpo-Possebom 2018).

Two views, matching the two things the inversion produces.

:func:`plot_confidence_set` draws the estimated gap with the confidence set
shaded around it. The set is a set of effect *paths*, not a per-period interval,
so the band is the pair of paths the bounds generate: flat at the bound for the
constant class, a fan opening at the bound's slope for the linear one. It closes
to zero width over the pre-period because every candidate path is zero there.
This is the figure the authors' own ``SCM.CS`` draws when called with
``plot = TRUE``.

:func:`plot_sensitivity` draws the sweep: one interval per assignment tilt
:math:`\\phi`, against a zero line. Where the interval crosses zero the sign of
the estimated effect no longer survives the tilt, and the smallest such
:math:`\\phi` is the breakdown point.

Both take either the objects
:mod:`~mlsynth.utils.vanillasc_helpers.placebo_cs` returns or a fitted
``VanillaSC`` result, and both return their :class:`~matplotlib.figure.Figure`
without displaying or saving it.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ...exceptions import MlsynthPlottingError
from ..plotting import Plotter, mlsynth_style
from .placebo_cs import PlaceboConfidenceSet, SensitivityRow

_METHOD_TAG = "placebo-inverted confidence set"


def _short_reason(reason: str) -> str:
    """A tilt's failure, short enough to sit on an axis.

    The full sentence stays on the result; the axis carries which of the two
    ways the search ends, since that is what the reader compares across tilts.
    """
    low = (reason or "").lower()
    if "unbounded" in low:
        return "unbounded"
    if "empty" in low:
        return "empty"
    return "no set"


def _inference_details(result: Any) -> dict:
    """The ``details`` of a fitted result, once it is established to be ours.

    Refuses a result produced by a different ``inference=`` mode, and a result
    whose search failed -- the latter carries the reason the set is missing, so
    it is reported instead of being rediscovered by the caller.
    """
    inference = getattr(result, "inference", None)
    method = "" if inference is None else (inference.method or "")
    if _METHOD_TAG not in method:
        raise MlsynthPlottingError(
            "this result carries no placebo_cs confidence set to draw "
            f"(its inference method is {method!r}); refit with "
            "inference='placebo_cs'")
    details = dict(inference.details or {})
    if "unavailable_reason" in details:
        raise MlsynthPlottingError(
            "the placebo_cs search produced no confidence set to draw: "
            f"{details['unavailable_reason']}")
    return details


def _paths_from(source: Any, gap: Optional[Sequence[float]]):
    """Normalise the two accepted inputs to ``(gap, lower, upper, times, mark)``."""
    if isinstance(source, PlaceboConfidenceSet):
        if gap is None:
            raise MlsynthPlottingError(
                "plotting a bare PlaceboConfidenceSet needs the estimated gap "
                "as the second argument; pass a fitted result instead to take "
                "it from the fit")
        lower = np.asarray(source.lower_path, dtype=float)
        upper = np.asarray(source.upper_path, dtype=float)
        times = mark = None
    else:
        details = _inference_details(source)
        if "lower_path" not in details or "upper_path" not in details:
            raise MlsynthPlottingError(
                "this placebo_cs result carries no effect paths to draw")
        lower = np.asarray(details["lower_path"], dtype=float)
        upper = np.asarray(details["upper_path"], dtype=float)
        series = getattr(source, "time_series", None)
        if gap is None:
            gap = None if series is None else series.estimated_gap
        times = None if series is None else series.time_periods
        mark = None if series is None else series.intervention_time
    if gap is None:
        raise MlsynthPlottingError("there is no estimated gap to draw")
    gap = np.asarray(gap, dtype=float).reshape(-1)
    if gap.size != lower.size:
        raise MlsynthPlottingError(
            f"the gap has length {gap.size} but the confidence paths have "
            f"length {lower.size}; they must cover the same periods")
    return gap, lower, upper, times, mark


def plot_confidence_set(
    source: Any,
    gap: Optional[Sequence[float]] = None,
    *,
    times: Optional[Sequence[Any]] = None,
    intervention: Optional[Any] = None,
    ax: Optional[Any] = None,
    title: str = "Placebo-inverted confidence set",
    outcome: str = "Treated - synthetic control",
    time: str = "",
    color: Optional[str] = None,
    figsize: tuple = (7, 5),
) -> Any:
    """Draw the gap with the confidence set shaded around it.

    Parameters
    ----------
    source
        A fitted ``VanillaSC`` result from ``inference="placebo_cs"``, or the
        :class:`~mlsynth.utils.vanillasc_helpers.placebo_cs.PlaceboConfidenceSet`
        that :func:`~mlsynth.utils.vanillasc_helpers.placebo_cs.confidence_set`
        returns.
    gap
        The estimated gap, required when ``source`` is a bare confidence set and
        read off the result otherwise.
    times, intervention
        Override the time axis and the intervention marker; both default to what
        the result carries.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, neither shown nor saved.
    """
    import matplotlib.pyplot as plt

    gap, lower, upper, res_times, res_mark = _paths_from(source, gap)
    if times is None:
        times = res_times
    if times is None:
        times = np.arange(gap.size)
    if intervention is None:
        intervention = res_mark

    with mlsynth_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        plotter = Plotter(figsize=figsize)
        plotter.gap(times, gap, intervention=intervention,
                    interval=(lower, upper),
                    interval_label="Confidence set", outcome=outcome,
                    time=time, title=title, color=color, ax=ax)
    return fig


def _rows_from(source: Any):
    """Normalise a sweep to ``(rows, breakdown)``.

    ``rows`` are ``(phi, lower, upper, reason)`` tuples with ``None`` bounds
    where the search failed at that tilt.
    """
    if isinstance(source, (list, tuple)):
        rows = list(source)
        if not rows:
            raise MlsynthPlottingError(
                "the sensitivity sweep is empty: there is no tilt to draw")
        if not all(isinstance(r, SensitivityRow) for r in rows):
            raise MlsynthPlottingError(
                "expected the rows sensitivity_sweep returns")
        out = [(r.phi,
                None if r.confidence_set is None else r.confidence_set.lower,
                None if r.confidence_set is None else r.confidence_set.upper,
                r.reason) for r in rows]
        resolved = [r for r in out if r[1] is not None]
        breakdown = next((p for p, lo, hi, _ in resolved if lo <= 0.0 <= hi), None)
        return out, breakdown

    details = _inference_details(source)
    raw = details.get("sensitivity")
    if not raw:
        raise MlsynthPlottingError(
            "this placebo_cs result carries no sensitivity sweep; refit with "
            "placebo_cs_sweep set to the tilts to try")
    out = [(float(r["phi"]), r.get("lower"), r.get("upper"), r.get("reason", ""))
           for r in raw]
    return out, details.get("breakdown_phi")


def plot_sensitivity(
    source: Any,
    *,
    ax: Optional[Any] = None,
    title: str = "Sensitivity to the assignment tilt",
    ylabel: str = "Effect",
    figsize: tuple = (7, 5),
    color: str = "#2b6cb0",
) -> Any:
    """Draw the swept bounds against the tilt, with the breakdown point marked.

    Parameters
    ----------
    source
        A fitted ``VanillaSC`` result whose ``placebo_cs_sweep`` was set, or the
        list :func:`~mlsynth.utils.vanillasc_helpers.placebo_cs.sensitivity_sweep`
        returns.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, neither shown nor saved.
    """
    import matplotlib.pyplot as plt

    rows, breakdown = _rows_from(source)

    with mlsynth_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # One bar per resolved tilt, drawn as a line and not a collection, so
        # matplotlib's automatic legend placement can see the data it has to
        # avoid.
        resolved = [(p, lo, hi) for p, lo, hi, _ in rows if lo is not None]
        for i, (phi, lo, hi) in enumerate(resolved):
            ax.plot([phi, phi], [lo, hi], color=color, linewidth=6,
                    alpha=0.35, solid_capstyle="butt",
                    label="Confidence set" if i == 0 else None)
            ax.plot([phi, phi], [lo, hi], marker="_", linestyle="none",
                    color=color)

        ax.axhline(0.0, color="black", linewidth=0.8, label="No effect")
        if breakdown is not None:
            ax.axvline(float(breakdown), color="grey", linestyle="--",
                       linewidth=1.2,
                       label=f"Breakdown $\\phi$ = {float(breakdown):g}")

        # A tilt the search could not resolve is on the axis with its reason, so
        # a gap in the sweep is never read as a gap in the evidence. The label
        # is placed in axes coordinates and clipped, so a long reason cannot
        # stretch the figure around the data.
        import matplotlib.transforms as mtransforms

        blend = mtransforms.blended_transform_factory(ax.transData,
                                                      ax.transAxes)
        for phi, lo, _hi, reason in rows:
            if lo is None:
                ax.axvline(phi, color="grey", linestyle=":", linewidth=1.0)
                ax.text(phi, 0.04, _short_reason(reason), rotation=90,
                        fontsize=8, ha="right", va="bottom", color="grey",
                        transform=blend, clip_on=True)

        ax.set_xlabel("Assignment tilt $\\phi$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Headroom so the automatic legend placement has somewhere to go: the
        # sets are vertical bars, which "best" otherwise lands on top of.
        ax.margins(y=0.18)
        ax.legend(framealpha=0.92)
    return fig
