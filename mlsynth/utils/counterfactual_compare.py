"""Cross-method counterfactual comparison on a common time axis.

Different synthetic-control estimators all emit the same observable -- a
counterfactual path for the treated unit, and (when the estimator carries
inference) a per-period prediction interval around it. When several are fit on
one panel, the analyst wants to read them against each other and against the
observed series. Doing that by hand means one ``ax.plot`` per method plus a
second loop to pull the per-period bounds out of ``inference.details``; this
module collapses both into one call.

:func:`compare_counterfactuals` normalizes each method to ``(time,
counterfactual, lower, upper, att, pre_rmse)`` -- reading those straight off a
standardized :class:`~mlsynth.config_models.BaseEstimatorResults` (so the stored
``effects.att`` and ``pre_rmse`` are used, not recomputed) or off an explicit
spec for results that do not expose the standard surface (e.g. the SPILLSYNTH
dispatcher, whose counterfactual is assembled per method). It returns a
:class:`CounterfactualComparison` with a tidy ``curves`` frame (data form), a
``summary`` frame (one row per method), and ``.plot()`` (plot form), mirroring
the data/plot split of :mod:`mlsynth.utils.design_compare`.
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MlsynthConfigError, MlsynthEstimationError


# --------------------------------------------------------------------------- #
# Normalized per-method record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Curve:
    """One method's counterfactual on its own time support."""

    label: str
    time: np.ndarray
    counterfactual: np.ndarray
    lower: np.ndarray            # full length, NaN where no band
    upper: np.ndarray            # full length, NaN where no band
    att: Optional[float]
    pre_rmse: Optional[float]
    observed: Optional[np.ndarray]   # treated series, if the spec carried one


def _as_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise MlsynthEstimationError(f"{name} is empty.")
    return arr


def _align_band(
    lower: Any,
    upper: Any,
    *,
    time: np.ndarray,
    periods: Any,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spread a (possibly short) prediction interval onto the curve's time axis.

    Returns full-length ``(lower, upper)`` arrays, NaN outside the band. Bounds
    come in pairs: one side without the other is a malformed band. They align to
    ``time`` either by an explicit ``periods`` list, or directly when their
    length already matches the curve.
    """
    have_lo = lower is not None
    have_hi = upper is not None
    if have_lo != have_hi:
        raise MlsynthConfigError(
            f"Method {label!r}: a prediction interval needs both 'lower' and "
            "'upper'; got only one side."
        )
    n = time.size
    if not have_lo:
        return np.full(n, np.nan), np.full(n, np.nan)

    lo = _as_1d(lower, f"{label} lower")
    hi = _as_1d(upper, f"{label} upper")
    if lo.size != hi.size:
        raise MlsynthConfigError(
            f"Method {label!r}: 'lower' and 'upper' have different lengths "
            f"({lo.size} vs {hi.size})."
        )

    full_lo = np.full(n, np.nan)
    full_hi = np.full(n, np.nan)
    if periods is not None:
        band_periods = np.asarray(periods).reshape(-1)
        if band_periods.size != lo.size:
            raise MlsynthEstimationError(
                f"Method {label!r}: band 'periods' length ({band_periods.size}) "
                f"does not match the bounds ({lo.size})."
            )
        index = {p: i for i, p in enumerate(time.tolist())}
        for p, l, u in zip(band_periods.tolist(), lo, hi):
            if p not in index:
                raise MlsynthEstimationError(
                    f"Method {label!r}: band period {p!r} is not in the "
                    "counterfactual's time axis."
                )
            full_lo[index[p]] = l
            full_hi[index[p]] = u
    elif lo.size == n:
        full_lo[:] = lo
        full_hi[:] = hi
    else:
        raise MlsynthEstimationError(
            f"Method {label!r}: prediction interval has length {lo.size} but the "
            f"counterfactual has length {n}; pass 'periods' to align them."
        )
    return full_lo, full_hi


def _band_from_details(details: Any) -> Tuple[Any, Any, Any]:
    """Pull ``(lower, upper, periods)`` out of an ``inference.details`` mapping."""
    if not isinstance(details, ABCMapping):
        return None, None, None
    return (
        details.get("counterfactual_lower"),
        details.get("counterfactual_upper"),
        details.get("periods"),
    )


def _normalize(
    label: str,
    spec: Any,
    *,
    global_time: Optional[np.ndarray],
) -> _Curve:
    """Reduce any accepted spec to a :class:`_Curve`."""
    cf: Any
    spec_time: Any = None
    lower = upper = periods = None
    att: Optional[float] = None
    pre_rmse: Optional[float] = None
    observed: Optional[Any] = None

    if hasattr(spec, "time_series") and getattr(spec, "time_series") is not None:
        ts = spec.time_series
        cf = ts.counterfactual_outcome
        if cf is None:
            raise MlsynthEstimationError(
                f"Method {label!r}: result carries no counterfactual_outcome."
            )
        spec_time = ts.time_periods
        observed = ts.observed_outcome
        effects = getattr(spec, "effects", None)
        att = getattr(effects, "att", None) if effects is not None else None
        pre_rmse = getattr(spec, "pre_rmse", None)
        inference = getattr(spec, "inference", None)
        if inference is not None:
            lower, upper, periods = _band_from_details(
                getattr(inference, "details", None)
            )
    elif isinstance(spec, ABCMapping):
        if "counterfactual" not in spec:
            raise MlsynthConfigError(
                f"Method {label!r}: spec dict must contain 'counterfactual'."
            )
        cf = spec["counterfactual"]
        spec_time = spec.get("time")
        lower = spec.get("lower")
        upper = spec.get("upper")
        periods = spec.get("periods")
        att = spec.get("att")
        pre_rmse = spec.get("pre_rmse")
        observed = spec.get("observed")
    else:
        cf = spec  # array-like: the counterfactual itself

    cf = _as_1d(cf, f"{label} counterfactual")

    if spec_time is not None:
        time = np.asarray(spec_time).reshape(-1)
    elif global_time is not None:
        time = np.asarray(global_time).reshape(-1)
    else:
        time = np.arange(cf.size)
    if time.size != cf.size:
        raise MlsynthEstimationError(
            f"Method {label!r}: time axis ({time.size}) and counterfactual "
            f"({cf.size}) have different lengths."
        )

    full_lo, full_hi = _align_band(
        lower, upper, time=time, periods=periods, label=label
    )
    obs_arr = (None if observed is None
               else np.asarray(observed, dtype=float).reshape(-1))
    return _Curve(
        label=label, time=time, counterfactual=cf,
        lower=full_lo, upper=full_hi,
        att=(None if att is None else float(att)),
        pre_rmse=(None if pre_rmse is None else float(pre_rmse)),
        observed=obs_arr,
    )


# --------------------------------------------------------------------------- #
# Public comparison container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CounterfactualComparison:
    """Result of :func:`compare_counterfactuals`.

    Parameters
    ----------
    curves : pd.DataFrame
        Long form, one row per (method, period): ``method``, ``period``,
        ``counterfactual``, ``lower``, ``upper`` (the last two NaN where the
        method has no prediction interval at that period).
    summary : pd.DataFrame
        One row per method (indexed by label), columns ``att`` and ``pre_rmse``
        read from the stored result fields (NaN when a method does not carry
        them).
    observed : pd.Series, optional
        Observed treated series indexed by period, when one was supplied or could
        be read off a standardized result.
    """

    curves: pd.DataFrame
    summary: pd.DataFrame
    observed: Optional[pd.Series] = None

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Overlay the counterfactuals (the plot form). See
        :func:`plot_counterfactual_comparison`."""
        return plot_counterfactual_comparison(self, ax=ax, **kwargs)


def compare_counterfactuals(
    methods: Mapping[str, Any],
    *,
    observed: Any = None,
    time: Any = None,
    fit_window: Optional[Tuple[Any, Any]] = None,
) -> CounterfactualComparison:
    """Line up several methods' counterfactuals on a common time axis.

    Parameters
    ----------
    methods : mapping ``{label: spec}``
        Ordered mapping (label preserved as the method name). Each ``spec`` is
        one of: a standardized :class:`~mlsynth.config_models.BaseEstimatorResults`
        (the counterfactual, time, ``effects.att``, ``pre_rmse`` and any
        ``inference.details`` band are read off it); a mapping with at least
        ``counterfactual`` and optionally ``time``/``lower``/``upper``/
        ``periods``/``att``/``pre_rmse``/``observed``; or a plain array treated
        as the counterfactual.
    observed : array-like or pd.Series, optional
        Observed treated series. A bare array is paired with ``time`` (or the
        first method's time). When omitted, it is taken from the first
        standardized result that carries ``time_series.observed_outcome``.
    time : array-like, optional
        Default x-axis for specs that do not carry their own time index.
    fit_window : (low, high), optional
        When given, add a ``window_rmse`` column to ``summary``: the RMSE of
        observed minus counterfactual over the periods ``low <= t <= high``
        (NaN for a method with no period in the window). Needs an observed
        series -- supplied or read off a standardized result -- else a
        :class:`~mlsynth.exceptions.MlsynthConfigError` is raised. This is the
        fit loss over the matching window, a tighter figure than the stored
        all-pre ``pre_rmse``.

    Returns
    -------
    CounterfactualComparison
        ``.curves`` and ``.summary`` (data form) and ``.plot()`` (plot form).
    """
    if fit_window is not None and len(tuple(fit_window)) != 2:
        raise MlsynthConfigError(
            "fit_window must be a (low, high) pair."
        )
    if not isinstance(methods, ABCMapping):
        raise MlsynthConfigError(
            "methods must be a mapping {label: result-or-spec}."
        )
    if len(methods) == 0:
        raise MlsynthConfigError("methods is empty; pass at least one method.")

    global_time = None if time is None else np.asarray(time).reshape(-1)

    curves: List[_Curve] = [
        _normalize(label, spec, global_time=global_time)
        for label, spec in methods.items()
    ]

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for c in curves:
        summary_rows.append({
            "method": c.label,
            "att": np.nan if c.att is None else c.att,
            "pre_rmse": np.nan if c.pre_rmse is None else c.pre_rmse,
        })
        for t, cf, lo, hi in zip(c.time, c.counterfactual, c.lower, c.upper):
            rows.append({"method": c.label, "period": t,
                         "counterfactual": cf, "lower": lo, "upper": hi})

    curves_df = pd.DataFrame(
        rows, columns=["method", "period", "counterfactual", "lower", "upper"]
    )
    summary_df = pd.DataFrame(summary_rows).set_index("method")

    observed_series = _resolve_observed(observed, curves, global_time)

    if fit_window is not None:
        summary_df["window_rmse"] = _window_rmse(
            curves, observed_series, fit_window
        )

    return CounterfactualComparison(
        curves=curves_df, summary=summary_df, observed=observed_series
    )


def _window_rmse(
    curves: List[_Curve],
    observed: Optional[pd.Series],
    fit_window: Tuple[Any, Any],
) -> List[float]:
    """Per-method RMSE of observed minus counterfactual over the window."""
    if observed is None:
        raise MlsynthConfigError(
            "fit_window needs an observed series; supply observed= or pass "
            "results that carry time_series.observed_outcome."
        )
    lo, hi = fit_window
    obs = observed
    out: List[float] = []
    for c in curves:
        cf = pd.Series(c.counterfactual, index=c.time)
        common = obs.index.intersection(cf.index)
        in_win = [p for p in common if lo <= p <= hi]
        if not in_win:
            out.append(np.nan)
            continue
        resid = obs.loc[in_win].to_numpy() - cf.loc[in_win].to_numpy()
        out.append(float(np.sqrt(np.mean(resid ** 2))))
    return out


def _resolve_observed(
    observed: Any,
    curves: List[_Curve],
    global_time: Optional[np.ndarray],
) -> Optional[pd.Series]:
    """Pick the observed series: explicit arg, else the first result's own."""
    if observed is not None:
        if isinstance(observed, pd.Series):
            return observed
        arr = np.asarray(observed, dtype=float).reshape(-1)
        index = global_time if global_time is not None else curves[0].time
        index = np.asarray(index).reshape(-1)
        if index.size != arr.size:
            raise MlsynthEstimationError(
                "observed length does not match the time axis "
                f"({arr.size} vs {index.size})."
            )
        return pd.Series(arr, index=index, name="observed")
    for c in curves:
        if c.observed is not None and c.observed.size == c.time.size:
            return pd.Series(c.observed, index=c.time, name="observed")
    return None


# --------------------------------------------------------------------------- #
# Plot form
# --------------------------------------------------------------------------- #
_PALETTE = ("C0", "C3", "C1", "C4", "C2", "C5")


def plot_counterfactual_comparison(
    comparison: CounterfactualComparison,
    ax: Any = None,
    *,
    colors: Optional[Mapping[str, Any]] = None,
    styles: Optional[Mapping[str, Any]] = None,
    dodge: float = 0.0,
    capsize: float = 1.5,
    elinewidth: float = 0.7,
    observed_color: str = "black",
    legend: bool = True,
) -> Any:
    """Overlay each method's counterfactual, with prediction-interval error bars.

    Draws the observed series (when present) plus one line per method; where a
    method carries a prediction interval, its bounds are shown as per-period
    error bars, optionally dodged by ``dodge`` per method so overlapping bars
    stay legible. Renders in the in-house mlsynth style and returns the axis.

    Parameters
    ----------
    comparison : CounterfactualComparison
        The object returned by :func:`compare_counterfactuals`.
    colors, styles : mapping ``{label: value}``, optional
        Per-method line colour and linestyle overrides; defaults cycle a palette
        and a solid line.
    dodge : float, default 0.0
        Horizontal offset applied to each method's error bars, centred on zero.
    capsize, elinewidth : float
        Error-bar cosmetics.
    observed_color : str, default "black"
        Colour of the observed series.
    legend : bool, default True
        Whether to draw the legend.
    """
    import matplotlib.pyplot as plt

    from .plotting import mlsynth_style

    colors = dict(colors or {})
    styles = dict(styles or {})
    method_order = list(comparison.summary.index)
    n = len(method_order)

    def _draw(ax: Any) -> Any:
        if comparison.observed is not None:
            obs = comparison.observed
            ax.plot(np.asarray(obs.index), np.asarray(obs.values),
                    color=observed_color, lw=1.7, zorder=5,
                    label=str(obs.name or "observed"))
        for k, method in enumerate(method_order):
            sub = comparison.curves[comparison.curves["method"] == method]
            t = np.asarray(sub["period"], dtype=float)
            cf = np.asarray(sub["counterfactual"], dtype=float)
            color = colors.get(method, _PALETTE[k % len(_PALETTE)])
            ls = styles.get(method, "-")
            ax.plot(t, cf, color=color, ls=ls, lw=1.2, label=str(method))
            lo = np.asarray(sub["lower"], dtype=float)
            hi = np.asarray(sub["upper"], dtype=float)
            band = np.isfinite(lo) & np.isfinite(hi)
            if band.any():
                offset = (k - (n - 1) / 2.0) * dodge
                ax.errorbar(
                    t[band] + offset, cf[band],
                    yerr=[cf[band] - lo[band], hi[band] - cf[band]],
                    fmt="none", ecolor=color, elinewidth=elinewidth,
                    capsize=capsize, alpha=0.8, zorder=4,
                )
        if legend:
            ax.legend()
        return ax

    if ax is not None:
        return _draw(ax)
    with mlsynth_style():
        _, ax = plt.subplots(figsize=(6, 3.5))
        _draw(ax)
        ax.figure.tight_layout()
        plt.show()
    return ax
