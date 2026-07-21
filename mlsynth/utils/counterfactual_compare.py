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
    weights: Optional[Dict[str, float]]   # donor weights, if the spec carried any


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
    weights: Optional[Mapping[str, Any]] = None

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
        weights = getattr(spec, "donor_weights", None)
        # Canonical per-period band (already aligned to the counterfactual's
        # axis) is the source of truth; fall back to the legacy details mapping
        # for results that predate it.
        if getattr(ts, "counterfactual_lower", None) is not None:
            lower = ts.counterfactual_lower
            upper = ts.counterfactual_upper
        else:
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
        weights = spec.get("weights")
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
    weights_dict = (
        None if not weights
        else {str(k): float(v) for k, v in dict(weights).items()}
    )
    return _Curve(
        label=label, time=time, counterfactual=cf,
        lower=full_lo, upper=full_hi,
        att=(None if att is None else float(att)),
        pre_rmse=(None if pre_rmse is None else float(pre_rmse)),
        observed=obs_arr,
        weights=weights_dict,
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
    weights : pd.DataFrame
        Long form, one row per (method, donor): ``method``, ``donor``,
        ``weight``. Empty when no method exposes donor weights.
    """

    curves: pd.DataFrame
    summary: pd.DataFrame
    observed: Optional[pd.Series] = None
    weights: pd.DataFrame = None  # type: ignore[assignment]

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Overlay the counterfactuals (the plot form). See
        :func:`plot_counterfactual_comparison`."""
        return plot_counterfactual_comparison(self, ax=ax, **kwargs)

    def plot_weights(self, ax: Any = None, **kwargs: Any) -> Any:
        """Grouped bar chart of each method's donor weights (the plot form).
        See :func:`plot_weights_comparison`."""
        return plot_weights_comparison(self, ax=ax, **kwargs)


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
    weight_rows: List[Dict[str, Any]] = []
    for c in curves:
        summary_rows.append({
            "method": c.label,
            "att": np.nan if c.att is None else c.att,
            "pre_rmse": np.nan if c.pre_rmse is None else c.pre_rmse,
        })
        for t, cf, lo, hi in zip(c.time, c.counterfactual, c.lower, c.upper):
            rows.append({"method": c.label, "period": t,
                         "counterfactual": cf, "lower": lo, "upper": hi})
        if c.weights:
            for donor, w in c.weights.items():
                weight_rows.append({"method": c.label, "donor": donor,
                                    "weight": w})

    curves_df = pd.DataFrame(
        rows, columns=["method", "period", "counterfactual", "lower", "upper"]
    )
    summary_df = pd.DataFrame(summary_rows).set_index("method")
    weights_df = pd.DataFrame(weight_rows, columns=["method", "donor", "weight"])

    observed_series = _resolve_observed(observed, curves, global_time)

    if fit_window is not None:
        summary_df["window_rmse"] = _window_rmse(
            curves, observed_series, fit_window
        )

    return CounterfactualComparison(
        curves=curves_df, summary=summary_df, observed=observed_series,
        weights=weights_df,
    )


def compare_estimators(
    estimators: Any,
    *,
    show_bands: bool,
    observed: Any = None,
    time: Any = None,
    fit_window: Optional[Tuple[Any, Any]] = None,
) -> CounterfactualComparison:
    """Fit several observational estimators on one panel and compare them.

    The observational twin of :func:`mlsynth.utils.design_compare.compare_methods`.
    Hand it fully-configured estimator instances (so every option is yours); it
    fits each, reads its counterfactual and canonical per-period band off the
    standardized contract, and returns a :class:`CounterfactualComparison`
    (summary table + ``.plot()`` overlay).

    Parameters
    ----------
    estimators : sequence or mapping
        Estimator instances (each with a ``.fit()``), or already-fitted
        :class:`~mlsynth.config_models.BaseEstimatorResults`. A bare sequence is
        labelled by class name (a numeric suffix disambiguates collisions); pass
        a ``{label: estimator}`` mapping to name them yourself.
    show_bands : bool
        Required. When True, each method's canonical prediction interval (where
        it carries one) is included so ``.plot()`` shades it; when False, only
        the counterfactual lines are compared. There is no default -- the choice
        is explicit.
    observed, time, fit_window
        Forwarded to :func:`compare_counterfactuals`.

    Returns
    -------
    CounterfactualComparison
    """
    if isinstance(estimators, ABCMapping):
        items = list(estimators.items())
    else:
        items = []
        seen: Dict[str, int] = {}
        for est in estimators:
            name = type(est).__name__
            seen[name] = seen.get(name, 0) + 1
            label = name if seen[name] == 1 else f"{name} ({seen[name]})"
            items.append((label, est))
    if not items:
        raise MlsynthConfigError("Pass at least one estimator to compare.")

    methods: "Dict[str, Any]" = {}
    observed_ref: Optional[np.ndarray] = None
    for label, est in items:
        res = est.fit() if hasattr(est, "fit") else est
        ts = getattr(res, "time_series", None)
        # Standard EffectResult path (canonical band on time_series) with a
        # fallback to the flat accessors for any result that still lacks a
        # standardized time_series sub-model.
        if ts is not None and getattr(ts, "counterfactual_outcome", None) is not None:
            cf = ts.counterfactual_outcome
            time = getattr(ts, "time_periods", None)
            obs = getattr(ts, "observed_outcome", None)
            band = None
            if getattr(ts, "has_prediction_interval", False):
                band = (ts.counterfactual_lower, ts.counterfactual_upper)
        else:
            cf = getattr(res, "counterfactual", None)
            time = None
            obs = None
            gap = getattr(res, "gap", None)
            if cf is not None and gap is not None:
                obs = np.asarray(cf, float).reshape(-1) + np.asarray(gap, float).reshape(-1)
            band = getattr(res, "counterfactual_band", None)
        if cf is None or np.all(~np.isfinite(np.asarray(cf, float).reshape(-1))):
            raise MlsynthConfigError(
                f"Method {label!r}: estimator produced no counterfactual to "
                "compare (design results and estimators without a counterfactual "
                "series are not supported here)."
            )
        if obs is not None:
            obs_arr = np.asarray(obs, dtype=float).reshape(-1)
            if observed_ref is None:
                observed_ref = obs_arr
            elif (obs_arr.shape == observed_ref.shape
                  and not np.allclose(obs_arr, observed_ref, equal_nan=True)):
                raise MlsynthConfigError(
                    "compare_estimators expects all estimators fit on the same "
                    f"panel; {label!r} has a different observed series."
                )
        spec: Dict[str, Any] = {
            "counterfactual": cf,
            "time": time,
            "att": getattr(res, "att", None),
            "pre_rmse": getattr(res, "pre_rmse", None),
            "observed": obs,
            # Carry donor weights through so ``cmp.weights`` / ``plot_weights``
            # work; the flat accessor delegates to ``result.weights.donor_weights``.
            "weights": getattr(res, "donor_weights", None),
        }
        if show_bands and band is not None and band[0] is not None:
            spec["lower"] = band[0]
            spec["upper"] = band[1]
        methods[label] = spec

    return compare_counterfactuals(
        methods, observed=observed, time=time, fit_window=fit_window)


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
    band: str = "errorbar",
    capsize: float = 1.5,
    elinewidth: float = 0.7,
    fill_alpha: float = 0.18,
    observed_color: str = "black",
    legend: bool = True,
) -> Any:
    """Overlay each method's counterfactual, with its prediction interval.

    Draws the observed series (when present) plus one line per method; where a
    method carries a prediction interval, its bounds are shown either as
    per-period error bars (``band="errorbar"``, optionally dodged by ``dodge``
    so overlapping bars stay legible) or as a shaded region
    (``band="fill"``). Renders in the in-house mlsynth style and returns the
    axis.

    Parameters
    ----------
    comparison : CounterfactualComparison
        The object returned by :func:`compare_counterfactuals`.
    colors, styles : mapping ``{label: value}``, optional
        Per-method line colour and linestyle overrides; defaults cycle a palette
        and a solid line.
    dodge : float, default 0.0
        Horizontal offset applied to each method's error bars, centred on zero
        (``band="errorbar"`` only).
    band : {"errorbar", "fill"}, default "errorbar"
        How to render each prediction interval: per-period error bars, or a
        shaded band between the bounds.
    capsize, elinewidth : float
        Error-bar cosmetics (``band="errorbar"`` only).
    fill_alpha : float, default 0.18
        Opacity of the shaded band (``band="fill"`` only).
    observed_color : str, default "black"
        Colour of the observed series.
    legend : bool, default True
        Whether to draw the legend.
    """
    import matplotlib.pyplot as plt

    from .plotting import mlsynth_style

    if band not in ("errorbar", "fill"):
        raise ValueError(
            f"band must be 'errorbar' or 'fill', got {band!r}."
        )
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
            band_mask = np.isfinite(lo) & np.isfinite(hi)
            if band_mask.any():
                if band == "fill":
                    ax.fill_between(
                        t[band_mask], lo[band_mask], hi[band_mask],
                        color=color, alpha=fill_alpha, linewidth=0, zorder=2,
                    )
                else:
                    offset = (k - (n - 1) / 2.0) * dodge
                    # The band is an absolute [lower, upper] region; a method's
                    # band need not centre on the plotted point (e.g. scpi's band
                    # sits on its own reconstruction, not Y0.W), so clamp the
                    # error distances to be non-negative rather than let a
                    # non-bracketing band raise in errorbar.
                    ax.errorbar(
                        t[band_mask] + offset, cf[band_mask],
                        yerr=[np.maximum(cf[band_mask] - lo[band_mask], 0.0),
                              np.maximum(hi[band_mask] - cf[band_mask], 0.0)],
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


def plot_weights_comparison(
    comparison: CounterfactualComparison,
    ax: Any = None,
    *,
    colors: Optional[Mapping[str, Any]] = None,
    threshold: float = 1e-3,
    max_donors: Optional[int] = None,
    sort: bool = True,
    label: Any = str,
    bar_width: Optional[float] = None,
    ylabel: str = "donor weight",
    legend: bool = True,
) -> Any:
    """Grouped bar chart of each method's donor weights.

    One group of bars per donor, one bar per method, so the methods' weight
    vectors can be read side by side (the same comparison the counterfactual
    overlay makes, on the weights rather than the paths). Donors with
    negligible weight in every method are dropped.

    Parameters
    ----------
    comparison : CounterfactualComparison
        The object returned by :func:`compare_counterfactuals`; its ``weights``
        frame supplies the donor weights.
    colors : mapping ``{label: value}``, optional
        Per-method bar colour; defaults cycle the palette.
    threshold : float, default 1e-3
        Drop a donor whose ``|weight|`` is below this in every method.
    max_donors : int, optional
        Keep only the ``max_donors`` donors with the largest total weight.
    sort : bool, default True
        Order donors by descending total weight (else by first appearance).
    label : callable, default ``str``
        Maps a donor key to its axis label (e.g. to strip a parenthetical).
    bar_width : float, optional
        Total width shared by a donor's bars; defaults to ``0.8``.
    ylabel : str, default "donor weight"
        Y-axis label.
    legend : bool, default True
        Whether to draw the legend.

    Raises
    ------
    MlsynthConfigError
        When no method exposes donor weights above ``threshold``.
    """
    import matplotlib.pyplot as plt

    from .plotting import mlsynth_style

    colors = dict(colors or {})
    method_order = list(comparison.summary.index)
    w = comparison.weights

    # {donor: {method: weight}} restricted to donors that clear the threshold.
    pivot: Dict[Any, Dict[str, float]] = {}
    totals: Dict[Any, float] = {}
    if w is not None and not w.empty:
        for donor, grp in w.groupby("donor", sort=False):
            by_method = dict(zip(grp["method"], grp["weight"]))
            if max(abs(v) for v in by_method.values()) < threshold:
                continue
            pivot[donor] = by_method
            totals[donor] = float(sum(by_method.values()))

    if not pivot:
        raise MlsynthConfigError(
            "no donor weights to plot: no method exposes donor_weights above "
            f"threshold={threshold!r}."
        )

    donors = list(pivot)
    if sort:
        donors.sort(key=lambda d: -totals[d])
    if max_donors is not None:
        donors = donors[:max_donors]

    n = len(method_order)
    total_w = 0.8 if bar_width is None else bar_width
    bw = total_w / n
    x = np.arange(len(donors))

    def _draw(ax: Any) -> Any:
        for k, method in enumerate(method_order):
            heights = [pivot[d].get(method, 0.0) for d in donors]
            offset = (k - (n - 1) / 2.0) * bw
            ax.bar(x + offset, heights, bw,
                   color=colors.get(method, _PALETTE[k % len(_PALETTE)]),
                   label=str(method))
        ax.set_xticks(x)
        ax.set_xticklabels([label(d) for d in donors])
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        return ax

    if ax is not None:
        return _draw(ax)
    with mlsynth_style():
        _, ax = plt.subplots(figsize=(6, 3.0))
        _draw(ax)
        ax.figure.tight_layout()
        plt.show()
    return ax
