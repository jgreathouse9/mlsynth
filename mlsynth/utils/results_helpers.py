"""Shared helpers for building standardized result sub-models.

Lets each estimator expose its counterfactual, weights, and treatment-effect
metrics through the same pydantic models (no black boxes), computed from one
consistent source so ATT / %ATT / gap / pre+post RMSE / R-squared are derived
identically across every estimator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from . import effectutils as em
from . import fitutils as fm
from ..config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..exceptions import MlsynthConfigError


def normalize_counterfactual_band(
    lower: Any,
    upper: Any,
    *,
    time_periods: Any,
    periods: Any = None,
):
    """Spread a (possibly post-only) counterfactual band onto the full time axis.

    Returns full-length ``(lower, upper)`` arrays aligned to ``time_periods``,
    NaN where the band does not reach, or ``(None, None)`` when no band is given.
    Bounds come in pairs; a one-sided band or a length mismatch is a malformed
    band (:class:`~mlsynth.exceptions.MlsynthConfigError`). When ``periods`` is
    given the band aligns by matching those labels into ``time_periods``;
    otherwise a full-length band is used as-is. This is the single aligner behind
    the canonical ``TimeSeriesResults`` band fields.
    """
    have_lo = lower is not None
    have_hi = upper is not None
    if have_lo != have_hi:
        raise MlsynthConfigError(
            "A prediction interval needs both 'lower' and 'upper'; got one side.")
    if not have_lo:
        return None, None

    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.size != hi.size:
        raise MlsynthConfigError(
            f"Prediction-interval 'lower' and 'upper' differ in length "
            f"({lo.size} vs {hi.size}).")

    axis = np.asarray(time_periods).reshape(-1)
    n = axis.size
    full_lo = np.full(n, np.nan)
    full_hi = np.full(n, np.nan)
    if periods is not None:
        band_p = np.asarray(periods).reshape(-1)
        if band_p.size != lo.size:
            raise MlsynthConfigError(
                f"Band 'periods' length ({band_p.size}) does not match the "
                f"bounds ({lo.size}).")
        index = {p: i for i, p in enumerate(axis.tolist())}
        for p, l, u in zip(band_p.tolist(), lo, hi):
            if p not in index:
                raise MlsynthConfigError(
                    f"Band period {p!r} is not in the counterfactual time axis.")
            full_lo[index[p]] = l
            full_hi[index[p]] = u
    elif lo.size == n:
        full_lo[:] = lo
        full_hi[:] = hi
    else:
        raise MlsynthConfigError(
            f"Prediction interval has length {lo.size} but the counterfactual "
            f"has length {n}; pass 'periods' to align a post-only band.")
    return full_lo, full_hi


def make_weights_results(
    donor_weights: Dict[Any, float],
    constraint: str,
    extra: Optional[Dict[str, Any]] = None,
) -> WeightsResults:
    """Wrap a ``{donor: weight}`` mapping in a standardized WeightsResults.

    Parameters
    ----------
    donor_weights : dict
        Mapping of donor unit name -> weight.
    constraint : str
        Human-readable description of the weight constraint (e.g.
        ``"simplex (non-negative, sum to 1)"`` or
        ``"unconstrained regression weights"``).
    extra : dict, optional
        Additional fields merged into ``summary_stats`` (e.g. time weights,
        intercepts, per-unit weight maps).
    """
    # WeightsResults.donor_weights requires string keys; coerce so unit ids
    # that are integers (a common panel convention) validate cleanly.
    donor_weights = {str(k): float(v) for k, v in (donor_weights or {}).items()}
    w = np.array(list(donor_weights.values()), dtype=float) if donor_weights \
        else np.array([])
    summary: Dict[str, Any] = {
        "sum_of_weights": float(w.sum()) if w.size else 0.0,
        "n_donors": int(w.size),
        "n_nonzero": int((np.abs(w) > 1e-6).sum()),
        "n_negative": int((w < 0).sum()),
        "max_abs_weight": float(np.abs(w).max()) if w.size else 0.0,
        "constraint": constraint,
    }
    if extra:
        summary.update(extra)
    return WeightsResults(donor_weights=donor_weights, summary_stats=summary)


def effect_metrics(
    observed_outcome: np.ndarray,
    counterfactual_outcome: np.ndarray,
    n_pre_periods: int,
    n_post_periods: int,
) -> Dict[str, Any]:
    """Canonical treatment-effect / fit metrics from two outcome paths.

    The single source of truth for the series-derived quantities every
    effect estimator reports, so they are computed identically everywhere:
    ATT (mean post-period gap), percent ATT, the per-period gap, pre- and
    post-period RMSE, and pre-period R-squared.

    Parameters
    ----------
    observed_outcome, counterfactual_outcome : np.ndarray
        Treated-unit observed and counterfactual paths over all ``T`` periods.
    n_pre_periods, n_post_periods : int
        Number of pre- and post-treatment periods (``T0`` and ``T1``).

    Returns
    -------
    dict
        ``att``, ``att_percent``, ``rmse_pre``, ``rmse_post``,
        ``r_squared_pre`` (floats; ``nan`` where undefined) and ``gap``
        (the full-length ``observed - counterfactual`` array).
    """
    obs = em._ravel(observed_outcome)
    full_gap = em.gap(obs, counterfactual_outcome)
    pre_gap, post_gap = em.split_pre_post(full_gap, n_pre_periods, n_post_periods)
    obs_pre, _ = em.split_pre_post(obs, n_pre_periods, n_post_periods)
    _, cf_post = em.split_pre_post(counterfactual_outcome, n_pre_periods, n_post_periods)

    att_value = em.att(post_gap)
    return {
        "att": att_value,
        "att_percent": em.percent_att(att_value, cf_post),
        "rmse_pre": fm.rmse(pre_gap) if n_pre_periods > 0 else float("nan"),
        "rmse_post": fm.rmse(post_gap) if n_post_periods > 0 else float("nan"),
        "r_squared_pre": fm.r_squared(obs_pre, pre_gap) if n_pre_periods > 0 else float("nan"),
        "gap": full_gap,
    }


def build_effect_submodels(
    observed_outcome: np.ndarray,
    counterfactual_outcome: np.ndarray,
    n_pre_periods: int,
    n_post_periods: int,
    *,
    time_periods: Optional[np.ndarray] = None,
    weights: Optional[WeightsResults] = None,
    inference: Optional[InferenceResults] = None,
    method_name: Optional[str] = None,
    is_recommended: bool = True,
    att_std_err: Optional[float] = None,
    effects_overrides: Optional[Dict[str, Any]] = None,
    fit_overrides: Optional[Dict[str, Any]] = None,
    additional_effects: Optional[Dict[str, Any]] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
    intervention_time: Optional[Any] = None,
    prediction_interval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the standardized EffectResult sub-models from two outcome paths.

    Computes the series-derived metrics via :func:`effect_metrics` and maps
    them into ``effects`` / ``time_series`` / ``fit_diagnostics``, attaching
    the supplied ``weights`` / ``inference`` / ``method_details``. Returns a
    ``{sub-model name: model}`` dict ready to splat onto a result container.

    ``effects_overrides`` / ``fit_overrides`` let an estimator pin a value it
    computes authoritatively (e.g. a method-specific ATT or R-squared) instead
    of the canonical one, so validated numbers never drift.
    """
    m = effect_metrics(observed_outcome, counterfactual_outcome, n_pre_periods, n_post_periods)

    effects_kwargs: Dict[str, Any] = {
        "att": m["att"],
        "att_percent": m["att_percent"],
        "att_std_err": att_std_err,
        "additional_effects": additional_effects,
    }
    if effects_overrides:
        effects_kwargs.update(effects_overrides)

    fit_kwargs: Dict[str, Any] = {
        "rmse_pre": m["rmse_pre"],
        "rmse_post": m["rmse_post"],
        "r_squared_pre": m["r_squared_pre"],
        "additional_metrics": additional_metrics,
    }
    if fit_overrides:
        fit_kwargs.update(fit_overrides)

    obs = np.asarray(observed_outcome, dtype=float).ravel()
    ts_kwargs: Dict[str, Any] = dict(
        observed_outcome=obs,
        counterfactual_outcome=np.asarray(counterfactual_outcome, dtype=float).ravel(),
        estimated_gap=m["gap"],
        time_periods=None if time_periods is None else np.asarray(time_periods),
        intervention_time=intervention_time,
    )
    if prediction_interval:
        axis = (np.asarray(time_periods) if time_periods is not None
                else np.arange(obs.size))
        pi = dict(prediction_interval)
        lo, hi = normalize_counterfactual_band(
            pi.get("lower"), pi.get("upper"), time_periods=axis,
            periods=pi.get("periods"))
        lo_s, hi_s = normalize_counterfactual_band(
            pi.get("lower_simultaneous"), pi.get("upper_simultaneous"),
            time_periods=axis, periods=pi.get("periods"))
        ts_kwargs.update(
            counterfactual_lower=lo, counterfactual_upper=hi,
            counterfactual_lower_simultaneous=lo_s,
            counterfactual_upper_simultaneous=hi_s,
            prediction_interval_level=pi.get("level"),
            prediction_interval_kind=pi.get("kind"),
        )
    submodels: Dict[str, Any] = {
        "effects": EffectsResults(**effects_kwargs),
        "time_series": TimeSeriesResults(**ts_kwargs),
        "fit_diagnostics": FitDiagnosticsResults(**fit_kwargs),
    }
    if weights is not None:
        submodels["weights"] = weights
    if inference is not None:
        submodels["inference"] = inference
    if method_name is not None:
        submodels["method_details"] = MethodDetailsResults(
            method_name=method_name, is_recommended=is_recommended
        )
    return submodels
