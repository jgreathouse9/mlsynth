"""Shared helpers for building standardized result sub-models.

Lets each estimator expose its counterfactual, weights, and treatment-effect
metrics through the same pydantic models (no black boxes), computed from one
consistent source so ATT / %ATT / gap / pre+post RMSE / R-squared are derived
identically across every estimator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)


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
    obs = np.asarray(observed_outcome, dtype=float).ravel()
    cf = np.asarray(counterfactual_outcome, dtype=float).ravel()
    gap = obs - cf
    pre = slice(0, n_pre_periods)
    post = slice(n_pre_periods, n_pre_periods + n_post_periods)

    pre_resid = gap[pre]
    rmse_pre = float(np.sqrt(np.mean(pre_resid ** 2))) if n_pre_periods > 0 else float("nan")
    obs_pre = obs[pre]
    var_pre = float(np.mean((obs_pre - obs_pre.mean()) ** 2)) if n_pre_periods > 0 else 0.0
    r_squared_pre = (
        float(1.0 - np.mean(pre_resid ** 2) / var_pre)
        if n_pre_periods > 0 and var_pre != 0.0
        else float("nan")
    )

    if n_post_periods > 0:
        post_gap = gap[post]
        att = float(np.mean(post_gap))
        cf_post_mean = float(np.mean(cf[post]))
        att_percent = float(100.0 * att / cf_post_mean) if cf_post_mean != 0.0 else float("nan")
        rmse_post = float(np.sqrt(np.mean(post_gap ** 2)))
    else:
        att = att_percent = rmse_post = float("nan")

    return {
        "att": att,
        "att_percent": att_percent,
        "rmse_pre": rmse_pre,
        "rmse_post": rmse_post,
        "r_squared_pre": r_squared_pre,
        "gap": gap,
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
    submodels: Dict[str, Any] = {
        "effects": EffectsResults(**effects_kwargs),
        "time_series": TimeSeriesResults(
            observed_outcome=obs,
            counterfactual_outcome=np.asarray(counterfactual_outcome, dtype=float).ravel(),
            estimated_gap=m["gap"],
            time_periods=None if time_periods is None else np.asarray(time_periods),
        ),
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
