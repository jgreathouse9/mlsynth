"""Assemble DMLFM posterior draws into the standard result containers."""

from __future__ import annotations

import numpy as np

from ...config_models import (
    BaseEstimatorResults, EffectsResults, FitDiagnosticsResults,
    InferenceResults, MethodDetailsResults, TimeSeriesResults,
)
from .setup import DMLFMInputs
from .sampler import DMLFMDraws

SHORT_PRE_PERIOD = 20  # Supplementary Table A4: below this the coverage degrades


def _label(value):
    """A plain Python scalar, so cohort keys survive the text boundary."""
    return value.item() if hasattr(value, "item") else value


def _cohorts(adoption: np.ndarray, gap_mean: np.ndarray, time_labels: np.ndarray):
    """Cohort ATTs keyed by adoption label, and an event study by relative time.

    Mirrors the aggregation MCNNM reports under staggered adoption, so the two
    estimators' staggered output reads the same way.
    """
    cohort_att, by_cohort = {}, {}
    for k, a in enumerate(adoption):
        by_cohort.setdefault(int(a), []).append(k)
    for a, members in sorted(by_cohort.items()):
        cohort_att[_label(time_labels[a])] = float(gap_mean[members, a:].mean())

    by_event: dict = {}
    n_periods = gap_mean.shape[1]
    for k, a in enumerate(adoption):
        for t in range(n_periods):
            by_event.setdefault(int(t - a), []).append(gap_mean[k, t])
    event_study = {e: float(np.mean(v)) for e, v in sorted(by_event.items())}
    return cohort_att, event_study


def assemble(inputs: DMLFMInputs, draws: DMLFMDraws, observed: np.ndarray,
             alpha: float) -> BaseEstimatorResults:
    n_tr, n_periods = inputs.n_treated, inputs.n_periods
    adoption = np.asarray(inputs.adoption_index, dtype=int)

    # The sampler stacks the treated units in panel order, each in time order.
    cf_draws = draws.counterfactual.reshape(n_tr, n_periods, -1)
    obs = np.asarray(observed, float).reshape(n_tr, n_periods)
    gap_draws = obs[:, :, None] - cf_draws                   # (n_tr, T, ndraw)

    # Post-adoption cells, per treated unit. Under one cohort this is the
    # single pre/post split; under staggered adoption each unit has its own.
    post = np.zeros((n_tr, n_periods), dtype=bool)
    for k, a in enumerate(adoption):
        post[k, a:] = True

    att_draws = gap_draws[post].mean(axis=0)                 # (ndraw,)
    lo, hi = np.quantile(att_draws, [alpha / 2, 1 - alpha / 2])
    cf_mean = cf_draws.mean(axis=2)
    gap_mean = obs - cf_mean
    scale = np.abs(cf_mean[post]).mean()

    cohort_att, event_study = _cohorts(adoption, gap_mean, inputs.time_labels)
    staggered = len(cohort_att) > 1

    effects = EffectsResults(
        att=float(att_draws.mean()),
        att_percent=float(100.0 * att_draws.mean() / scale) if scale > 0 else None,
        additional_effects={
            "att_median": float(np.median(att_draws)),
            "att_sd": float(att_draws.std(ddof=1)),
            "cohort_att": cohort_att,
            "event_study": event_study,
        })

    fit = FitDiagnosticsResults(
        rmse_pre=float(np.sqrt((gap_mean[~post] ** 2).mean())),
        pre_periods=int(inputs.pre_periods),
        post_periods=int(n_periods - inputs.pre_periods))

    ts = TimeSeriesResults(
        observed_outcome=obs.T,
        counterfactual_outcome=cf_mean.T,
        estimated_gap=gap_mean.T,
        time_periods=np.asarray(inputs.time_labels).reshape(-1, 1))

    inference = InferenceResults(
        method="bayesian posterior predictive",
        confidence_level=float(1.0 - alpha),
        ci_lower=float(lo), ci_upper=float(hi),
        details={
            "per_period_lower": np.quantile(gap_draws, alpha / 2, axis=2).T.tolist(),
            "per_period_upper": np.quantile(gap_draws, 1 - alpha / 2, axis=2).T.tolist(),
        })

    details = MethodDetailsResults(
        method_name="DMLFM",
        parameters={
            "prior": inputs.prior,
            "r": inputs.r,
            "ar1": inputs.ar1,
            "niter": inputs.niter,
            "burn": inputs.burn,
            "draws_kept": int(cf_draws.shape[2]),
            "treated_unit": inputs.treated_name,
            "treated_units": list(inputs.treated_names),
            "adoption_periods": [_label(inputs.time_labels[a]) for a in adoption],
            "staggered": bool(staggered),
            "treated_cells": int(post.sum()),
            "short_pre_period": bool(inputs.pre_periods < SHORT_PRE_PERIOD),
        })

    omega_abs = (np.abs(draws.omega_gamma).mean(axis=1) if draws.omega_gamma.size
                 else np.zeros(0))
    extras = {
        "att_draws": att_draws,
        "gap_draws": gap_draws,
        "counterfactual_draws": cf_draws,
        "post_mask": post,
        "adoption_index": adoption,
        "omega_gamma_abs": omega_abs,
        "omega_gamma_spectrum": np.sort(omega_abs)[::-1],
        "gamma_mean": (draws.gamma.mean(axis=2) if draws.gamma.size
                       else np.zeros((inputs.n_units, 0))),
        "factors_mean": (draws.factors.mean(axis=2) if draws.factors.size
                         else np.zeros((inputs.n_periods, 0))),
        "phi": draws.phi.mean(axis=1) if draws.phi.size else np.zeros(0),
        "sigma2": float(draws.sigma2.mean()),
    }

    return BaseEstimatorResults(
        effects=effects, fit_diagnostics=fit, time_series=ts,
        inference=inference, method_details=details,
        additional_outputs=extras)
