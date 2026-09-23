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


def assemble(inputs: DMLFMInputs, draws: DMLFMDraws, observed: np.ndarray,
             alpha: float) -> BaseEstimatorResults:
    cf_draws = draws.counterfactual                    # (T, ndraw)
    gap_draws = observed[:, None] - cf_draws
    post = slice(inputs.pre_periods, inputs.n_periods)
    pre = slice(0, inputs.pre_periods)

    att_draws = gap_draws[post].mean(axis=0)
    lo, hi = np.quantile(att_draws, [alpha / 2, 1 - alpha / 2])
    cf_mean = cf_draws.mean(axis=1)
    gap_mean = observed - cf_mean

    effects = EffectsResults(
        att=float(att_draws.mean()),
        att_percent=float(100.0 * att_draws.mean() / np.abs(cf_mean[post]).mean())
        if np.abs(cf_mean[post]).mean() > 0 else None,
        additional_effects={
            "att_median": float(np.median(att_draws)),
            "att_sd": float(att_draws.std(ddof=1)),
        })

    fit = FitDiagnosticsResults(
        rmse_pre=float(np.sqrt((gap_mean[pre] ** 2).mean())),
        pre_periods=int(inputs.pre_periods),
        post_periods=int(inputs.n_periods - inputs.pre_periods))

    ts = TimeSeriesResults(
        observed_outcome=observed.reshape(-1, 1),
        counterfactual_outcome=cf_mean.reshape(-1, 1),
        estimated_gap=gap_mean.reshape(-1, 1),
        time_periods=np.asarray(inputs.time_labels).reshape(-1, 1))

    inference = InferenceResults(
        method="bayesian posterior predictive",
        confidence_level=float(1.0 - alpha),
        ci_lower=float(lo), ci_upper=float(hi),
        details={
            "per_period_lower": np.quantile(gap_draws, alpha / 2, axis=1).tolist(),
            "per_period_upper": np.quantile(gap_draws, 1 - alpha / 2, axis=1).tolist(),
        })

    details = MethodDetailsResults(
        method_name="DMLFM",
        parameters={
            "r": inputs.r,
            "ar1": inputs.ar1,
            "niter": inputs.niter,
            "burn": inputs.burn,
            "draws_kept": int(cf_draws.shape[1]),
            "treated_unit": inputs.treated_name,
            "short_pre_period": bool(inputs.pre_periods < SHORT_PRE_PERIOD),
        })

    omega_abs = (np.abs(draws.omega_gamma).mean(axis=1) if draws.omega_gamma.size
                 else np.zeros(0))
    extras = {
        "att_draws": att_draws,
        "gap_draws": gap_draws,
        "counterfactual_draws": cf_draws,
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
