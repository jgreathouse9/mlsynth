"""Orchestrator for the SCD estimator: inputs -> :class:`BaseEstimatorResults`.

Runs the point estimator, optionally the repeated-cross-section inference, and
packs the standardized result: the effect path as the estimated gap, the
treated group mean as the observed series, and the per-period confidence bands
(when requested) in the inference details.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from .inference import build_inference_operators, fit_point, per_period_bands
from .structures import SCDInputs


def run_scd(
    inputs: SCDInputs,
    differencing: str,
    compute_inference: bool,
    alpha: float,
    kappa: float,
    n_grid: int,
    grid_radius: float,
    tolerance: float,
    random_state: int,
    plot_config: Optional[Any] = None,
) -> BaseEstimatorResults:
    """Run SCD and return the standardized :class:`BaseEstimatorResults`."""
    T0 = inputs.T0
    post = slice(T0, inputs.Ttot)

    if compute_inference:
        ops = build_inference_operators(inputs, differencing)
        hat_w, theta = ops.hat_w, ops.theta
        bands = per_period_bands(
            ops, inputs, alpha=alpha, kappa=kappa, tol=tolerance,
            n_grid=n_grid, radius=grid_radius, random_state=random_state,
        )
    else:
        fit = fit_point(inputs, differencing)
        hat_w, theta = fit.hat_w, fit.theta
        bands = None

    observed = inputs.group_means[0].copy()          # treated raw group mean
    counterfactual = observed - theta                # gap = observed - counterfactual = theta
    att = float(theta[post].mean())

    donor_weights = {str(name): float(w) for name, w in zip(inputs.donor_names, hat_w)}
    rmse_pre = float(np.sqrt(np.mean(theta[:T0] ** 2)))

    effects = EffectsResults(
        att=att,
        att_std_err=(bands["att_std_err"] if bands else None),
        additional_effects={"effect_path": theta},
    )
    fit_diagnostics = FitDiagnosticsResults(
        rmse_pre=rmse_pre,
        rmse_post=float(np.std(theta[post])),
    )
    time_series = TimeSeriesResults(
        observed_outcome=observed,
        counterfactual_outcome=counterfactual,
        estimated_gap=theta,
        time_periods=np.asarray(inputs.time_labels),
        intervention_time=inputs.time_labels[T0],
    )
    weights = WeightsResults(
        donor_weights=donor_weights,
        summary_stats={"nonzero": int(np.sum(hat_w > 1e-6))},
    )
    inference: Optional[InferenceResults] = None
    if bands is not None:
        inference = InferenceResults(
            ci_lower=bands["att_ci"][0],
            ci_upper=bands["att_ci"][1],
            standard_error=bands["att_std_err"],
            confidence_level=1.0 - alpha,
            method="scd-confidence-set",
            details={
                "period": np.asarray(inputs.time_labels),
                "theta": theta,
                "lower": bands["lower"],
                "upper": bands["upper"],
                "se": bands["se"],
                "sigma2": bands["sigma2"],
                "hatV_trace": bands["hatV_trace"],
                "conf_set_size": bands["conf_set_size"],
                "n_grid": bands["n_grid"],
                "intervention_time": inputs.time_labels[T0],
            },
        )
    method_details = MethodDetailsResults(
        method_name="SCD",
        parameters_used={
            "differencing": differencing,
            "data_type": "repeated cross-sections",
            "compute_inference": compute_inference,
            "alpha": alpha,
            "kappa": kappa,
            "K": inputs.K,
            "T0": T0,
        },
    )
    return BaseEstimatorResults(
        effects=effects,
        fit_diagnostics=fit_diagnostics,
        time_series=time_series,
        weights=weights,
        inference=inference,
        method_details=method_details,
        plot_config=plot_config,
    )
