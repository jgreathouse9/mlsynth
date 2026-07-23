"""CSCM orchestration: dataprep -> weights -> rate ratio -> results.

Assembles the standardized :class:`BaseEstimatorResults`. The primary CSCM
estimand is a rate ratio (reported in ``effects.additional_effects`` and, as a
percentage, in ``att_percent``) with a cross-fitted t-interval on
``inference``; the additive ATT and the counterfactual path follow the usual
contract so the result plots and compares like any other estimator.
"""

from __future__ import annotations

import numpy as np

from ...config_models import (
    BaseEstimatorResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..results_helpers import build_effect_submodels, make_weights_results
from .engine import crossfit_rate_ratio, fit_cscm_weights
from .plotter import plot_cscm
from .setup import prepare_cscm_inputs


def run_cscm(config) -> BaseEstimatorResults:
    """Fit CSCM and assemble :class:`BaseEstimatorResults`."""
    prep = prepare_cscm_inputs(
        config.df, config.outcome, config.treat, config.unitid, config.time
    )
    y, Y0 = prep["y"], prep["Y0"]
    T0, T1 = prep["pre_periods"], prep["post_periods"]
    donor_names = prep["donor_names"]
    time_labels = prep["time_labels"]

    # full-sample weights (for the counterfactual + weight table)
    y0post = Y0[T0:].mean(axis=0)
    fit = fit_cscm_weights(
        Y0[:T0], y[:T0], y0post, v_method=config.v_method,
        n_lambda=config.n_lambda, lambda_min_ratio=config.lambda_min_ratio,
        min_1se=config.min_1se,
    )
    w = np.asarray(fit["w"], dtype=float)
    counterfactual = Y0 @ w

    # cross-fitted, bias-corrected rate ratio (the CSCM estimand)
    cr = crossfit_rate_ratio(
        y, Y0, T0, T1, config.K, config.ci_level, v_method=config.v_method,
        n_lambda=config.n_lambda, lambda_min_ratio=config.lambda_min_ratio,
        min_1se=config.min_1se,
    )

    donor_weights = {donor_names[i]: float(w[i]) for i in range(len(donor_names))}
    scm_weights = {str(donor_names[i]): float(fit["w_scm"][i])
                   for i in range(len(donor_names))}
    weights = make_weights_results(
        donor_weights,
        constraint="non-negative, adding-up relaxed (CSCM)",
        extra={"lambda": float(fit["lambda"])},
    )
    inference = InferenceResults(
        ci_lower=cr["rr_lower"],
        ci_upper=cr["rr_upper"],
        standard_error=cr["log_rr_se"],
        confidence_level=config.ci_level,
        method="crossfit_rate_ratio",
        details={
            "scale": "rate_ratio",
            "log_rr": cr["log_rr"],
            "tau_k": np.asarray(cr["tau_k"], dtype=float),
            "K": cr["K"],
            "r": cr["r"],
        },
    )

    submodels = build_effect_submodels(
        observed_outcome=y,
        counterfactual_outcome=counterfactual,
        n_pre_periods=T0,
        n_post_periods=T1,
        time_periods=None if time_labels is None else np.asarray(time_labels),
        weights=weights,
        inference=inference,
        method_name="CSCM",
        effects_overrides={"att_percent": 100.0 * (cr["rate_ratio"] - 1.0)},
        additional_effects={
            "rate_ratio": cr["rate_ratio"],
            "log_rate_ratio": cr["log_rr"],
            "rate_ratio_ci": (cr["rr_lower"], cr["rr_upper"]),
        },
    )

    method_details = MethodDetailsResults(
        method_name="CSCM",
        parameters_used={
            "K": config.K,
            "v_method": config.v_method,
            "lambda": float(fit["lambda"]),
            "n_lambda": config.n_lambda,
            "min_1se": config.min_1se,
        },
    )

    submodels.pop("method_details", None)   # replace with the richer CSCM one
    resolved = config.resolved_plot()
    results = BaseEstimatorResults(
        **submodels,
        method_details=method_details,
        additional_outputs={"scm_weights": scm_weights},
        plot_config=resolved,
    )

    if resolved.display:
        plot_cscm(config, y, counterfactual, time_labels, T0, resolved)

    return results
