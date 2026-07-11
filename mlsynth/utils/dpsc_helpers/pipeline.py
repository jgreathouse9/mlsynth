"""Orchestration for DPSC: privatized release + privacy-noise quantification."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ...config_models import (
    BaseEstimatorResults,
    InferenceResults,
    MethodDetailsResults,
)
from ..results_helpers import build_effect_submodels, make_weights_results
from .mechanisms import (
    non_private_counterfactual,
    run_objective_perturbation,
    run_output_perturbation,
)
from .setup import prepare_dpsc_inputs


def _draw(config, inputs, rng):
    """One privatized counterfactual path + weights for the configured mechanism."""
    pre_donor = inputs.donor_matrix[: inputs.T0]
    pre_target = inputs.y_treated[: inputs.T0]
    donor_full = inputs.donor_matrix
    if config.mechanism == "output":
        return run_output_perturbation(
            rng, pre_donor, pre_target, donor_full,
            config.ridge_lambda, config.epsilon1, config.epsilon2)
    return run_objective_perturbation(
        rng, pre_donor, pre_target, donor_full,
        config.ridge_lambda, config.epsilon1, config.epsilon2, config.delta)


def run_dpsc(config) -> BaseEstimatorResults:
    """Fit DPSC and return :class:`BaseEstimatorResults`.

    The reported counterfactual is a single seeded differentially private
    release. The privacy noise -- the dominant source of uncertainty -- is
    quantified by the Monte Carlo standard deviation of the ATT over
    ``n_draws`` independent privatized draws, and reported as the standard error
    and interval.
    """
    inputs = prepare_dpsc_inputs(config)
    y = inputs.y_treated
    T = y.shape[0]
    T0 = inputs.T0
    post = slice(T0, T)
    n_post = T - T0

    # The released (seeded) private counterfactual.
    rng = np.random.RandomState(config.seed)
    counterfactual, weights, info = _draw(config, inputs, rng)
    att = float(np.mean((y - counterfactual)[post]))

    # Non-private ridge reference (the epsilon -> infinity target).
    cf_np, _ = non_private_counterfactual(
        inputs.donor_matrix[:T0], y[:T0], inputs.donor_matrix, config.ridge_lambda)
    att_non_private = float(np.mean((y - cf_np)[post]))

    # Privacy-noise Monte Carlo: the spread of the private ATT across draws.
    mc = np.empty(config.n_draws)
    mc_rng = np.random.RandomState(config.seed + 1)
    for i in range(config.n_draws):
        cf_i, _, _ = _draw(config, inputs, mc_rng)
        mc[i] = np.mean((y - cf_i)[post])
    att_se = float(np.std(mc, ddof=1)) if config.n_draws > 1 else float("nan")
    z = norm.ppf(1.0 - config.alpha / 2.0)
    att_lo = att - z * att_se if np.isfinite(att_se) else float("nan")
    att_hi = att + z * att_se if np.isfinite(att_se) else float("nan")

    donor_weights = {d: float(w) for d, w in zip(inputs.donor_names, weights)}
    weights_res = make_weights_results(
        donor_weights,
        constraint="ridge regression coefficients (differentially private; "
                   "unconstrained, may be negative)")

    inference = InferenceResults(
        standard_error=att_se if np.isfinite(att_se) else None,
        ci_lower=att_lo if np.isfinite(att_lo) else None,
        ci_upper=att_hi if np.isfinite(att_hi) else None,
        confidence_level=1.0 - config.alpha,
        method=f"differential-privacy noise ({config.mechanism} perturbation, "
               f"Monte Carlo over {config.n_draws} draws)",
        details={
            "att_private": att,
            "att_non_private": att_non_private,
            "att_mc_mean": float(np.mean(mc)),
            "mechanism": config.mechanism,
            "epsilon1": config.epsilon1, "epsilon2": config.epsilon2,
            "epsilon_total": config.epsilon1 + config.epsilon2,
            "delta": config.delta, "ridge_lambda": config.ridge_lambda,
            **{k: float(v) for k, v in info.items()},
        },
    )

    submodels = build_effect_submodels(
        observed_outcome=y,
        counterfactual_outcome=np.asarray(counterfactual, dtype=float),
        n_pre_periods=T0,
        n_post_periods=n_post,
        time_periods=inputs.time_labels,
        weights=weights_res,
        inference=inference,
        att_std_err=att_se if np.isfinite(att_se) else None,
        effects_overrides={"att": att},
        additional_effects={"att_non_private": att_non_private},
        intervention_time=(inputs.time_labels[T0] if T0 < T else inputs.time_labels[-1]),
    )

    return BaseEstimatorResults(
        **submodels,
        method_details=MethodDetailsResults(
            method_name="DPSC",
            is_recommended=False,
            parameters_used={
                "mechanism": config.mechanism,
                "epsilon1": config.epsilon1, "epsilon2": config.epsilon2,
                "delta": config.delta, "ridge_lambda": config.ridge_lambda,
                "n_draws": config.n_draws, "seed": config.seed,
            },
        ),
        additional_outputs={
            "treated_name": inputs.treated_name,
            "pre_periods": T0,
            "att_non_private": att_non_private,
            "noise_scales": {k: float(v) for k, v in info.items()},
        },
    )
