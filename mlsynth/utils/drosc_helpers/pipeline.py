"""DROSC orchestration: inputs -> point estimand -> optional union-CI inference
-> standardized :class:`BaseEstimatorResults`."""
from __future__ import annotations

import warnings

import numpy as np

from ...config_models import BaseEstimatorResults, InferenceResults
from ..results_helpers import build_effect_submodels, make_weights_results
from .estimation import drosc_point, sc_weights
from .inference import drosc_union_ci
from .setup import prepare_drosc_inputs
from .structures import DROSCInputs


def run_drosc(config) -> BaseEstimatorResults:
    """Fit DROSC and return standardized results."""
    inp: DROSCInputs = prepare_drosc_inputs(config)
    y = inp.y
    T = len(y)
    pre = inp.pre_periods
    lam = float(config.robustness_lambda)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau, beta = drosc_point(
            inp.Y0, inp.Y1, inp.X0, inp.X1, robustness_lambda=lam,
            nu=config.nu, eta=config.eta, slack_rate=config.slack_rate,
        )
        sc_w, sc_resid = sc_weights(inp.Y0, inp.X0)

    if beta is None or not np.isfinite(tau):
        raise RuntimeError(
            "DROSC moment-band optimisation did not return a feasible solution."
        )

    counterfactual = inp.donor_matrix @ beta                 # X b over all periods
    classical_att = float((inp.Y1 - inp.X1 @ sc_w).mean())

    donor_weights = {str(n): float(w) for n, w in zip(inp.donor_names, beta)}
    weights = make_weights_results(
        donor_weights,
        constraint="simplex (non-negative, sum to 1) intersect pre-treatment moment band",
        extra={"robustness_lambda": lam},
    )

    inference = None
    if config.inference:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            union = drosc_union_ci(
                inp.Y0, inp.Y1, inp.X0, inp.X1, robustness_lambda=lam,
                alpha=config.alpha, alpha0=config.alpha0, c_sample=config.c_sample,
                n_perturbations=config.n_perturbations, gamma=config.gamma,
                conditional=config.conditional, dependent=config.dependent,
                seed=config.seed,
            )
        if union:
            lows = [iv[0] for iv in union]
            highs = [iv[1] for iv in union]
            inference = InferenceResults(
                ci_lower=float(min(lows)),        # enveloping hull of the union
                ci_upper=float(max(highs)),
                confidence_level=1.0 - config.alpha,
                method="drosc-perturbation-union (Koo-Guo 2026)",
                details={
                    "ci_intervals": union,        # the (possibly disjoint) union
                    "n_intervals": len(union),
                    "conditional": bool(config.conditional),
                    "dependent": bool(config.dependent),
                    "n_perturbations": int(config.n_perturbations),
                },
            )

    submodels = build_effect_submodels(
        observed_outcome=y,
        counterfactual_outcome=counterfactual,
        n_pre_periods=pre,
        n_post_periods=int(T - pre),
        time_periods=inp.time_labels,
        weights=weights,
        inference=inference,
        method_name="DROSC",
        additional_effects={"classical_sc_att": classical_att},
        additional_metrics={"robustness_lambda": lam},
        intervention_time=(inp.time_labels[pre] if pre < T else None),
    )
    return BaseEstimatorResults(
        **submodels,
        additional_outputs={
            "donor_names": list(inp.donor_names),
            "treated_name": inp.treated_name,
            "robustness_lambda": lam,
            "classical_sc_att": classical_att,
            "beta": np.asarray(beta, float),
            "pre_periods": pre,
        },
    )
