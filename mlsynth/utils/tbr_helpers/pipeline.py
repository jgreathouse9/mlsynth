"""TBR orchestration: fit, build the posteriors, assemble the result."""

from __future__ import annotations

import numpy as np

from ...config_models import InferenceResults, WeightsResults
from ...exceptions import MlsynthDataError
from ..results_helpers import build_effect_submodels
from .posterior import (
    cumulative_posterior,
    fit_pretest,
    interval,
    iroas_fixed_cost,
    iroas_simulated,
)
from .setup import build_inputs
from .structures import CumulativeEffect, IROASResult, TBRFit, TBRResults

FIXED_COST_TOL = 1e-10


def _as_fit(fit) -> TBRFit:
    return TBRFit(alpha=fit.alpha, beta=fit.beta, sigma_sq=fit.sigma_sq,
                  df=fit.df, n_pretest=fit.n_pretest,
                  rank_deficient=fit.rank_deficient)


def _as_cumulative(loc, scale, df, level, periods) -> CumulativeEffect:
    lo, hi = interval(loc, scale, df, level)
    return CumulativeEffect(
        estimate=[float(v) for v in loc], scale=[float(v) for v in scale],
        lower=[float(v) for v in lo], upper=[float(v) for v in hi],
        level=float(level), df=int(df), periods=list(periods))


def run(config) -> TBRResults:
    inputs = build_inputs(config)
    T0, level = inputs.n_pre, float(config.level)
    y, x = inputs.y, inputs.x

    fit = fit_pretest(y[:T0], x[:T0])
    loc, scale = cumulative_posterior(fit, y[T0:], x[T0:])
    post_labels = inputs.time_labels[T0:]
    cumulative = _as_cumulative(loc, scale, fit.df, level, post_labels)

    counterfactual = np.concatenate([y[:T0], fit.predict(x[T0:])])

    cost_fit = cumulative_cost = iroas = None
    if config.cost_col:
        cfit = fit_pretest(inputs.cost_y[:T0], inputs.cost_x[:T0])
        closs, cscale = cumulative_posterior(cfit, inputs.cost_y[T0:],
                                             inputs.cost_x[T0:])
        cost_fit = _as_fit(cfit)
        cumulative_cost = _as_cumulative(closs, cscale, cfit.df, level,
                                         post_labels)
        total_cost = float(closs[-1])
        if abs(total_cost) <= FIXED_COST_TOL:
            raise MlsynthDataError(
                f"the cumulative incremental cost is {total_cost:.3g}, so iROAS "
                f"is a ratio with a zero denominator; drop cost_col or supply a "
                f"panel where the intervention carries spend")
        # Section 3.4: no spend outside the treated test cells leaves the cost
        # counterfactual zero with certainty, and the ratio a rescaled t.
        fixed = bool(abs(float(np.sum(inputs.cost_x))) <= FIXED_COST_TOL
                     and abs(float(np.sum(inputs.cost_y[:T0]))) <= FIXED_COST_TOL)
        if fixed:
            point, lo, hi = iroas_fixed_cost(loc, scale, fit.df, level,
                                             total_cost)
            point, lo, hi = float(point[-1]), float(lo[-1]), float(hi[-1])
        else:
            point, lo, hi = iroas_simulated(loc, scale, closs, cscale, fit.df,
                                            level)
        iroas = IROASResult(
            estimate=point, lower=lo, upper=hi, level=level,
            total_incremental_cost=total_cost,
            total_incremental_response=float(loc[-1]), fixed_cost=fixed)

    weights = WeightsResults(
        donor_weights={},
        summary_stats={"method": "TBR",
                       "note": "two aggregated group series; no donor weights",
                       "n_control_units": len(inputs.control_units),
                       "n_treated_units": len(inputs.treated_units)})
    std_inference = InferenceResults(
        method="bayesian_posterior",
        ci_lower=float(cumulative.lower[-1]) / inputs.n_test,
        ci_upper=float(cumulative.upper[-1]) / inputs.n_test,
        confidence_level=level,
        details={"estimand": "cumulative effect Delta(T), eqn 4",
                 "posterior": f"shifted, scaled t on {fit.df} df",
                 "cumulative_lower": float(cumulative.lower[-1]),
                 "cumulative_upper": float(cumulative.upper[-1])},
    )
    att = float(loc[-1]) / inputs.n_test          # mean per-period effect
    submodels = build_effect_submodels(
        observed_outcome=y,
        counterfactual_outcome=counterfactual,
        n_pre_periods=T0,
        n_post_periods=inputs.n_test,
        time_periods=np.asarray(inputs.time_labels),
        weights=weights,
        inference=std_inference,
        method_name="TBR",
        effects_overrides={"att": att},
        additional_effects={"cumulative_effect": float(loc[-1])},
        intervention_time=inputs.intervention_time,
    )
    n_cool = inputs.n_cooldown
    n_int = inputs.n_test - n_cool
    return TBRResults(
        **submodels,
        tbr_fit=_as_fit(fit),
        cost_fit=cost_fit,
        cumulative=cumulative,
        cumulative_cost=cumulative_cost,
        iroas=iroas,
        intervention_periods=n_int,
        cooldown_periods=n_cool,
        effect_at_intervention_end=float(loc[n_int - 1]),
        effect_at_cooldown_end=float(loc[-1]),
        treated_units=list(inputs.treated_units),
        control_units=list(inputs.control_units),
        unassigned_units=list(inputs.unassigned_units),
        filled_cells=inputs.filled_cells,
    )
