"""Top-level orchestration for CMBSTS: ingest -> sample -> assemble result."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...config_models import InferenceResults, WeightsResults
from ..results_helpers import build_effect_submodels
from .engine import build_ssm, causal_effect, run_gibbs
from .setup import prepare_cmbsts_inputs
from .structures import CMBSTSInference, CMBSTSInputs, CMBSTSPosterior, CMBSTSResults


def _prior_scale(Y_pre: np.ndarray, scale: float, rho: float) -> np.ndarray:
    """Inverse-Wishart prior scale: ``scale`` * pre-period covariance with the
    off-diagonal set by the prior cross-series correlation ``rho``."""
    d = Y_pre.shape[1]
    var = Y_pre.var(axis=0, ddof=1)
    var = np.where(var <= 0, 1.0, var)                  # guard a constant series
    s0 = np.empty((d, d))
    for i in range(d):
        for j in range(d):
            s0[i, j] = var[i] if i == j else np.sqrt(var[i] * var[j]) * rho
    s0 = scale * 0.5 * (s0 + s0.T)
    return s0 + 1e-8 * np.eye(d)                         # keep strictly PD


def run_cmbsts(config: Any) -> CMBSTSResults:
    """Fit CMBSTS and return a standardized :class:`CMBSTSResults`."""
    inputs: CMBSTSInputs = prepare_cmbsts_inputs(config)
    Y, X, T0, T, d = inputs.Y, inputs.X, inputs.T0, inputs.T, inputs.d
    rng = np.random.default_rng(config.seed)

    s0 = _prior_scale(Y[:T0], float(config.prior_scale), float(config.prior_rho))
    nu0 = int(config.nu0) if config.nu0 is not None else d + 2
    burn = int(config.burn) if config.burn is not None else config.niter // 10
    Tm, Z, dist, M = build_ssm(d, list(config.components), config.seas_period, config.cycle_period)

    X_pre = X[:T0] if X is not None else None
    fit = run_gibbs(Y[:T0], Tm, Z, dist, M, s0, s0, nu0, int(config.niter), burn, rng, X_pre=X_pre)
    X_post = X[T0:] if X is not None else None
    eff = causal_effect(Y[T0:], fit, Tm, Z, rng, X_post=X_post,
                        excl_post=inputs.excl_post, ci_alpha=float(config.ci_alpha),
                        horizon=config.horizon)

    cf_full = np.vstack([fit["prefit_mean"], eff["cf_post_mean"]])     # (T, d)
    inference = CMBSTSInference(
        series_names=inputs.series_names,
        att_mean=eff["att_mean"], att_lower=eff["att_lower"], att_upper=eff["att_upper"],
        cum_mean=eff["cum_mean"], cum_lower=eff["cum_lower"], cum_upper=eff["cum_upper"],
        effect_path=eff["effect_path"], effect_lower=eff["effect_lower"],
        effect_upper=eff["effect_upper"], counterfactual_post=eff["cf_post_mean"],
        counterfactual_full=cf_full, att_samples=eff["att_samples"], ci_alpha=float(config.ci_alpha),
    )

    reg_names = list(inputs.covariate_names) + [str(c) for c in inputs.control_names]
    inclusion: Optional[Dict[str, float]] = None
    if "inclusion" in fit and reg_names:
        inclusion = {n: float(p) for n, p in zip(reg_names, fit["inclusion"])}
    posterior = CMBSTSPosterior(
        inclusion_probs=inclusion, n_draws=int(fit["n_kept"]), niter=int(config.niter),
        burn=burn, components=list(config.components),
        seas_period=config.seas_period, cycle_period=config.cycle_period,
    )

    # Standardized surface over the treated series (column 0).
    att0 = float(eff["att_mean"][0])
    std_inference = InferenceResults(
        method="bayesian_posterior",
        ci_lower=float(eff["att_lower"][0]), ci_upper=float(eff["att_upper"][0]),
        confidence_level=float(1.0 - config.ci_alpha),
        standard_error=float(np.std(eff["att_samples"][:, 0])),
        details=inference,
    )
    weights = WeightsResults(
        donor_weights={},
        summary_stats={"method": "CMBSTS",
                       "note": "multivariate state-space; no donor weights"},
    )
    submodels = build_effect_submodels(
        observed_outcome=Y[:, 0],
        counterfactual_outcome=cf_full[:, 0],
        n_pre_periods=T0,
        n_post_periods=T - T0,
        time_periods=np.asarray(inputs.time_labels),
        weights=weights,
        inference=std_inference,
        method_name="CMBSTS",
        att_std_err=float(np.std(eff["att_samples"][:, 0])),
        effects_overrides={"att": att0},
        intervention_time=inputs.intervention_time,
    )
    return CMBSTSResults(**submodels, inputs=inputs, posterior=posterior, inference_detail=inference)
