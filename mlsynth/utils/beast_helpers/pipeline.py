"""BEAST driver: fit the immunized ATT path and assemble standardized results."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ...config_models import BaseEstimatorResults, InferenceResults, MethodDetailsResults
from ...exceptions import MlsynthEstimationError
from ..results_helpers import build_effect_submodels, make_weights_results
from .estimator import (
    balance_ok,
    balance_weights,
    calibration_lasso,
    immunized_att,
    orthogonality_reg,
)
from .setup import prepare_beast_inputs


def run_beast(config) -> BaseEstimatorResults:
    """Fit BEAST and return :class:`BaseEstimatorResults`."""
    ins = prepare_beast_inputs(config)
    X, d, Y = ins.X, ins.d, ins.Y
    pre, T = ins.pre, Y.shape[0]

    beta = calibration_lasso(d, X, c=config.c_cal)
    W = balance_weights(d, X, beta)
    if not balance_ok(W, tol=config.balance_tol):
        raise MlsynthEstimationError(
            f"BEAST: the covariate-balancing weights do not sum to one "
            f"(sum(W) = {float(np.sum(W)):.3f}); the calibration is degenerate. "
            "This is the over-saturated / high-dimensional regime BEAST is not "
            "built for -- reduce the covariate set or use a sparse, informative one.")

    tau = np.empty(T)
    se = np.empty(T)
    for t in range(T):
        yt = Y[t, :]
        mu = orthogonality_reg(yt, d, X, beta, c=config.c_ort) if config.immunity else None
        tau[t], se[t] = immunized_att(yt, d, X, beta, mu)

    # Treated-unit counterfactual path: observed minus the per-period effect.
    counterfactual = ins.y_treated - tau
    z = norm.ppf(1.0 - config.alpha / 2.0)

    # Post-period ATT and its interval (per-period errors treated as independent;
    # serially-correlated post residuals make this optimistic -- documented).
    post = slice(pre, T)
    att = float(np.mean(tau[post]))
    att_se = float(np.sqrt(np.mean(se[post] ** 2)) / np.sqrt(max(T - pre, 1)))
    att_lo, att_hi = att - z * att_se, att + z * att_se

    donor_w = {u: float(w) for u, w in zip(ins.unit_names, W) if u != ins.treated_name}
    weights = make_weights_results(
        donor_w, constraint="covariate balancing (exponential tilting), sum to 1")

    inference = InferenceResults(
        ci_lower=att_lo, ci_upper=att_hi, standard_error=att_se,
        confidence_level=1.0 - config.alpha,
        method="immunized doubly-robust ATT (Blehaut-D'Haultfoeuille-L'Hour-Tsybakov 2021)",
        details={
            "periods": list(ins.time_labels[post]),
            "tau": tau[post], "se": se[post],
            "pi_lower": tau[post] - z * se[post], "pi_upper": tau[post] + z * se[post],
            "att": att, "att_lower": att_lo, "att_upper": att_hi,
            "sum_weights": float(np.sum(W)),
            "n_selected": int(np.sum(np.abs(beta[1:]) > 1e-6)),
        },
    )
    submodels = build_effect_submodels(
        observed_outcome=ins.y_treated, counterfactual_outcome=counterfactual,
        n_pre_periods=pre, n_post_periods=int(T - pre),
        time_periods=ins.time_labels, weights=weights, inference=inference,
        effects_overrides={"att": att, "att_std_err": att_se},
        intervention_time=(ins.time_labels[pre] if pre < T else None),
    )
    return BaseEstimatorResults(
        **submodels,
        method_details=MethodDetailsResults(
            method_name="BEAST",
            parameters_used={
                "c_cal": config.c_cal, "c_ort": config.c_ort,
                "immunity": config.immunity,
                "n_covariates": len(ins.feature_names),
                "n_selected": int(np.sum(np.abs(beta[1:]) > 1e-6)),
                "sum_weights": float(np.sum(W)),
            },
        ),
        additional_outputs={
            "treated_name": ins.treated_name,
            "feature_names": list(ins.feature_names),
            "beta": beta,
            "effect_path": tau,
            "effect_se": se,
            "pre_periods": pre,
        },
    )
