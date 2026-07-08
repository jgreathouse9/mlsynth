"""Assemble the SMC point estimate from pivoted inputs (``run_smc``)."""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError
from .estimation import counterfactual, smc_weights
from .structures import SMCFit, SMCInputs


def _build_matching_matrix(inputs: SMCInputs):
    """Stack the pre-outcome rows with standardized covariate rows (Algorithm 3).

    Pure Algorithm 1 (no covariates) returns the ``(T0, J)`` pre-outcome block.
    With covariates, each covariate's pre-mean row is scaled to the mean standard
    deviation of the outcome rows (the reference's equal-variance scaling) and
    stacked on top, so predictors and outcomes enter on a common scale.
    """
    T0 = inputs.T0
    y_out = inputs.Y_treated[:T0]                     # (T0,)
    X_out = inputs.Y_donors[:T0, :]                   # (T0, J)
    if not inputs.has_covariates:
        return X_out, y_out

    out_sd = X_out.std(axis=1, ddof=0)               # per pre-period row sd
    scale = float(np.mean(out_sd[out_sd > 0])) if np.any(out_sd > 0) else 1.0
    cov_all = np.column_stack([inputs.cov_treated, inputs.cov_donors])  # (P, J+1)
    cov_sd = cov_all.std(axis=1, ddof=0)
    cov_sd = np.where(cov_sd > 0, cov_sd, 1.0)
    y_cov = inputs.cov_treated / cov_sd * scale       # (P,)
    X_cov = inputs.cov_donors / cov_sd[:, None] * scale
    X = np.vstack([X_cov, X_out])                     # (P + T0, J)
    y = np.concatenate([y_cov, y_out])
    return X, y


def run_smc(inputs: SMCInputs, *, ridge: float = 1e-3) -> SMCFit:
    """Fit SMC (deterministic Algorithm 1 / 3) and return the point estimate."""
    try:
        X, y = _build_matching_matrix(inputs)
        weights = smc_weights(X, y, ridge=ridge)
        cf = counterfactual(inputs.Y_donors, weights)
    except Exception as exc:  # pragma: no cover - defensive; smc_weights is total
        raise MlsynthEstimationError(f"SMC estimation failed: {exc}") from exc

    obs = inputs.Y_treated
    gap = obs - cf
    T0 = inputs.T0
    pre_rmse = float(np.sqrt(np.mean(gap[:T0] ** 2)))
    att = float(np.mean(gap[T0:])) if T0 < inputs.T else float("nan")
    donor_weights = {
        lbl: float(c) for lbl, c in zip(inputs.donor_labels, weights.combined)
    }
    return SMCFit(
        att=att,
        weights=weights.combined,
        theta=weights.theta,
        w=weights.w,
        bias=weights.bias,
        sigma2=weights.sigma2,
        counterfactual=cf,
        gap=gap,
        pre_rmse=pre_rmse,
        n_matching_rows=int(X.shape[0]),
        n_covariates=len(inputs.covariate_names),
        donor_weights=donor_weights,
    )
