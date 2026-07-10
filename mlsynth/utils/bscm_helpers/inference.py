"""Counterfactual paths, credible bands, and ATT inference for BSCM.

Given posterior samples of the intercept ``beta_0^{(t)}`` and donor weights
``beta^{(t)}``, the counterfactual treated outcome at every period is::

    Y_hat^{(0),(t)}_t  =  beta_0^{(t)}  +  X_t beta^{(t)},

matching the reference Stan model. The post-treatment ATT for each draw is the
mean gap ``Y^{(1)}_t - Y_hat^{(0),(t)}_t`` over ``t > T_0``.
"""

from __future__ import annotations

import numpy as np

from .structures import BSCMInference, BSCMInputs


def compute_inference(
    inputs: BSCMInputs,
    beta0_samples: np.ndarray,
    beta_samples: np.ndarray,
    ci_alpha: float = 0.05,
) -> BSCMInference:
    """Assemble :class:`BSCMInference` from posterior coefficient samples.

    Parameters
    ----------
    inputs : BSCMInputs
        Prepared panel inputs.
    beta0_samples : np.ndarray
        Length-``n_samples`` posterior intercept draws.
    beta_samples : np.ndarray
        Shape ``(N, n_samples)`` posterior donor-weight draws.
    ci_alpha : float
        Two-sided significance level; bands at ``ci_alpha / 2`` and
        ``1 - ci_alpha / 2`` percentiles.
    """

    if beta_samples.ndim != 2 or beta_samples.shape[0] != inputs.N:
        raise ValueError(
            f"beta_samples must have shape (N, n_samples) with N={inputs.N}; "
            f"got {beta_samples.shape}."
        )

    lower_pct = 100 * (ci_alpha / 2)
    upper_pct = 100 * (1 - ci_alpha / 2)

    # Counterfactual draws over all periods: (T, n_samples).
    cf = inputs.X_all @ beta_samples + beta0_samples[None, :]
    cf_mean = cf.mean(axis=1)
    cf_lower = np.percentile(cf, lower_pct, axis=1)
    cf_upper = np.percentile(cf, upper_pct, axis=1)

    if inputs.T0 < inputs.T:
        y_post = inputs.y_target[inputs.T0:]
        att_samples = (y_post[:, None] - cf[inputs.T0:]).mean(axis=0)
        att_mean = float(att_samples.mean())
        att_lower = float(np.percentile(att_samples, lower_pct))
        att_upper = float(np.percentile(att_samples, upper_pct))
    else:
        att_samples = np.array([])
        att_mean = att_lower = att_upper = float("nan")

    return BSCMInference(
        att_mean=att_mean,
        att_ci_lower=att_lower,
        att_ci_upper=att_upper,
        att_samples=att_samples,
        ci_alpha=ci_alpha,
        counterfactual_mean=cf_mean,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
    )
