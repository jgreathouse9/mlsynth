"""Counterfactual paths, credible bands, and ATT inference for BVS-SS.

Given posterior samples ``\\mu^{(t)}`` from the Gibbs sampler, the
counterfactual treated outcome at every period is

    Y_hat^{(0)}_t  =  X_demean_t \\mu^{(t)}  +  mean_Y,

where ``mean_Y`` is the pre-treatment mean of the treated outcome (the
sampler operates on demeaned data; we add the mean back to recover the
original outcome scale).

The post-treatment ATT for each draw is

    ATT^{(t)}  =  (1 / T_post) \\sum_{t > T_0}
                       (Y^{(1)}_t  -  Y_hat^{(0), (t)}_t)
                =  (1 / T_post) \\sum_{t > T_0}
                       ((Y^{(1)}_t - mean_Y)  -  X_demean_t \\mu^{(t)}).
"""

from __future__ import annotations

import numpy as np

from .structures import BVSSInference, BVSSInputs


def compute_inference(
    inputs: BVSSInputs,
    mu_samples: np.ndarray,
    ci_alpha: float = 0.05,
) -> BVSSInference:
    """Assemble :class:`BVSSInference` from posterior ``\\mu`` samples.

    Parameters
    ----------
    inputs : BVSSInputs
        Demeaned panel inputs.
    mu_samples : np.ndarray
        Shape ``(N, n_samples)`` posterior samples of ``\\mu`` after
        burn-in.
    ci_alpha : float
        Two-sided significance level. Credible bands are at the
        ``ci_alpha / 2`` and ``1 - ci_alpha / 2`` percentiles.
    """

    if mu_samples.ndim != 2 or mu_samples.shape[0] != inputs.N:
        raise ValueError(
            f"mu_samples must have shape (N, n_samples) with N={inputs.N}; "
            f"got {mu_samples.shape}."
        )

    lower_pct = 100 * (ci_alpha / 2)
    upper_pct = 100 * (1 - ci_alpha / 2)

    # In-sample (pre-treatment) counterfactuals in demeaned coords.
    Y_cf_pre = inputs.X_pre_demean @ mu_samples       # (T0, n_samples)
    Y_cf_pre_mean = Y_cf_pre.mean(axis=1) + inputs.mean_Y
    pre_lower = np.percentile(Y_cf_pre, lower_pct, axis=1) + inputs.mean_Y
    pre_upper = np.percentile(Y_cf_pre, upper_pct, axis=1) + inputs.mean_Y

    if inputs.X_post_demean is not None:
        Y_cf_post = inputs.X_post_demean @ mu_samples  # (T_post, n_samples)
        Y_cf_post_mean = Y_cf_post.mean(axis=1) + inputs.mean_Y
        post_lower = np.percentile(Y_cf_post, lower_pct, axis=1) + inputs.mean_Y
        post_upper = np.percentile(Y_cf_post, upper_pct, axis=1) + inputs.mean_Y

        Y_obs_post_demean = inputs.y_target[inputs.T0:] - inputs.mean_Y
        # ATT^(t) = mean over post periods of (Y_obs - X_demean @ mu^(t))
        att_samples = (Y_obs_post_demean[:, None] - Y_cf_post).mean(axis=0)
        att_mean = float(att_samples.mean())
        att_lower = float(np.percentile(att_samples, lower_pct))
        att_upper = float(np.percentile(att_samples, upper_pct))

        cf_mean = np.concatenate([Y_cf_pre_mean, Y_cf_post_mean])
        cf_lower = np.concatenate([pre_lower, post_lower])
        cf_upper = np.concatenate([pre_upper, post_upper])
    else:
        cf_mean = Y_cf_pre_mean
        cf_lower = pre_lower
        cf_upper = pre_upper
        att_samples = np.array([])
        att_mean = float("nan")
        att_lower = float("nan")
        att_upper = float("nan")

    return BVSSInference(
        att_mean=att_mean,
        att_ci_lower=att_lower,
        att_ci_upper=att_upper,
        att_samples=att_samples,
        ci_alpha=ci_alpha,
        counterfactual_mean=cf_mean,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
    )
