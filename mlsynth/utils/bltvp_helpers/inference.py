"""Counterfactual paths, credible bands, and ATT inference for BLTVP.

The sampler already returns posterior predictive draws of the untreated
potential outcome over every period, so the work here is summarising them and
differencing against the observed series. The post-treatment ATT for each draw
is the mean gap over ``t >= T0``.

The bands are predictive, not just parametric: each draw carries observation
noise, which is what makes them comparable to the intervals the paper reports.
"""

from __future__ import annotations

import numpy as np

from .structures import BLTVPInference, BLTVPInputs


def compute_inference(
    inputs: BLTVPInputs,
    predictive: np.ndarray,
    ci_alpha: float = 0.05,
) -> BLTVPInference:
    """Assemble :class:`BLTVPInference` from posterior predictive draws.

    Parameters
    ----------
    inputs : BLTVPInputs
        Prepared panel inputs.
    predictive : np.ndarray
        Shape ``(T, n_samples)`` draws of the untreated potential outcome.
    ci_alpha : float
        Two-sided significance level; bands at the ``ci_alpha / 2`` and
        ``1 - ci_alpha / 2`` percentiles.
    """
    predictive = np.asarray(predictive, dtype=float)
    if predictive.ndim != 2 or predictive.shape[0] != inputs.T:
        raise ValueError(
            f"predictive must have shape (T, n_samples) with T={inputs.T}; "
            f"got {predictive.shape}."
        )

    lower_pct = 100.0 * (ci_alpha / 2.0)
    upper_pct = 100.0 * (1.0 - ci_alpha / 2.0)

    cf_mean = predictive.mean(axis=1)
    cf_lower = np.percentile(predictive, lower_pct, axis=1)
    cf_upper = np.percentile(predictive, upper_pct, axis=1)

    if inputs.T0 < inputs.T:
        y_post = inputs.y_target[inputs.T0:]
        att_samples = (y_post[:, None] - predictive[inputs.T0:]).mean(axis=0)
        att_mean = float(att_samples.mean())
        att_lower = float(np.percentile(att_samples, lower_pct))
        att_upper = float(np.percentile(att_samples, upper_pct))
    else:
        att_samples = np.array([])
        att_mean = att_lower = att_upper = float("nan")

    return BLTVPInference(
        att_mean=att_mean,
        att_ci_lower=att_lower,
        att_ci_upper=att_upper,
        att_samples=att_samples,
        ci_alpha=ci_alpha,
        counterfactual_mean=cf_mean,
        counterfactual_lower=cf_lower,
        counterfactual_upper=cf_upper,
    )
