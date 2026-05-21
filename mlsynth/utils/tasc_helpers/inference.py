"""Counterfactual inference for TASC.

Given the smoothed states from the full forward / backward pass (Algorithm 3,
lines 1-3 of the post-EM block), the counterfactual is

    y_hat_{0, t} = h_1^T m_t^s

and the posterior variance of the target observation is

    Var(y_{0, t} | y_{1:T_donors})  =  h_1^T P_t^s h_1  +  R_{1, 1}

(adding back the observation noise restores the variance of an *observation*
rather than the latent target).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .structures import (
    TASCInference,
    TASCParameters,
    TASCSmoothedStates,
)


def counterfactual_with_ci(
    smoothed: TASCSmoothedStates,
    params: TASCParameters,
    alpha: float,
) -> TASCInference:
    """Compute the TASC counterfactual path and posterior CIs.

    Parameters
    ----------
    smoothed : TASCSmoothedStates
        Smoothed states from the full pass over all ``T`` periods. Index 0
        holds the smoothed prior; indices 1..T hold the smoothed posteriors
        for each observation timestep.
    params : TASCParameters
        Final EM-learned parameters. ``H[0]`` is ``h_1`` and ``R[0, 0]`` is
        the target's observation-noise variance ``R_{1, 1}``.
    alpha : float
        Significance level. The bands are ``y_hat +/- z_{1 - alpha/2} * sd``.
    """

    h1 = params.H[0]
    r11 = float(params.R[0, 0])

    m_s = smoothed.m_s[1:]  # drop the prior, shape (T, d)
    P_s = smoothed.P_s[1:]  # shape (T, d, d)

    counterfactual = m_s @ h1

    # h_1^T P_t^s h_1, computed in one shot.
    latent_var = np.einsum("i,tij,j->t", h1, P_s, h1)
    posterior_var = latent_var + r11
    posterior_var = np.maximum(posterior_var, 0.0)
    sd = np.sqrt(posterior_var)

    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_lower = counterfactual - z * sd
    ci_upper = counterfactual + z * sd

    return TASCInference(
        counterfactual=counterfactual,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        posterior_variance=posterior_var,
        alpha=alpha,
    )
