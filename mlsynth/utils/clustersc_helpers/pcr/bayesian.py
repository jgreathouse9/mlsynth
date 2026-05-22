"""Bayesian posterior weight solver for PCR-SC (Bayani 2022, Ch. 1).

Wraps :func:`mlsynth.utils.bayesutils.BayesSCM` to produce a Gaussian
posterior over the weight vector :math:`f` given the HSVT-denoised
pre-period donor matrix and the treated unit. Draws ``n_samples`` from
that posterior, propagates each through the *denoised* donor matrix in
both pre and post periods (Algorithm 4 Step 5 of Rho et al. 2025), and
returns

* the posterior mean weights,
* a posterior median counterfactual,
* per-period :math:`(1-\\alpha)` credible bands.

The Bayesian variant is an mlsynth extension on top of the paper's
frequentist Algorithm 2; the denoising and projection steps are
identical, only the inner OLS is replaced with a Gaussian posterior.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError
from ...bayesutils import BayesSCM


def solve_bayesian(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
    denoised_donor_full: np.ndarray,
    *,
    alpha: float = 0.05,
    n_samples: int = 1000,
    alpha_prior: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return weights, median counterfactual, and per-period credible bands.

    Parameters
    ----------
    denoised_donor_pre : np.ndarray
        HSVT-denoised pre-period donor matrix, shape ``(T0, J)``.
    target_pre : np.ndarray
        Pre-period treated outcomes, shape ``(T0,)``.
    denoised_donor_full : np.ndarray
        HSVT-denoised donor matrix across all periods, shape ``(T, J)``.
        Used to project posterior draws into the counterfactual band.
    alpha : float
        Nominal level: returned band is the central ``1 - alpha``
        posterior interval.
    n_samples : int
        Number of posterior draws for the predictive band.
    alpha_prior : float
        Prior precision for the Gaussian weights prior (passed to
        :func:`BayesSCM`).
    rng : np.random.Generator, optional
        Source of randomness. A fresh ``default_rng()`` is used otherwise.

    Returns
    -------
    f_hat : np.ndarray
        Posterior mean weights, shape ``(J,)``.
    cf_median : np.ndarray
        Posterior median of the counterfactual, shape ``(T,)``.
    cf_lower : np.ndarray
        Lower per-period bound at level ``alpha/2``, shape ``(T,)``.
    cf_upper : np.ndarray
        Upper per-period bound at level ``1 - alpha/2``, shape ``(T,)``.
    """
    if not (0.0 < alpha < 1.0):
        raise MlsynthEstimationError("alpha must lie in (0, 1).")

    rng = rng or np.random.default_rng()

    noise_var = float(np.var(target_pre, ddof=1)) if target_pre.size > 1 else 1.0
    f_hat, f_cov, _, _ = BayesSCM(
        denoised_donor_matrix=denoised_donor_pre,
        target_outcome_pre_intervention=target_pre,
        observation_noise_variance=noise_var,
        weights_prior_precision=alpha_prior,
    )

    samples = rng.multivariate_normal(mean=f_hat, cov=f_cov, size=n_samples)
    # cf draws: project each posterior weight sample through the *denoised*
    # donor matrix in both pre and post (Algorithm 4 Step 5).
    cf_draws = denoised_donor_full @ samples.T  # (T, n_samples)

    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    cf_lower = np.percentile(cf_draws, lo_q, axis=1)
    cf_upper = np.percentile(cf_draws, hi_q, axis=1)
    cf_median = np.median(cf_draws, axis=1)

    return f_hat, cf_median, cf_lower, cf_upper
