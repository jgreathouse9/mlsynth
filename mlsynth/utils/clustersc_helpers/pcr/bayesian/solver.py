"""Bayesian posterior weight solver for PCR-SC (Bayani 2022, Ch. 1).

Wraps :func:`.posterior.BayesSCM` to produce a Gaussian posterior over the
weight vector :math:`f` given the HSVT-denoised pre-period donor matrix and
the treated unit. Draws ``n_samples`` from that posterior, propagates each
through the *denoised* donor matrix in both pre and post periods (Algorithm 4
Step 5 of Rho et al. 2025), and returns

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

from mlsynth.exceptions import MlsynthEstimationError
from .posterior import BayesSCM


def _estimate_noise_var(denoised_donor_pre: np.ndarray, target_pre: np.ndarray) -> float:
    """Estimate sigma^2 from the OLS residual of the target on the denoised donors.

    The denoised donor block spans the rank-r signal subspace; the variance of
    the target left unexplained by it estimates the observation noise. Falls
    back to the total target variance only if the residual estimate is
    degenerate (e.g. an exact fit).
    """
    if target_pre.size <= 1:
        return 1.0
    beta, *_ = np.linalg.lstsq(denoised_donor_pre, target_pre, rcond=None)
    resid = target_pre - denoised_donor_pre @ beta
    dof = max(target_pre.size - np.linalg.matrix_rank(denoised_donor_pre), 1)
    noise_var = float(resid @ resid) / dof
    if not np.isfinite(noise_var) or noise_var <= 0:
        noise_var = float(np.var(target_pre, ddof=1))
    return noise_var if (np.isfinite(noise_var) and noise_var > 0) else 1.0


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

    # Observation-noise variance for the Gaussian likelihood. Estimate it from
    # the pre-period *fit residual* -- the variance of the treated series left
    # unexplained by the (rank-r) denoised donor span -- rather than the total
    # variance of the target. var(target_pre) conflates the signal with the
    # noise and, for strong-signal targets, massively over-inflates the
    # posterior (a looser counterfactual than the frequentist OLS path). The
    # residual after projecting the target onto the denoised donor column space
    # isolates the noise level sigma^2, matching the empirical-Bayes plug-in of
    # Amjad-Shah-Shen (2018, Sec. 5.3).
    noise_var = _estimate_noise_var(denoised_donor_pre, target_pre)
    f_hat, f_cov, _, _ = BayesSCM(
        denoised_donor_matrix=denoised_donor_pre,
        target_outcome_pre_intervention=target_pre,
        observation_noise_variance=noise_var,
        weights_prior_precision=alpha_prior,
    )

    # Point counterfactual: the posterior-MEAN projection through the denoised
    # donor matrix (exact for a Gaussian posterior), not the Monte-Carlo median
    # of draws. The median-of-draws is a noisy estimator dominated by the
    # prior-width weight directions in the donor null space; the mean projection
    # coincides with the frequentist OLS path when the prior is weak.
    counterfactual = denoised_donor_full @ f_hat

    # Credible band: percentiles of the projected posterior draws (Algorithm 4
    # Step 5). Projecting through the *denoised* (rank-r) donor matrix confines
    # the band to the signal subspace, so it is not inflated by raw donor noise.
    samples = rng.multivariate_normal(mean=f_hat, cov=f_cov, size=n_samples)
    cf_draws = denoised_donor_full @ samples.T  # (T, n_samples)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    cf_lower = np.percentile(cf_draws, lo_q, axis=1)
    cf_upper = np.percentile(cf_draws, hi_q, axis=1)

    return f_hat, counterfactual, cf_lower, cf_upper
