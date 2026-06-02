"""Bayesian Dirichlet simplex synthetic control (O'Riordan & Gilligan-Lee 2025, p.12).

The paper's SC model is the Bayesian simplex regression

.. math::

   y^t \\sim \\mathcal N\\Bigl(\\textstyle\\sum_i \\beta_i x_i^t,\\ \\sigma_y\\Bigr),
   \\quad \\beta_i \\ge 0,\\ \\textstyle\\sum_i \\beta_i = 1,
   \\quad \\beta \\sim \\mathrm{Dirichlet}(0.4),
   \\quad \\sigma_y \\sim \\mathcal N^+(0, 1),

with the target and donors standardised to zero mean and unit standard
deviation over the pre-intervention window (which absorbs the intercept
:math:`\\alpha` of equation (4)). The :math:`\\mathrm{Dirichlet}(0.4)` prior
(concentration ``< 1``) regularises the weights toward sparse corners of the
simplex. 95% credible intervals come from the 2.5 / 97.5 percentiles of the
posterior predictive distribution.

The posterior has no closed form (a Dirichlet prior is not conjugate to a
Gaussian likelihood under the simplex constraint), so this module draws from it
with a self-contained Metropolis-Hastings sampler -- a Dirichlet random-walk
proposal on :math:`\\beta` in the native simplex space (no Jacobian) and a
log-random-walk on :math:`\\sigma_y` -- with proposal scales adapted during
warm-up to a healthy acceptance rate. No probabilistic-programming dependency
is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True)
class BayesianSCFit:
    """Posterior summary of the Dirichlet simplex SC."""

    weights: np.ndarray            # posterior-mean simplex weights, length n
    counterfactual: np.ndarray     # posterior-mean counterfactual, length T
    cf_lower: np.ndarray           # posterior-predictive lower band, length T
    cf_upper: np.ndarray           # posterior-predictive upper band, length T
    att: float                     # posterior-mean ATT (mean post-period gap)
    att_ci: Tuple[float, float]    # posterior-predictive ATT credible interval
    sigma: float                   # posterior-mean residual sd (standardised)
    accept_beta: float
    accept_sigma: float
    n_samples: int


def _log_dir(x: np.ndarray, conc: np.ndarray) -> float:
    return float(gammaln(conc.sum()) - gammaln(conc).sum()
                 + ((conc - 1.0) * np.log(x)).sum())


def bayesian_simplex_sc(
    y: np.ndarray,
    D: np.ndarray,
    T0: int,
    *,
    alpha: float = 0.4,
    sigma_prior_scale: float = 1.0,
    n_samples: int = 4000,
    n_warmup: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> BayesianSCFit:
    """Fit the Dirichlet(``alpha``) Bayesian simplex SC and summarise the posterior.

    Parameters
    ----------
    y : np.ndarray
        Treated-unit outcome, length ``T``.
    D : np.ndarray
        Donor matrix, shape ``(T, n)``.
    T0 : int
        Number of pre-intervention periods (weights are fit on the pre-window).
    alpha : float
        Dirichlet concentration (paper uses 0.4; ``< 1`` favours sparse weights).
    sigma_prior_scale : float
        Scale of the half-normal prior on the (standardised) residual sd.
    n_samples, n_warmup : int
        Post-warm-up draws and warm-up iterations.
    ci_level : float
        Credible-interval level (paper uses 0.95).
    seed : int

    Returns
    -------
    BayesianSCFit
    """
    rng = np.random.default_rng(seed)
    T, n = D.shape

    # Standardise target and donors over the pre-intervention window.
    mu_y = y[:T0].mean()
    sd_y = y[:T0].std() + 1e-12
    mu_d = D[:T0].mean(axis=0)
    sd_d = D[:T0].std(axis=0) + 1e-12
    ys = (y - mu_y) / sd_y
    Ds = (D - mu_d) / sd_d

    Xp = Ds[:T0]
    yp = ys[:T0]
    XtX = Xp.T @ Xp
    Xty = Xp.T @ yp
    yty = float(yp @ yp)
    conc = np.full(n, float(alpha))

    def loglik(beta: np.ndarray, sigma: float) -> float:
        rss = yty - 2.0 * float(beta @ Xty) + float(beta @ XtX @ beta)
        return -0.5 * T0 * np.log(2 * np.pi * sigma ** 2) - 0.5 * rss / sigma ** 2

    def log_half_normal(sigma: float) -> float:
        return (np.log(2.0) - 0.5 * np.log(2 * np.pi * sigma_prior_scale ** 2)
                - 0.5 * (sigma / sigma_prior_scale) ** 2)

    def logpost(beta: np.ndarray, sigma: float) -> float:
        return loglik(beta, sigma) + _log_dir(beta, conc) + log_half_normal(sigma)

    beta = np.full(n, 1.0 / n)
    sigma = max(0.1, float(np.std(yp - Xp @ beta)))
    lp = logpost(beta, sigma)

    kappa = max(50.0, 50.0 * n)        # Dirichlet RW concentration (larger = smaller step)
    sigma_step = 0.15
    total = n_warmup + n_samples
    draws_b = np.zeros((n_samples, n))
    draws_s = np.zeros(n_samples)
    acc_b = acc_s = 0
    win_b = win_s = 0                  # acceptance counters for adaptation window

    for it in range(total):
        # ---- beta: Dirichlet random-walk proposal in native simplex space ----
        prop = rng.dirichlet(np.maximum(kappa * beta, 1e-6))
        prop = np.clip(prop, 1e-12, None)
        prop /= prop.sum()
        lp_prop = logpost(prop, sigma)
        log_q = (_log_dir(beta, kappa * prop) - _log_dir(prop, kappa * beta))
        accepted_b = np.log(rng.uniform()) < (lp_prop - lp + log_q)
        if accepted_b:
            beta, lp = prop, lp_prop
            win_b += 1
            if it >= n_warmup:
                acc_b += 1

        # ---- sigma: log random walk (target carries the +log sigma Jacobian) ----
        s_prop = float(np.exp(np.log(sigma) + rng.normal(0, sigma_step)))
        lp_s_prop = logpost(beta, s_prop) + np.log(s_prop)
        lp_s_cur = logpost(beta, sigma) + np.log(sigma)
        accepted_s = np.log(rng.uniform()) < (lp_s_prop - lp_s_cur)
        if accepted_s:
            sigma = s_prop
            lp = logpost(beta, sigma)
            win_s += 1
            if it >= n_warmup:
                acc_s += 1

        # ---- adapt proposal scales during warm-up (target ~0.3 / ~0.4) ----
        if it < n_warmup and (it + 1) % 100 == 0:
            rb, rs = win_b / 100.0, win_s / 100.0
            kappa *= np.exp(0.5 * (0.3 - rb))      # lower accept -> larger kappa (smaller step)
            kappa = float(np.clip(kappa, 5.0, 1e7))
            sigma_step *= np.exp(0.5 * (rs - 0.4))
            sigma_step = float(np.clip(sigma_step, 1e-3, 2.0))
            win_b = win_s = 0

        if it >= n_warmup:
            draws_b[it - n_warmup] = beta
            draws_s[it - n_warmup] = sigma

    # Posterior-mean weights and counterfactual (on the original scale).
    w_mean = draws_b.mean(axis=0)
    cf_mean = (Ds @ w_mean) * sd_y + mu_y

    # Posterior-predictive draws of the counterfactual: X beta + N(0, sigma).
    mean_std = draws_b @ Ds.T                                   # (S, T) standardised
    noise = rng.normal(0.0, 1.0, size=mean_std.shape) * draws_s[:, None]
    cf_pred = (mean_std + noise) * sd_y + mu_y                  # (S, T) original scale
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    cf_lower = np.quantile(cf_pred, lo_q, axis=0)
    cf_upper = np.quantile(cf_pred, hi_q, axis=0)

    post = np.arange(T) >= T0
    gap_pred = (y[None, :] - cf_pred)[:, post]                  # (S, n_post)
    att_draws = gap_pred.mean(axis=1)
    att = float(np.mean((y - cf_mean)[post]))
    att_ci = (float(np.quantile(att_draws, lo_q)),
              float(np.quantile(att_draws, hi_q)))

    return BayesianSCFit(
        weights=w_mean, counterfactual=cf_mean, cf_lower=cf_lower,
        cf_upper=cf_upper, att=att, att_ci=att_ci, sigma=float(draws_s.mean()),
        accept_beta=acc_b / n_samples, accept_sigma=acc_s / n_samples,
        n_samples=n_samples,
    )
