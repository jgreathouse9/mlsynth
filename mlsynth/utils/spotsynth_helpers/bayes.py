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

The authors fit this model in Stan (Hamiltonian Monte Carlo / NUTS), which they
could not share. mlsynth fits the identical model with **NumPyro**'s NUTS -- the
same HMC family -- so it reproduces their estimation procedure as closely as an
open-source tool can. NumPyro is an optional dependency; if it is not installed,
use ``inference="frequentist"`` for the dependency-free simplex least-squares
point estimate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ...exceptions import MlsynthDataError, MlsynthEstimationError


@dataclass(frozen=True)
class BayesianSCFit:
    """Posterior summary of the Dirichlet simplex SC (NUTS)."""

    weights: np.ndarray            # posterior-mean simplex weights, length n
    counterfactual: np.ndarray     # posterior-mean counterfactual, length T
    cf_lower: np.ndarray           # posterior-predictive lower band, length T
    cf_upper: np.ndarray           # posterior-predictive upper band, length T
    att: float                     # posterior-mean ATT (mean post-period gap)
    att_ci: Tuple[float, float]    # posterior-predictive ATT credible interval
    sigma: float                   # posterior-mean residual sd (standardised)
    accept_prob: float             # mean NUTS acceptance probability
    n_samples: int


def bayesian_simplex_sc(
    y: np.ndarray,
    D: np.ndarray,
    T0: int,
    *,
    alpha: float = 0.4,
    sigma_prior_scale: float = 1.0,
    n_samples: int = 4000,
    n_warmup: int = 1000,
    n_chains: int = 2,
    ci_level: float = 0.95,
    seed: int = 0,
) -> BayesianSCFit:
    """Fit the Dirichlet(``alpha``) Bayesian simplex SC with NumPyro NUTS.

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
        Total post-warm-up draws (across chains) and warm-up iterations per chain.
    n_chains : int
        Number of NUTS chains.
    ci_level : float
        Credible-interval level (paper uses 0.95).
    seed : int

    Returns
    -------
    BayesianSCFit

    Raises
    ------
    MlsynthEstimationError
        If NumPyro / JAX are not installed.
    """
    # Validate inputs BEFORE requiring the optional NumPyro dependency, so the
    # data-shape / finiteness guards raise identically whether or not NumPyro is
    # installed (the sampler is never reached on bad input).
    y = np.asarray(y, dtype=float)
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise MlsynthDataError(
            f"Donor matrix D must be 2-D (T, n); got shape {D.shape}.")
    if y.ndim != 1 or y.shape[0] != D.shape[0]:
        raise MlsynthDataError(
            f"y (len {y.shape[0]}) and D (T={D.shape[0]}) must share the T axis.")
    if D.shape[1] < 1:
        raise MlsynthDataError("Bayesian SC needs at least one donor column.")
    if not (0 < T0 < D.shape[0]):
        raise MlsynthDataError(
            f"T0 must satisfy 0 < T0 < T (T={D.shape[0]}); got T0={T0}.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(D))):
        raise MlsynthDataError("y or D contains non-finite values.")

    try:
        import os
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MlsynthEstimationError(
            "inference='bayes' requires NumPyro (pip install numpyro). "
            "Use inference='frequentist' for the dependency-free simplex SC."
        ) from exc

    T, n = D.shape
    mu_y = y[:T0].mean()
    sd_y = y[:T0].std() + 1e-12
    mu_d = D[:T0].mean(axis=0)
    sd_d = D[:T0].std(axis=0) + 1e-12
    ys = (y - mu_y) / sd_y
    Ds = (D - mu_d) / sd_d

    def model(Xpre, ypre):
        beta = numpyro.sample("beta", dist.Dirichlet(alpha * jnp.ones(n)))
        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior_scale))
        numpyro.sample("y", dist.Normal(Xpre @ beta, sigma), obs=ypre)

    per_chain = max(250, int(n_samples) // max(1, n_chains))
    mcmc = MCMC(NUTS(model), num_warmup=int(n_warmup), num_samples=per_chain,
                num_chains=int(n_chains), progress_bar=False)
    mcmc.run(jax.random.PRNGKey(int(seed)), jnp.asarray(Ds[:T0]),
             jnp.asarray(ys[:T0]),
             extra_fields=("accept_prob", "diverging"))
    samples = mcmc.get_samples()
    draws_b = np.asarray(samples["beta"])
    draws_s = np.asarray(samples["sigma"])
    extra = mcmc.get_extra_fields()
    try:
        accept = float(np.mean(np.asarray(extra["accept_prob"])))
    except Exception:  # pragma: no cover
        accept = float("nan")
    n_divergent = int(np.sum(np.asarray(extra["diverging"]))) \
        if "diverging" in extra else 0
    if n_divergent > 0:
        warnings.warn(
            f"NUTS reported {n_divergent} divergent transition(s); the "
            "posterior may be biased. Increase n_warmup or reparameterise.",
            RuntimeWarning, stacklevel=2,
        )
    if np.isfinite(accept) and accept < 0.2:
        warnings.warn(
            f"NUTS mean acceptance probability is low ({accept:.2f}); the "
            "sampler may not have converged. Increase n_warmup/n_samples.",
            RuntimeWarning, stacklevel=2,
        )

    w_mean = draws_b.mean(axis=0)
    cf_mean = (Ds @ w_mean) * sd_y + mu_y

    rng = np.random.default_rng(seed)
    mean_std = draws_b @ Ds.T
    noise = rng.normal(0.0, 1.0, size=mean_std.shape) * draws_s[:, None]
    cf_pred = (mean_std + noise) * sd_y + mu_y
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    cf_lower = np.quantile(cf_pred, lo_q, axis=0)
    cf_upper = np.quantile(cf_pred, hi_q, axis=0)

    post = np.arange(T) >= T0
    att_draws = (y[None, :] - cf_pred)[:, post].mean(axis=1)
    att = float(np.mean((y - cf_mean)[post]))
    att_ci = (float(np.quantile(att_draws, lo_q)),
              float(np.quantile(att_draws, hi_q)))

    return BayesianSCFit(
        weights=w_mean, counterfactual=cf_mean, cf_lower=cf_lower,
        cf_upper=cf_upper, att=att, att_ci=att_ci, sigma=float(draws_s.mean()),
        accept_prob=accept, n_samples=int(draws_s.size),
    )
