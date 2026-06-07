"""Configuration for the BVSS estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class BVSSConfig(BaseEstimatorConfig):
    """Configuration for the Bayesian Synthetic Control with Soft Simplex (BVS-SS).

    Implements:

        Xu, Y., & Zhou, Q. (2025). "Bayesian Synthetic Control with a
        Soft Simplex Constraint." arXiv:2503.06454.

    Parameters
    ----------
    n_iter : int
        Total Gibbs iterations (including burn-in).
    burn_in : int
        Number of warm-up iterations discarded before reporting.
    kappa1, kappa2 : float
        Gamma hyperparameters for the prior on the observation
        precision phi. Paper Section 5.1 uses kappa1=kappa2=1.
    theta : float
        Bernoulli prior inclusion probability per donor. Paper's
        empirical Section 6 uses theta in 0.2--0.25.
    tau_a, tau_b : float
        Gamma-prior shape (a1) and rate (a2) for the soft-constraint
        variance tau. Paper Section 5.1 uses (0.01, 0.1).
    n_tau : int
        Number of MH steps for tau per outer iteration.
    tau_min : float
        Numerical floor on tau; proposals below are reflected.
    ci_alpha : float
        Two-sided significance level for credible intervals (default
        0.05 gives 95% bands).
    init_phi, init_tau : float
        Initial values for phi and tau.
    init_mu : list, optional
        Initial weight vector of length N. Defaults to the uniform
        simplex mu_i = 1 / N.
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    verbose : bool
        Show a tqdm progress bar during MCMC.
    seed : int, optional
        Seed for the numpy.random.Generator used inside the sampler.
    """

    n_iter: int = Field(default=2000, ge=10)
    burn_in: int = Field(default=1000, ge=0)
    kappa1: float = Field(default=1.0, gt=0)
    kappa2: float = Field(default=1.0, gt=0)
    theta: float = Field(default=0.25, gt=0.0, lt=1.0)
    tau_a: float = Field(default=0.01, gt=0)
    tau_b: float = Field(default=0.1, gt=0)
    n_tau: int = Field(default=11, ge=1)
    tau_min: float = Field(default=1e-6, gt=0)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    init_phi: float = Field(default=0.8, gt=0)
    init_tau: float = Field(default=1.0, gt=0)
    init_mu: Optional[List[float]] = Field(default=None)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)
    seed: Optional[int] = Field(default=None)
