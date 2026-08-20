"""Configuration for the BLTVP estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class BLTVPConfig(BaseEstimatorConfig):
    """Configuration for the Time Varying Parameter Bayesian Lasso (BLTVP).

    Implements:

        Klinenberg, D. (2023). "Synthetic Control with Time Varying
        Coefficients: A State Space Approach with Bayesian Shrinkage."
        Journal of Business & Economic Statistics 41(4):1065-1076.

    Each donor coefficient is split into a time-invariant part ``beta_j`` and
    a random-walk part scaled by ``sqrt(theta)_j`` (eq. 7), and the two are
    shrunk independently under a Bayesian Lasso. The model therefore decides,
    per donor, whether the relationship to the treated unit is time-varying,
    static, or absent -- instead of forcing one answer on every donor.

    Parameters
    ----------
    n_iter : int
        Total Gibbs iterations per chain, including burn-in.
    burn_in : int
        Warm-up iterations discarded per chain before reporting.
    chains : int
        Independent chains, pooled into one set of draws. The reference
        implementation runs one; raise it to judge convergence.
    intercept : bool
        Include an intercept as the ``(J+1)``-th regressor, which is Model 1
        as written in the paper. The author's own replication script
        suppresses it, and the empirical point estimate is insensitive either
        way, so the default follows the script.
    time_varying : bool
        Let the coefficients drift. Setting this ``False`` pins every
        ``sqrt(theta)_j`` to zero, which is the nesting the paper names (p.
        1067): the model collapses to a Bayesian Lasso synthetic control with
        static weights. Useful as a within-estimator comparison.
    ci_alpha : float
        Two-sided significance level for credible intervals (0.05 gives 95%
        bands).
    a1, g0, G0 : float
        Error-variance hierarchy: ``sigma2 ~ InvGamma(a1, a2)`` with
        ``a2 ~ Gamma(g0, G0)``. Defaults are ``shrinkTVP``'s, which is what
        the author's script calls.
    z1, z2 : float
        Gamma hyperprior on the Lasso rate for the time-invariant block.
    z1_dyn, z2_dyn : float
        Gamma hyperprior on the Lasso rate for the time-varying block.
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    verbose : bool
        Reserved for progress reporting.
    seed : int, optional
        Seed for the ``numpy.random.Generator`` used inside the sampler.
    """

    n_iter: int = Field(default=10000, ge=10,
                        description="Total Gibbs iterations per chain, including burn-in.")
    burn_in: int = Field(default=5000, ge=0,
                         description="Warm-up iterations discarded per chain.")
    chains: int = Field(default=1, ge=1,
                        description="Independent chains pooled into one set of draws.")
    intercept: bool = Field(default=False,
                            description="Add an intercept regressor (Model 1 as printed).")
    time_varying: bool = Field(default=True,
                               description="Allow coefficients to drift; False gives the static Bayesian Lasso nesting.")
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0,
                            description="Two-sided level for credible intervals.")
    a1: float = Field(default=2.5, gt=0.0, description="Shape of the sigma2 inverse gamma.")
    g0: float = Field(default=5.0, gt=0.0, description="Shape of the Gamma hyperprior on its rate.")
    G0: float = Field(default=5.0 / 1.5, gt=0.0, description="Rate of the Gamma hyperprior on its rate.")
    z1: float = Field(default=0.001, gt=0.0, description="Gamma shape for the static-block Lasso rate.")
    z2: float = Field(default=0.001, gt=0.0, description="Gamma rate for the static-block Lasso rate.")
    z1_dyn: float = Field(default=0.001, gt=0.0, description="Gamma shape for the dynamic-block Lasso rate.")
    z2_dyn: float = Field(default=0.001, gt=0.0, description="Gamma rate for the dynamic-block Lasso rate.")
    display_graphs: bool = Field(default=False, description="Show the counterfactual plot after fitting.")
    verbose: bool = Field(default=False, description="Reserved for progress reporting.")
    seed: Optional[int] = Field(default=None, description="Seed for the sampler's generator.")

    @model_validator(mode="after")
    def _check_burn_in(self) -> "BLTVPConfig":
        if self.burn_in >= self.n_iter:
            raise MlsynthConfigError(
                f"burn_in={self.burn_in} must be strictly less than "
                f"n_iter={self.n_iter}."
            )
        return self
