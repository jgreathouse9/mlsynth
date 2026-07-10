"""Configuration for the BSCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class BSCMConfig(BaseEstimatorConfig):
    """Configuration for the Bayesian Synthetic Control Methods (BSCM).

    Implements:

        Kim, S., Lee, C., & Gupta, S. (2020). "Bayesian Synthetic Control
        Methods." Journal of Marketing Research 57(5):831-852.

    BSCM regresses the treated unit on the donor pool with *no* simplex
    constraint (weights may be negative and need not sum to one), regularising
    the weights with a Bayesian global-local shrinkage prior. Two priors are
    offered: the ``horseshoe`` (global-local continuous shrinkage) and the
    ``spike_slab`` (discrete variable selection). Posterior samples are drawn
    by a pure-numpy Gibbs sampler, giving a counterfactual with credible bands
    and an ATT credible interval without any MCMC-engine dependency.

    Parameters
    ----------
    prior : {"horseshoe", "spike_slab"}
        Shrinkage prior on the donor weights. ``horseshoe`` is the paper's
        default (faster, continuous); ``spike_slab`` additionally reports
        per-donor inclusion probabilities.
    n_iter : int
        Total Gibbs iterations per chain (including burn-in).
    burn_in : int
        Warm-up iterations discarded per chain before reporting.
    chains : int
        Number of independent chains, sampled in one vectorised loop and
        pooled. Multiple chains enable convergence diagnostics.
    spike_scale : float
        Standard deviation of the spike component for ``spike_slab`` (the
        paper fixes the spike variance to 0.001, i.e. scale ~0.0316).
        Ignored for the horseshoe.
    ci_alpha : float
        Two-sided significance level for credible intervals (0.05 gives 95%
        bands).
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    verbose : bool
        Reserved for progress reporting.
    seed : int, optional
        Seed for the ``numpy.random.Generator`` used inside the sampler.
    """

    prior: Literal["horseshoe", "spike_slab"] = Field(default="horseshoe")
    n_iter: int = Field(default=2000, ge=10)
    burn_in: int = Field(default=1000, ge=0)
    chains: int = Field(default=4, ge=1)
    spike_scale: float = Field(default=(0.001) ** 0.5, gt=0)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)
    seed: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _check_burn_in(self) -> "BSCMConfig":
        if self.burn_in >= self.n_iter:
            raise MlsynthConfigError(
                f"burn_in={self.burn_in} must be strictly less than "
                f"n_iter={self.n_iter}."
            )
        return self
