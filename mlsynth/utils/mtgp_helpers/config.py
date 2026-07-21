"""Configuration for the MTGP estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class MTGPConfig(BaseEstimatorConfig):
    """Configuration for Multitask Gaussian Process synthetic control (MTGP).

    Implements the Gaussian model of:

        Ben-Michael, E., Arbour, D., Feller, A., Franks, A., & Raphael, S.
        (2023). "Estimating the effects of a California gun control program with
        multitask Gaussian processes." Annals of Applied Statistics 17(2).

    MTGP models the control potential outcomes with a multitask Gaussian process
    whose kernel is separable over time and units: a global time-GP trend, a
    low-rank (rank ``n_factors``) intrinsic-coregionalization term whose latent
    factors carry a squared-exponential smoothness prior over time, and unit
    intercepts. The treated unit's post-intervention cells are masked and
    imputed, so the posterior of that imputation is the counterfactual, giving
    the treatment effect a credible band that widens post-treatment. The
    posterior is drawn with NUTS (NumPyro, in double precision), so this
    estimator requires the ``[bayes]`` optional dependency
    (``pip install mlsynth[bayes]``).

    Parameters
    ----------
    n_factors : int
        Rank of the unit (intrinsic-coregionalization) kernel -- the number of
        shared latent time-GP factors. Paper default 5.
    population : str, optional
        Column giving each unit-period's population (or exposure). When
        provided, the observation noise scales as ``sqrt(1 / population)``
        (heteroskedastic by size, as in the paper); otherwise the noise is
        homoskedastic.
    n_warmup : int
        NUTS warm-up (adaptation) iterations per chain.
    n_samples : int
        Retained NUTS samples per chain (total, across chains).
    n_chains : int
        Number of NUTS chains.
    target_accept : float
        NUTS target acceptance probability (0.9 by default, as in the paper).
    max_tree_depth : int
        NUTS maximum tree depth (13 by default, as in the paper).
    ci_alpha : float
        Two-sided level for the credible interval (0.05 -> 95% band).
    seed : int
        PRNG seed (makes a fit reproducible).
    display_graphs : bool
        Plot the observed-vs-counterfactual path with the credible band.
    verbose : bool
        Show the NUTS progress bar.
    """

    n_factors: int = Field(default=5, ge=1)
    population: Optional[str] = Field(default=None)
    n_warmup: int = Field(default=1000, ge=1)
    n_samples: int = Field(default=1000, ge=1)
    n_chains: int = Field(default=4, ge=1)
    target_accept: float = Field(default=0.9, gt=0.0, lt=1.0)
    max_tree_depth: int = Field(default=13, ge=1)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = Field(default=0)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)

    @model_validator(mode="after")
    def _check(self) -> "MTGPConfig":
        if self.n_samples * self.n_chains < 2:
            raise MlsynthConfigError(
                "MTGP needs at least two posterior draws "
                "(n_samples * n_chains >= 2)."
            )
        return self
