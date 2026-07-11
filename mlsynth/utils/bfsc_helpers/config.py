"""Configuration for the BFSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class BFSCConfig(BaseEstimatorConfig):
    """Configuration for Bayesian Factor Synthetic Control (BFSC).

    Implements:

        Pinkney, S. (2021). "An Improved and Extended Bayesian Synthetic
        Control." arXiv:2103.16244.

    BFSC models the no-intervention outcome of every unit by a Bayesian latent
    factor model -- ``y_jt = F_t . beta_j + delta_t + kappa_j + e_jt`` -- with
    the factor matrix ``F`` estimated jointly with its loadings, a horseshoe+
    prior on the loadings that shrinks unused factors (so the factor count is a
    soft upper bound, not a hard choice), and year (``delta``) and unit
    (``kappa``) effects. The treated unit's post-intervention outcomes are
    masked and imputed, so the posterior of that imputation is the
    counterfactual; the treatment effect follows with a full credible band. The
    posterior is drawn with NUTS (NumPyro), so this estimator requires the
    ``[bayes]`` optional dependency (``pip install mlsynth[bayes]``).

    Parameters
    ----------
    n_factors : int
        Number of latent factors ``L`` (an upper bound: the horseshoe+ prior
        prunes the ones the data do not support). Paper default 8.
    n_warmup : int
        NUTS warm-up (adaptation) iterations per chain.
    n_samples : int
        Retained NUTS samples per chain.
    n_chains : int
        Number of NUTS chains.
    target_accept : float
        NUTS target acceptance probability. The heavy-tailed horseshoe+ wants a
        high value; 0.95 by default.
    ci_alpha : float
        Two-sided level for the credible interval (0.05 -> 95% band).
    seed : int
        PRNG seed (makes a fit reproducible).
    display_graphs : bool
        Plot the observed-vs-counterfactual path with the credible band.
    verbose : bool
        Show the NUTS progress bar.
    """

    n_factors: int = Field(default=8, ge=1)
    n_warmup: int = Field(default=500, ge=1)
    n_samples: int = Field(default=500, ge=1)
    n_chains: int = Field(default=4, ge=1)
    target_accept: float = Field(default=0.95, gt=0.0, lt=1.0)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = Field(default=0)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)

    @model_validator(mode="after")
    def _check(self) -> "BFSCConfig":
        if self.n_samples * self.n_chains < 2:
            raise MlsynthConfigError(
                "BFSC needs at least two posterior draws "
                "(n_samples * n_chains >= 2)."
            )
        return self
