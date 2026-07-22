"""Configuration for the MVBBSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class MVBBSCConfig(BaseEstimatorConfig):
    """Configuration for the Martinez--Vives-i-Bastida Bayesian synthetic control.

    Implements the outcome-only Bayesian synthetic control of

        Martinez, I. & Vives-i-Bastida, J. (2024). "Bayesian and Frequentist
        Inference for Synthetic Controls." arXiv:2206.01779.

    The no-intervention outcome of the treated unit is modelled as a
    simplex-weighted average of the donor pool with a uniform Dirichlet prior on
    the weights (``w ~ Dir(1)``), a ``HalfNormal`` idiosyncratic scale
    (``sigma``), and a Gaussian likelihood on the pre-treatment window. Both the
    treated and donor series are standardized by their pre-period mean and
    standard deviation before fitting and the counterfactual is transformed
    back to the outcome scale, so the fit is invariant to the units of the
    outcome. The posterior is drawn with NUTS (NumPyro) and the counterfactual
    is the posterior-predictive draw of the treated outcome in absence of
    treatment, giving a full credible band and an ATT credible interval. This
    estimator therefore requires the ``[bayes]`` optional dependency
    (``pip install mlsynth[bayes]``).

    Parameters
    ----------
    n_warmup : int
        NUTS warm-up (adaptation) iterations per chain.
    n_samples : int
        Retained NUTS samples per chain.
    n_chains : int
        Number of NUTS chains.
    target_accept : float
        NUTS target acceptance probability.
    ci_alpha : float
        Two-sided level for the credible interval (0.05 -> 95% band).
    seed : int
        PRNG seed (makes a fit reproducible, including the posterior-predictive
        draw used for the bands).
    display_graphs : bool
        Plot the observed-vs-counterfactual path with the credible band.
    verbose : bool
        Show the NUTS progress bar.
    """

    n_warmup: int = Field(default=1000, ge=1)
    n_samples: int = Field(default=1000, ge=1)
    n_chains: int = Field(default=4, ge=1)
    target_accept: float = Field(default=0.8, gt=0.0, lt=1.0)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = Field(default=0)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)

    @model_validator(mode="after")
    def _check(self) -> "MVBBSCConfig":
        if self.n_samples * self.n_chains < 2:
            raise MlsynthConfigError(
                "MVBBSC needs at least two posterior draws "
                "(n_samples * n_chains >= 2)."
            )
        return self
