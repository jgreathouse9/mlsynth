"""Configuration for the DPSC (differentially private synthetic control) estimator."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...config_models import BaseEstimatorConfig


class DPSCConfig(BaseEstimatorConfig):
    """Configuration for :class:`mlsynth.DPSC`.

    Differentially private synthetic control (Rho, Cummings & Misra 2023): a
    ridge synthetic control whose released counterfactual satisfies
    ``epsilon``-differential privacy over the *donor pool*. Privacy is spent in
    two stages -- learning the regression coefficients (``epsilon1``) and
    releasing the post-intervention prediction (``epsilon2``) -- so the total
    budget is ``epsilon1 + epsilon2``.

    The default ``mechanism="objective"`` is the objective-perturbation
    estimator (Algorithm 3), which perturbs the ridge objective; it is far more
    stable than output perturbation on typical panels. ``"output"`` (Algorithm
    2) perturbs the fitted coefficients directly and is provided for
    completeness -- its variance is large unless the donor pool is big.
    """

    mechanism: Literal["objective", "output"] = Field(
        default="objective",
        description="DP-ERM mechanism: 'objective' (Alg 3, perturb the ridge "
                    "objective; stable, recommended) or 'output' (Alg 2, perturb "
                    "the fitted coefficients; high variance on small donor pools).",
    )
    epsilon1: float = Field(
        default=1.0, gt=0.0,
        description="Privacy budget for learning the regression coefficients "
                    "(Stage 1). Smaller is more private and noisier.",
    )
    epsilon2: float = Field(
        default=1.0, gt=0.0,
        description="Privacy budget for releasing the post-intervention "
                    "prediction (Stage 2). The total budget is epsilon1 + epsilon2.",
    )
    ridge_lambda: float = Field(
        default=10.0, gt=0.0,
        description="Ridge penalty lambda. It regularizes the fit and, through "
                    "the sensitivity, controls the noise scale: larger lambda "
                    "lowers sensitivity (less noise) at the cost of a more biased "
                    "(shrunk) synthetic control.",
    )
    delta: float = Field(
        default=0.0, ge=0.0, lt=1.0,
        description="Approximate-DP slack for the objective mechanism. delta = 0 "
                    "(default) gives pure epsilon-DP with Laplace noise; delta > 0 "
                    "gives (epsilon, delta)-DP with Gaussian noise.",
    )
    n_draws: int = Field(
        default=500, ge=1,
        description="Number of independent privatized draws used to quantify the "
                    "privacy noise on the ATT (its Monte Carlo standard error and "
                    "interval). The reported counterfactual is a single seeded "
                    "release; n_draws only sizes the reported uncertainty.",
    )
    seed: int = Field(
        default=0,
        description="Seed for the privatized release and the noise Monte Carlo "
                    "(reproducibility).",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the privacy-noise interval on the ATT "
                    "(1 - alpha).",
    )
