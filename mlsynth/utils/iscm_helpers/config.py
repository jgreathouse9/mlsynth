"""Configuration for the ISCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from pydantic import Field
from ...config_models import BaseEstimatorConfig


class ISCMConfig(BaseEstimatorConfig):
    """Configuration for the Imperfect Synthetic Controls (ISCM) estimator.

    Powell, D. (2026). *"Imperfect Synthetic Controls,"* Journal of Applied
    Econometrics. Builds synthetic controls for every unit, identifies the
    treatment effect even when the treated unit is outside the convex hull,
    weights units by a data-driven fit metric, and uses Ibragimov-Muller
    inference valid for small donor pools. Inherits the standard ``df`` /
    ``outcome`` / ``treat`` / ``unitid`` / ``time`` interface.

    Parameters
    ----------
    inference : bool
        Run Ibragimov-Muller inference over the per-unit estimates.
        Default True.
    null_value : float
        Null effect ``alpha_0`` for the randomization test. Default 0.
    alpha : float
        Two-sided level for the confidence interval.
    n_draws : int
        Number of Rademacher sign-flip draws for the p-value.
    random_state : int
        Seed for the randomization-test RNG.
    """

    inference: bool = Field(
        default=True,
        description="Run Ibragimov-Muller inference over per-unit estimates.",
    )
    null_value: float = Field(
        default=0.0,
        description="Null effect alpha_0 for the randomization test.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the confidence interval.",
    )
    n_draws: int = Field(
        default=10000, ge=100,
        description="Number of Rademacher sign-flip draws for the p-value.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for the randomization-test RNG.",
    )
