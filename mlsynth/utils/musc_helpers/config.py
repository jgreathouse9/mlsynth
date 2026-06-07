"""Configuration for the MUSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class MUSCConfig(BaseEstimatorConfig):
    """Configuration for the Modified Unbiased Synthetic Control (MUSC) estimator.

    MUSC is the Bottmer, Imbens, Spiess & Warnick (2024 JBES)
    modification of the synthetic control method: a single column-
    sums-to-zero linear restriction is added to the standard SC
    quadratic programme so that the resulting estimator is exactly
    unbiased under random assignment of which unit is treated (the
    paper's Lemma 1).
    """

    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description=(
            "Two-sided significance level for both the Normal-approximation "
            "and the randomization-based CI surfaced on the MUSC inference "
            "object. Defaults to 0.05 (95% CI)."
        ),
    )
    run_inference: bool = Field(
        default=True,
        description=(
            "Whether to compute the Proposition 1 unbiased variance "
            "estimator (eq. 3.3) and the Section 3.5 randomization-based "
            "confidence interval. The randomization CI requires one "
            "leave-one-out QP refit per donor unit, so disabling it is "
            "useful for exploratory fits on large panels."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the MUSC quadratic programme. "
            "Defaults to None, which routes to cvxpy's default for the "
            "problem class (CLARABEL on cvxpy >= 1.4)."
        ),
    )
