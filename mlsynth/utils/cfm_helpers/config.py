"""Configuration for the CFM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class CFMConfig(BaseEstimatorConfig):
    """Configuration for the Causal Factor Model (CFM) estimator.

    Implements Bai & Wang (2026), *"Causal Inference Using Factor Models"*.
    CFM models both potential outcomes within a single factor structure and
    lets the treated unit's factor loadings break at the intervention date,
    targeting the systematic causal effect for a single treated unit. This
    is the Proposition 1 / constant-shift regime: the intervention changes
    treated units' exposure to the common shocks (loadings) but not the
    factor process itself.

    Parameters
    ----------
    factor_selection : {"er", "gr", "bai_ng"}
        How to choose the number of common factors. ``"er"`` and ``"gr"``
        are the Ahn-Horenstein (2013) eigenvalue-ratio and growth-ratio
        estimators (the paper's primary criteria); ``"bai_ng"`` uses the
        Bai-Ng (2002) information criterion. Ignored when ``n_factors`` is
        supplied.
    n_factors : int or None
        Override the data-driven factor count. ``None`` triggers
        ``factor_selection``.
    max_factors : int
        Upper bound passed to the factor-selection routine.
    factor_variance : bool
        Whether to add the factor-estimation variance component ``V_f``
        (Bai & Wang appendix A.2.2) to the standard errors. ``False``
        reports the treated-regression component ``V_reg`` only.
    alpha : float
        Two-sided significance level for CIs.
    """

    factor_selection: Literal["er", "gr", "bai_ng"] = Field(
        default="er",
        description="Factor-count criterion: er / gr (Ahn-Horenstein) or bai_ng.",
    )
    n_factors: Optional[int] = Field(
        default=None, ge=1,
        description="Optional override of the data-driven factor count.",
    )
    max_factors: int = Field(
        default=10, ge=1,
        description="Upper bound on the factor-selection routine.",
    )
    factor_variance: bool = Field(
        default=True,
        description="Add the factor-estimation variance component V_f to SEs.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for CIs.",
    )

    @model_validator(mode="after")
    def check_cfm_params(cls, values: Any) -> Any:
        if values.n_factors is not None and values.n_factors > values.max_factors:
            raise MlsynthConfigError(
                f"n_factors ({values.n_factors}) must not exceed "
                f"max_factors ({values.max_factors})."
            )
        return values
