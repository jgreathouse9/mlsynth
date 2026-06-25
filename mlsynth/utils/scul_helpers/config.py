"""Configuration for the SCUL estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SCULConfig(BaseEstimatorConfig):
    """Configuration for Synthetic Control Using Lasso (Hollingsworth & Wing 2022)."""

    donor_variables: Optional[List[str]] = Field(
        default=None,
        description=(
            "Extra panel columns (besides the outcome) to expand the donor pool "
            "with, one block of all donor units per variable -- SCUL's "
            "high-dimensional, multi-type pool (e.g. a price series alongside the "
            "outcome). None (default) uses only the donor units' outcome series."
        ),
    )
    number_initial_periods: int = Field(
        default=5, gt=1,
        description=(
            "Training-window length for the first rolling cross-validation run "
            "(reference NumberInitialTimePeriods)."
        ),
    )
    training_post_length: int = Field(
        default=7, gt=0,
        description=(
            "Out-of-sample window length scored in each cross-validation run "
            "(reference TrainingPostPeriodLength); ideally the post-period length."
        ),
    )
    cv_option: str = Field(
        default="median", pattern="^(median|min)$",
        description=(
            "Penalty-selection rule across CV windows: 'median' (the reference's "
            "robust lambda.median default) or 'min' (smallest mean CV MSE)."
        ),
    )
    cohensd_threshold: float = Field(
        default=0.25, gt=0.0,
        description=(
            "Unit-free pre-treatment fit threshold (mean |gap|/sd). Placebo units "
            "whose fit exceeds it are trimmed from the inference distribution, per "
            "the paper's quality-control recommendation."
        ),
    )
    inference: bool = Field(
        default=True,
        description="Compute the placebo-distribution p-value for the ATT.",
    )

    @model_validator(mode="after")
    def _check_cv_window(self) -> "SCULConfig":
        if self.training_post_length < 1 or self.number_initial_periods < 2:
            raise MlsynthConfigError(  # pragma: no cover - Field gt-constraints already enforce this
                "SCUL needs number_initial_periods >= 2 and training_post_length >= 1."
            )
        return self
