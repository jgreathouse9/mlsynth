"""Configuration for the rolling-transformation DiD estimator (ROLLDID).

Co-located with the helper package (mirrors the MAREX / lexscm
layout). Inherits :class:`BaseEstimatorConfig` (df / outcome / treat / unitid /
time + panel validation) and adds the rolling-DiD knobs.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class ROLLDIDConfig(BaseEstimatorConfig):
    """Configuration for ROLLDID (Lee & Wooldridge rolling-transformation DiD)."""

    rolling: str = Field(
        default="demean",
        description="Unit-level pre-treatment transformation: 'demean' "
        "(Procedure 2.1, remove the pre-period mean) or 'detrend' (Procedure "
        "3.1, remove a pre-period linear trend).",
    )
    inference: str = Field(
        default="exact",
        description="Inference for the cross-sectional regression: 'exact' "
        "(t_{N-2} under classical-linear-model normality), 'hc3' "
        "(heteroskedasticity-robust), or 'ri' (randomization inference).",
    )
    alpha: float = Field(
        default=0.05, description="Significance level for confidence intervals."
    )
    ri_reps: int = Field(
        default=1000, description="Permutations for randomization inference."
    )
    seed: int = Field(default=0, description="RNG seed for randomization inference.")

    @model_validator(mode="after")
    def _check_rolldid(self):
        if self.rolling not in ("demean", "detrend"):
            raise MlsynthConfigError(
                f"rolling must be 'demean' or 'detrend'; got {self.rolling!r}.")
        if self.inference not in ("exact", "hc3", "ri"):
            raise MlsynthConfigError(
                f"inference must be 'exact', 'hc3', or 'ri'; got {self.inference!r}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(f"alpha must be in (0, 1); got {self.alpha}.")
        if self.ri_reps < 1:
            raise MlsynthConfigError(f"ri_reps must be >= 1; got {self.ri_reps}.")
        return self
