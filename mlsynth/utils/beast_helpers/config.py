"""Config for the BEAST immunized doubly-robust synthetic control."""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class BEASTConfig(BaseEstimatorConfig):
    """Configuration for :class:`mlsynth.BEAST`.

    BEAST (Bléhaut, D'Haultfœuille, L'Hour & Tsybakov 2021) is a doubly-robust
    synthetic control: covariate-balancing (exponential-tilting) donor weights
    with an ℓ₁-penalised calibration, plus an immunizing outcome regression,
    giving an ATT that is asymptotically normal with an analytic standard error.

    It targets a *sparse, informative* covariate regime (a handful of predictors
    truly drive selection). It is not built for over-saturated ``p ~ n`` sets,
    where the balancing degenerates; that is guarded by ``balance_tol``.
    """

    covariates: List[str] = Field(
        ...,
        description="Unit-level covariates to balance (pre-treatment means, "
                    "normalised). At least one is required.",
    )
    outcome_lags: Optional[List[Any]] = Field(
        default=None,
        description="Pre-treatment time labels whose outcome values enter the "
                    "balancing design as extra covariates (Abadie-style lagged "
                    "outcomes, e.g. [1970, 1980, 1988]).",
    )
    c_cal: float = Field(
        default=0.03, gt=0.0,
        description="Calibration (balancing) penalty constant in the "
                    "Belloni-Chernozhukov-Hansen level c*Phi^{-1}(1-g/2p)/sqrt(n).",
    )
    c_ort: float = Field(
        default=0.3, gt=0.0,
        description="Orthogonality (outcome-regression) penalty constant.",
    )
    immunity: bool = Field(
        default=True,
        description="If True, use the immunized (doubly-robust) residual "
                    "y - X mu; if False, the non-immunized plug-in.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the ATT confidence interval (1 - alpha).",
    )
    balance_tol: float = Field(
        default=0.05, gt=0.0,
        description="Tolerance for the balance-validity check |sum(W) - 1|; "
                    "exceeding it raises MlsynthEstimationError (over-saturated "
                    "high-dimensional regime the method is not built for).",
    )

    @model_validator(mode="after")
    def _check(self):
        if not self.covariates:
            raise MlsynthConfigError("BEAST: at least one covariate is required.")
        if self.outcome_lags is not None and len(self.outcome_lags) == 0:
            raise MlsynthConfigError(
                "BEAST: outcome_lags must be None or a non-empty list of time labels.")
        return self
