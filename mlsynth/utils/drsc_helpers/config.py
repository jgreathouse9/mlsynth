"""Configuration for DRSC (Wied 2026).

Two inputs beyond the usual four columns, and both are intrinsic to the method
rather than conveniences. ``covariates`` names the columns forming the design
vector :math:`x`, because the estimand is a *conditional* distribution.
``evaluation_points`` names the covariate values to report it at, because
:math:`\\widehat f_t(x)` is a function of :math:`x`: unlike an ATT there is no
single number to return, and asking the caller to say which :math:`x` they care
about is the honest interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class DRSCConfig(BaseEstimatorConfig):
    """Typed configuration for :class:`mlsynth.DRSC`."""

    covariates: List[str] = Field(
        ...,
        description=(
            "Columns forming the design vector x, in order. An intercept is "
            "prepended automatically -- do not supply a constant column. "
            "Derived terms are the caller's responsibility: the paper's "
            "specification is [education, experience, experience^2], all "
            "standardised, so the squared term is passed as its own column."),
    )
    evaluation_points: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "Covariate values to report the effect at, as "
            "{label: {covariate: value}}. Every covariate must be named in "
            "every point. The conditional effect is a function of x, so at "
            "least one point is required; there is no meaningful default."),
    )
    link: Literal["probit", "logit"] = Field(
        default="probit",
        description=(
            "Link Lambda in F(y|x) = Lambda(x'theta(y)). Dette et al. (2025) "
            "find the choice has negligible impact on estimates; probit is "
            "the paper's default."),
    )
    n_grid: int = Field(
        default=32, gt=1,
        description=(
            "Number of outcome grid points y_l at which the distribution "
            "regression is run. Points are quantiles of the pooled outcome "
            "between `grid_lo` and `grid_hi`; ties are collapsed, so the "
            "active grid may be shorter than requested."),
    )
    grid_lo: float = Field(default=0.10, gt=0.0, lt=1.0,
                           description="Lowest quantile of the outcome grid.")
    grid_hi: float = Field(default=0.90, gt=0.0, lt=1.0,
                           description="Highest quantile of the outcome grid.")
    focus_region: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Optional (low, high) window on the outcome scale restricting the "
            "focused supremum test -- the paper's minimum-wage corridor. The "
            "full-support test is always reported alongside it."),
    )
    ridge: float = Field(
        default=0.0, ge=0.0,
        description=(
            "Ridge added to the Gram matrix before the weight solve "
            "(Remark 2.5). The Gram matrix is ill-conditioned by nature -- "
            "kappa ~ 1e5 on the paper's panel -- so a small value stabilises "
            "the weights at the cost of shrinking them toward 1/J. 0 (the "
            "default) reproduces the paper."),
    )
    alpha: float = Field(
        default=0.10, gt=0.0, lt=1.0,
        description="Level of the supremum test and the one-sided lower bound.")
    n_gp_draws: int = Field(
        default=10_000, gt=0,
        description=(
            "Gaussian-process draws for the supremum test's critical value. "
            "Note these are not reproducible across LAPACK builds even at a "
            "fixed seed: eigenvector signs from `eigh` are "
            "implementation-defined. The test statistic itself is exact."),
    )
    seed: int = Field(default=1992,
                      description="Seed for the Gaussian-process simulation.")

    @field_validator("covariates")
    @classmethod
    def _check_covariates(cls, v: List[str]) -> List[str]:
        if not v:
            raise MlsynthConfigError(
                "covariates is empty; DRSC estimates a *conditional* "
                "distribution and needs at least one covariate. For the "
                "unconditional analogue use DSC.")
        if len(set(v)) != len(v):
            dup = sorted({c for c in v if v.count(c) > 1})
            raise MlsynthConfigError(f"duplicate covariate column(s) {dup}.")
        return v

    @model_validator(mode="after")
    def _check_evaluation_points(self) -> "DRSCConfig":
        if not self.evaluation_points:
            raise MlsynthConfigError(
                "evaluation_points is empty; the conditional effect is a "
                "function of x, so at least one point must be named.")
        for label, point in self.evaluation_points.items():
            missing = [c for c in self.covariates if c not in point]
            extra = [c for c in point if c not in self.covariates]
            if missing or extra:
                raise MlsynthConfigError(
                    f"evaluation point {label!r} does not match the "
                    f"covariates: missing {missing}, unexpected {extra}. "
                    "Every covariate must be given a value at every point.")
        if self.grid_lo >= self.grid_hi:
            raise MlsynthConfigError(
                f"grid_lo ({self.grid_lo}) must be below grid_hi "
                f"({self.grid_hi}).")
        if self.focus_region is not None:
            lo, hi = self.focus_region
            if not lo < hi:
                raise MlsynthConfigError(
                    f"focus_region must be (low, high) with low < high; "
                    f"got ({lo}, {hi}).")
        return self
