"""Configuration for DMLFM."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class DMLFMConfig(BaseEstimatorConfig):
    """Dynamic multilevel latent factor model (Pang, Liu & Xu 2022).

    Defaults follow ``pblasso`` 1.0.8, the package behind the paper's figures.
    """

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Covariate columns. Each enters with a constant coefficient, and "
            "with a unit- or time-varying coefficient according to ``re``. "
            "Pass columns already aggregated the way the analysis wants -- "
            "DMLFM reads them per observation and does not average them."))
    re: Literal["none", "unit", "time", "both"] = Field(
        default="time",
        description=(
            "Which coefficient blocks vary. 'unit' adds unit-varying "
            "coefficients, 'time' adds time-varying ones, 'both' adds each, "
            "'none' keeps every coefficient constant."))
    r: int = Field(
        default=10, ge=0,
        description=(
            "Upper bound on the number of factors. Shrinkage on the loading "
            "scales switches unneeded factors off, so this is a ceiling, not a "
            "choice of the factor count."))
    ar1: bool = Field(
        default=True,
        description="Give the time-varying coefficients and factors AR(1) dynamics.")
    niter: int = Field(default=25000, gt=0, description="Total MCMC iterations.")
    burn: int = Field(default=5000, ge=0, description="Burn-in iterations to discard.")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0,
                         description="Credible-interval level; 0.05 gives 95% bands.")
    seed: int = Field(default=0, description="RNG seed.")

    xlasso: bool = Field(default=False, description="Shrink the constant coefficients.")
    zlasso: bool = Field(default=False,
                         description="Shrink the unit-varying coefficient scales.")
    alasso: bool = Field(default=False,
                         description="Shrink the time-varying coefficient scales.")
    flasso: bool = Field(default=True,
                         description="Shrink the factor loading scales, selecting factors.")

    a1: float = Field(default=0.001, description="Gamma shape for the constant-coefficient penalty.")
    a2: float = Field(default=0.001, description="Gamma rate for the constant-coefficient penalty.")
    b1: float = Field(default=0.001, description="Gamma shape for the unit-varying penalty.")
    b2: float = Field(default=0.001, description="Gamma rate for the unit-varying penalty.")
    c1: float = Field(default=0.001, description="Gamma shape for the time-varying penalty.")
    c2: float = Field(default=0.001, description="Gamma rate for the time-varying penalty.")
    p1: float = Field(default=0.001, description="Gamma shape for the factor-selection penalty.")
    p2: float = Field(default=0.001, description="Gamma rate for the factor-selection penalty.")
    e1: float = Field(default=0.001, description="Inverse-gamma shape for the error variance.")
    e2: float = Field(default=0.001, description="Inverse-gamma rate for the error variance.")

    @model_validator(mode="after")
    def _check(self) -> "DMLFMConfig":
        if self.burn >= self.niter:
            raise MlsynthConfigError(
                f"burn ({self.burn}) must be below niter ({self.niter})")
        if self.r == 0 and self.re == "none" and not self.covariates:
            raise MlsynthConfigError(
                "r=0 with re='none' and no covariates leaves nothing to fit")
        return self
