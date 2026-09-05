"""Configuration for the GPITS estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError
from .kernels import KERNELS, PERIODIC_KERNELS


class GPITSConfig(BaseEstimatorConfig):
    """Settings for Gaussian-process interrupted time series (Cho 2026)."""

    kernel: str = Field(
        default="gaussian",
        description=(
            "Covariance function. 'gaussian' is the paper's starting point "
            "(Eq. 14) and needs no 'period', but it is stationary, so far from "
            "the pre-period it reverts to the prior and its band flattens at a "
            "ceiling. 'gaussian_periodic_linear' is the working form (Eq. 15) "
            "and what you want for a seasonal series or a longer horizon: it "
            "adds a cycle at 'period' and a linear trend, so the band keeps "
            "widening with distance instead of levelling off. It requires "
            "'period'."
        ),
    )
    period: Optional[float] = Field(
        default=None,
        description=(
            "Seasonal period of the periodic component, in time steps (12 for "
            "monthly data with an annual cycle). Required when 'kernel' is "
            "periodic; ignored otherwise."
        ),
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Columns folded into the GP design alongside the time index. "
            "Period indicators (month, day-of-week) let the kernel learn "
            "period-specific baselines without a parametric mean function."
        ),
    )
    categorical_covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Subset of 'covariates' to one-hot encode. Encoded columns are "
            "scaled by sqrt(0.5) and are not centred, matching the reference."
        ),
    )
    length_scale: Optional[float] = Field(
        default=None,
        description=(
            "Kernel length-scale b. None selects it by the variance-maximising "
            "rule of Hartman et al. (2025), which reads only the covariates."
        ),
    )
    noise_variance: Optional[float] = Field(
        default=None,
        description=(
            "Observation-noise variance, which doubles as the ridge penalty on "
            "the RKHS norm (Eq. 16). None fits it by marginal likelihood with "
            "the length-scale already fixed."
        ),
    )
    interval_type: str = Field(
        default="prediction",
        description=(
            "'prediction' adds the noise variance to the band, so it covers a "
            "new observation; 'confidence' covers the latent mean only."
        ),
    )
    alpha: float = Field(
        default=0.05,
        description="Two-sided level for the counterfactual band and the ATT interval.",
    )
    placebo_periods: Optional[int] = Field(
        default=None,
        description=(
            "Run the temporal placebo check (Section 3.3) on this many "
            "pre-treatment periods: refit on everything earlier and predict "
            "one step ahead, where the true effect is zero. None skips it."
        ),
    )

    @model_validator(mode="after")
    def check_gpits_params(self) -> "GPITSConfig":
        if self.kernel not in KERNELS:
            raise MlsynthConfigError(
                f"'kernel' must be one of {sorted(KERNELS)}; got {self.kernel!r}."
            )
        if self.kernel in PERIODIC_KERNELS:
            if self.period is None:
                raise MlsynthConfigError(
                    f"kernel={self.kernel!r} needs a 'period' (e.g. 12 for "
                    "monthly data with an annual cycle)."
                )
            if not self.period > 0:
                raise MlsynthConfigError(
                    f"'period' must be strictly positive; got {self.period}."
                )
        if self.length_scale is not None and not self.length_scale > 0:
            raise MlsynthConfigError(
                f"'length_scale' must be strictly positive; got {self.length_scale}."
            )
        if self.noise_variance is not None and not self.noise_variance > 0:
            raise MlsynthConfigError(
                f"'noise_variance' must be strictly positive; got {self.noise_variance}."
            )
        if self.interval_type not in ("prediction", "confidence"):
            raise MlsynthConfigError(
                "'interval_type' must be 'prediction' or 'confidence'; "
                f"got {self.interval_type!r}."
            )
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(
                f"'alpha' must lie strictly between 0 and 1; got {self.alpha}."
            )
        if self.placebo_periods is not None and self.placebo_periods < 1:
            raise MlsynthConfigError(
                f"'placebo_periods' must be >= 1 when set; got {self.placebo_periods}."
            )
        if self.categorical_covariates:
            declared = set(self.covariates or ())
            unknown = [c for c in self.categorical_covariates if c not in declared]
            if unknown:
                raise MlsynthConfigError(
                    f"'categorical_covariates' must be a subset of 'covariates'; "
                    f"{unknown} are not declared as covariates."
                )
        return self
