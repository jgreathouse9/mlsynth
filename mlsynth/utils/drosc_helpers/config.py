"""Configuration for :class:`mlsynth.DROSC` (Distributionally Robust Synthetic
Control, Koo & Guo 2026)."""
from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class DROSCConfig(BaseEstimatorConfig):
    """Distributionally Robust Synthetic Control.

    DROSC (Koo & Guo 2026) reports the smallest-magnitude effect consistent with
    the donor weightings the pre-treatment moments cannot rule out -- the
    projection of the origin onto the interval of effects that set admits. Where
    the post-treatment weighting lies in the set, that effect is a lower bound on
    the true one's magnitude and cannot take the opposite sign, so a directional
    claim survives even though the size is understated.

    ``robustness_lambda`` widens the set, so the effect shrinks toward zero as it
    grows; read the sweep, not one value. Optional perturbation-based inference
    returns a (possibly disjoint) union confidence interval, which is a
    confidence set for the effects the data cannot reject.

    References
    ----------
    Koo, T. & Guo, Z. (2026). "Distributionally Robust Synthetic Control:
    Ensuring Robustness Against Highly Correlated Controls and Weight Shifts."
    arXiv:2511.02632.
    """

    robustness_lambda: float = Field(
        default=0.0, ge=0.0,
        description="Robustness radius lambda >= 0: how far a plausible "
                    "post-treatment weighting may violate the pre-treatment "
                    "moment condition. This is a tolerance for weight shift, a "
                    "modelling choice that does not shrink with more data; the "
                    "slack for sampling error in the estimated moments is added "
                    "separately and vanishes as the pre-period grows. 0 trusts "
                    "the classical identification; larger values widen the set "
                    "and drive the reported effect toward zero.",
    )
    inference: bool = Field(
        default=False,
        description="Compute the perturbation-based non-regular union confidence "
                    "interval for the ATT (slower; runs n_perturbations solves).",
    )
    conditional: bool = Field(
        default=False,
        description="Conditional (True) vs unconditional (False) perturbation "
                    "inference. Unconditional resamples the donor second moments; "
                    "conditional holds them at their observed values.",
    )
    dependent: bool = Field(
        default=False,
        description="Use a Newey-West HAC covariance (True) instead of the i.i.d. "
                    "sample covariance (False) for autocorrelated outcomes.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the confidence interval.",
    )
    n_perturbations: int = Field(
        default=500, ge=1,
        description="Number of perturbation draws M for the inference.",
    )
    seed: int = Field(
        default=0, ge=0,
        description="Seed for the perturbation draws (reproducible inference).",
    )
    nu: float = Field(
        default=0.01, ge=0.0,
        description="Tuning constant scaling the residual-based slack in the "
                    "moment band; inflated automatically until the band is "
                    "feasible (paper default 0.01).",
    )
    eta: float = Field(
        default=0.01, ge=0.0,
        description="Tuning constant scaling the lambda-based slack in the moment "
                    "band (paper default 0.01).",
    )
    slack_rate: float = Field(
        default=0.5, gt=0.0,
        description="Exponent c on log(max(T0, N)) in the band-slack rate "
                    "(paper default 1/2).",
    )
    gamma: float = Field(
        default=1.0, ge=0.0,
        description="Ridge inflation factor for near-singular perturbation "
                    "covariances when T0 is small relative to the moment "
                    "dimension (paper default 1).",
    )
    alpha0: float = Field(
        default=0.01, gt=0.0, lt=1.0,
        description="Perturbation-draw significance budget: the outlier-drop "
                    "level, subtracted from alpha for the interval half-width. "
                    "Must be strictly less than alpha.",
    )
    c_sample: float = Field(
        default=0.01, gt=0.0,
        description="Tuning constant for the per-draw band slack in inference; "
                    "inflated automatically until >= 10% of draws are feasible.",
    )

    @field_validator("robustness_lambda")
    @classmethod
    def _finite_lambda(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise MlsynthConfigError("robustness_lambda must be finite.")
        return v

    @model_validator(mode="after")
    def _check_drosc(cls, values: Any) -> Any:
        if values.alpha0 >= values.alpha:
            raise MlsynthConfigError(
                f"alpha0 ({values.alpha0}) must be strictly less than alpha "
                f"({values.alpha}); the interval half-width uses "
                f"(alpha - alpha0)."
            )
        return values
