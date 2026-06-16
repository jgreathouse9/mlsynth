"""Configuration for the Orthogonalized Synthetic Control (ORTHSC) estimator.

Fry, J. (2026). "Orthogonalized Synthetic Controls." ORTHSC is an IV synthetic
control whose ATT estimate is Neyman-orthogonalized with respect to the
(partially identified, simplex-constrained) control weights, with a
fixed-smoothing Series-HAC variance and a Sun (2013) bandwidth giving a t-test
that controls size without a consistent variance.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class OrthSCConfig(BaseEstimatorConfig):
    """Configuration for ORTHSC.

    Beyond the standard panel fields (``df``, ``outcome``, ``treat``,
    ``unitid``, ``time``), ORTHSC needs a set of untreated units to use as
    instruments -- Fry's method uses the outcomes of units excluded from the
    control pool as instruments for the control weights.
    """

    instruments: List[str] = Field(
        ..., description="Untreated unit labels to use as instruments (outcomes "
        "of units excluded from the control pool, per Fry).")
    controls: Optional[List[str]] = Field(
        default=None, description="Control-pool unit labels (the synthetic "
        "control donors). If omitted, every donor not named as an instrument is "
        "used as a control.")
    alpha: float = Field(default=0.05,
                         description="Significance level for the t-test CI.")
    beta0: float = Field(default=0.0,
                         description="Null value of the ATT for the t-test.")
    include_constant: bool = Field(
        default=True, description="Add a constant as an extra instrument "
        "(mean-matching moment), as in the reference.")

    @model_validator(mode="after")
    def _check(self):
        if not self.instruments:
            raise MlsynthConfigError("ORTHSC requires at least one instrument unit.")
        if len(set(self.instruments)) != len(self.instruments):
            raise MlsynthConfigError("instruments contains duplicate labels.")
        if self.controls is not None:
            overlap = set(self.controls) & set(self.instruments)
            if overlap:
                raise MlsynthConfigError(
                    f"units cannot be both control and instrument: {sorted(overlap)}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(f"alpha must be in (0, 1); got {self.alpha}.")
        return self
