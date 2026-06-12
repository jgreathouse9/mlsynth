"""Configuration for the GeoLift market-selection design tool.

Co-located with the helper package (mirrors MAREX / lexscm). Inherits the
experimental-design base :class:`BaseMAREXConfig` (df / outcome / unitid / time
+ panel validation) and adds the market-selection knobs.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseMAREXConfig
from ...exceptions import MlsynthConfigError


class GeoLiftConfig(BaseMAREXConfig):
    """Configuration for the GeoLift market-selection design (GEOLIFT)."""

    # --- candidate test region ---
    treatment_size: int = Field(
        ..., description="Number of markets to treat (test-set size)."
    )
    to_be_treated: Optional[List] = Field(
        default=None, description="Units forced into every candidate test set."
    )
    not_to_be_treated: Optional[List] = Field(
        default=None, description="Units forbidden from any candidate (stay donors)."
    )

    # --- power-simulation grid ---
    durations: List[int] = Field(
        ..., description="Treatment durations (periods) to scan."
    )
    effect_sizes: List[float] = Field(
        ..., description="Proportional effect sizes to inject (include 0.0)."
    )
    lookback_window: int = Field(
        default=1, description="Number of backward lookback placements per (candidate, duration)."
    )

    # --- pre/post split ---
    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 column marking post-treatment periods; when "
        "given the design uses only the pre-period (same result as a pre-only panel).",
    )

    # --- estimation / inference ---
    how: str = Field(default="sum", description="Treated aggregation: 'sum' or 'mean'.")
    augment: Optional[str] = Field(
        default="ridge", description="Point-fit estimator: 'ridge' (ASCM) or None (simplex)."
    )
    fixed_effects: bool = Field(
        default=True,
        description="Unit fixed effects (augsynth/GeoLift 'fixed_effects=TRUE', the "
        "default): demean every unit by its own pre-period mean before fitting. With "
        "the mean-of-units treated aggregate this reproduces augsynth's effect "
        "estimate and conformal p-value.",
    )
    alpha: float = Field(default=0.1, description="Significance level for detection.")
    power_threshold: float = Field(default=0.8, description="Power needed to 'detect' an effect (MDE).")
    ns: int = Field(default=1000, description="Conformal permutation count (iid only).")
    conformal_type: str = Field(
        default="iid",
        description="Conformal permutation scheme: 'iid' (independent, augsynth/GeoLift "
        "default) or 'block' (the T moving-block cyclic shifts, for serially-dependent "
        "residuals).",
    )

    # --- candidate generation ---
    run_stochastic: bool = Field(default=False, description="Use GeoLift's stochastic paired-jitter generation.")
    stochastic_mode: str = Field(default="global", description="'global' (faithful) or 'per_anchor' (corrected).")
    seed: int = Field(default=0, description="RNG seed (candidate sampling + conformal permutations).")

    # --- display ---
    display_graphs: bool = Field(
        default=True,
        description="Plot the recommended design during fit (design phase, or the "
        "realized post phase when post_col leaves a post window).",
    )

    @model_validator(mode="after")
    def _check_design(self):
        if self.treatment_size < 1:
            raise MlsynthConfigError(f"treatment_size must be >= 1; got {self.treatment_size}.")
        if not self.durations or any(d < 1 for d in self.durations):
            raise MlsynthConfigError("durations must be a non-empty list of positive integers.")
        if not self.effect_sizes:
            raise MlsynthConfigError("effect_sizes must be non-empty.")
        if self.lookback_window < 1:
            raise MlsynthConfigError(f"lookback_window must be >= 1; got {self.lookback_window}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(f"alpha must be in (0, 1); got {self.alpha}.")
        if not 0.0 < self.power_threshold < 1.0:
            raise MlsynthConfigError(f"power_threshold must be in (0, 1); got {self.power_threshold}.")
        if self.how not in ("sum", "mean"):
            raise MlsynthConfigError(f"how must be 'sum' or 'mean'; got {self.how!r}.")
        if self.augment not in ("ridge", None):
            raise MlsynthConfigError(f"augment must be 'ridge' or None; got {self.augment!r}.")
        if self.conformal_type not in ("iid", "block"):
            raise MlsynthConfigError(
                f"conformal_type must be 'iid' or 'block'; got {self.conformal_type!r}."
            )
        return self
