"""Configuration for the SDIDGEO geo-experiment design.

Co-located with the helper package (mirrors MAREX / lexscm). Inherits the
experimental-design base :class:`BaseMAREXConfig` (df / outcome / unitid / time
plus panel validation) and adds the market-selection knobs.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseMAREXConfig
from ...exceptions import MlsynthConfigError


class SDIDGEOConfig(BaseMAREXConfig):
    """Configuration for the SDIDGEO market-selection design."""

    # --- candidate test region ---
    treatment_size: int = Field(
        ..., description="Number of markets to treat (test-region size)."
    )
    to_be_treated: Optional[List] = Field(
        default=None, description="Markets forced into every candidate region."
    )
    not_to_be_treated: Optional[List] = Field(
        default=None,
        description="Markets barred from any candidate; they remain donors.",
    )
    run_stochastic: bool = Field(
        default=False,
        description="Sample each candidate's neighbours from adjacent "
        "correlation tiers instead of taking the top ones deterministically.",
    )
    stochastic_mode: str = Field(
        default="global",
        description="'global' draws one tier pattern for every anchor; "
        "'per_anchor' draws a fresh one per anchor.",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Column flagging post-treatment periods. Absent, the whole "
        "panel is treated as pre-period history to simulate over.",
    )

    # --- simulation grid ---
    durations: List[int] = Field(
        ..., description="Treatment durations (periods) to scan."
    )
    effect_sizes: List[float] = Field(
        ..., description="Effect sizes to inject, as proportional lifts."
    )
    lookback_window: int = Field(
        default=5,
        description="Backward pseudo-treatment placements per (candidate, "
        "duration).",
    )

    # --- inference and decision rule ---
    alpha: float = Field(
        default=0.1, description="Significance level for detection."
    )
    power_threshold: float = Field(
        default=0.8, description="Power an effect must exceed to count as detected."
    )
    n_draws: int = Field(
        default=200,
        description="Placebo draws behind each standard error (Arkhangelsky "
        "et al. Algorithm 4).",
    )

    # --- budgeting ---
    cpic: Optional[float] = Field(
        default=None,
        description="Cost per incremental conversion. When set, the required "
        "investment = cpic x effect_size x summed-treated-volume is reported.",
    )
    budget: Optional[float] = Field(
        default=None,
        description="Spend cap. When set (with cpic), candidates whose "
        "detectable investment exceeds it are dropped from the design.",
    )

    # --- execution ---
    seed: int = Field(
        default=0,
        description="RNG seed for candidate sampling and placebo draws.",
    )
    n_jobs: int = Field(
        default=1,
        description="Worker processes over candidates. Each candidate is pure "
        "at the fixed seed, so any worker count returns the same shortlist.",
    )

    @model_validator(mode="after")
    def _validate_grid(self) -> "SDIDGEOConfig":
        if self.treatment_size < 1:
            raise MlsynthConfigError(
                f"treatment_size must be >= 1; got {self.treatment_size}.")
        if not self.durations or any(d < 1 for d in self.durations):
            raise MlsynthConfigError(
                "durations must be a non-empty list of positive integers.")
        if not self.effect_sizes:
            raise MlsynthConfigError("effect_sizes must be non-empty.")
        if self.lookback_window < 1:
            raise MlsynthConfigError(
                f"lookback_window must be >= 1; got {self.lookback_window}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(
                f"alpha must be in (0, 1); got {self.alpha}.")
        if not 0.0 < self.power_threshold < 1.0:
            raise MlsynthConfigError(
                f"power_threshold must be in (0, 1); got {self.power_threshold}.")
        if self.n_draws < 3:
            raise MlsynthConfigError(
                "n_draws must be at least 3 for a placebo standard error; got "
                f"{self.n_draws}.")
        if self.stochastic_mode not in ("global", "per_anchor"):
            raise MlsynthConfigError(
                "stochastic_mode must be 'global' or 'per_anchor'; got "
                f"{self.stochastic_mode!r}.")
        if self.budget is not None and self.cpic is None:
            raise MlsynthConfigError(
                "budget needs cpic: without a cost per incremental conversion "
                "there is no investment to compare the budget against.")
        return self
