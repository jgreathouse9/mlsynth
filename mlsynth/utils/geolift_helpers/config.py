"""Configuration for the GeoLift market-selection design tool.

Co-located with the helper package (mirrors MAREX / lexscm). Inherits the
experimental-design base :class:`BaseMAREXConfig` (df / outcome / unitid / time
+ panel validation) and adds the market-selection knobs.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
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

    # --- design constraints (geography / interference) ---
    cluster_col: Optional[str] = Field(
        default=None,
        description="Per-unit column of cluster labels (e.g. DMA / state). Markets "
        "in the same cluster interfere: at most one per candidate (treatment "
        "criterion), and a treated market's same-cluster geos are dropped from its "
        "donor pool (control criterion).",
    )
    adjacency: Optional[pd.DataFrame] = Field(
        default=None,
        description="Square unit-by-unit spillover matrix (index/columns are market "
        "labels). Pairs above `spillover_threshold` interfere, combined with "
        "`cluster_col` by logical OR.",
    )
    spillover_threshold: float = Field(
        default=0.0,
        description="Off-diagonal `adjacency` entries strictly above this mark an "
        "interfering pair.",
    )
    stratum_col: Optional[str] = Field(
        default=None,
        description="Per-unit column of stratum labels (region / tier / segment) for "
        "coverage quotas via `min_per_stratum` / `max_per_stratum`.",
    )
    min_per_stratum: Optional[int] = Field(
        default=None,
        description="Require at least this many treated markets from every stratum "
        "that contains a treatment-eligible market ('test in every region').",
    )
    max_per_stratum: Optional[int] = Field(
        default=None,
        description="Allow at most this many treated markets from any stratum (a quota).",
    )
    size_col: Optional[str] = Field(
        default=None,
        description="Per-unit column of market sizes for a treated-unit size band; "
        "out-of-band markets stay available as donors.",
    )
    min_size: Optional[float] = Field(
        default=None, description="Lower bound of the treated-unit size band (inclusive)."
    )
    max_size: Optional[float] = Field(
        default=None, description="Upper bound of the treated-unit size band (inclusive)."
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
    cpic: Optional[float] = Field(
        default=None,
        description="Cost per incremental conversion. When set, each candidate's "
        "required investment = cpic x effect_size x summed-treated-volume is "
        "reported (GeoLift's budget-planning layer).",
    )
    budget: Optional[float] = Field(
        default=None,
        description="Spend cap. When set (with cpic), candidates whose detectable "
        "investment exceeds the budget are dropped from the design.",
    )
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
    n_jobs: int = Field(
        default=1,
        description="Parallel workers for the candidate search (power simulation "
        "and per-candidate design fits, via joblib). 1 (default) runs serially. "
        ">1 or -1 (all cores) parallelizes across candidates; each candidate is "
        "independent and uses the fixed seed, so results are identical to the "
        "serial run -- only faster.",
    )

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
        if self.cpic is not None and self.cpic < 0:
            raise MlsynthConfigError(f"cpic must be >= 0; got {self.cpic}.")
        if self.budget is not None:
            if self.budget <= 0:
                raise MlsynthConfigError(f"budget must be > 0; got {self.budget}.")
            if self.cpic is None:
                raise MlsynthConfigError("budget requires cpic to be set.")
        # design constraints
        if self.min_per_stratum is not None and self.min_per_stratum < 1:
            raise MlsynthConfigError(
                f"min_per_stratum must be >= 1; got {self.min_per_stratum}.")
        if self.max_per_stratum is not None and self.max_per_stratum < 1:
            raise MlsynthConfigError(
                f"max_per_stratum must be >= 1; got {self.max_per_stratum}.")
        if (self.min_per_stratum is not None or self.max_per_stratum is not None) \
                and self.stratum_col is None:
            raise MlsynthConfigError(
                "min_per_stratum / max_per_stratum require stratum_col.")
        if (self.min_size is not None or self.max_size is not None) \
                and self.size_col is None:
            raise MlsynthConfigError("min_size / max_size require size_col.")
        if self.min_size is not None and self.max_size is not None \
                and self.min_size > self.max_size:
            raise MlsynthConfigError(
                f"min_size ({self.min_size}) must be <= max_size ({self.max_size}).")
        return self
