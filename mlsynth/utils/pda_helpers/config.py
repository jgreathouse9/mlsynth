"""Configuration for the PDA estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class PDAConfig(BaseEstimatorConfig):
    """Configuration for the Panel Data Approach (PDA) estimator."""
    method: str = Field(default="fs", description="Type of PDA to use: 'LASSO', 'l2', or 'fs'.", pattern="^(LASSO|l2|fs)$")
    methods: Optional[List[str]] = Field(default=None, description="Optional list of PDA variants to run; overrides `method` when set.")
    tau: Optional[float] = Field(default=None, description="User-specified treatment effect value (used as tau_l2 for 'l2' method).")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Significance level for confidence intervals and ATE inference.")
    fs_intercept: bool = Field(default=False, description="Forward-selection only: include a constant in the donor regression. False (default) matches Shi & Huang's simulation (no intercept, valid size on mean-zero factor data); True matches the released fsPDA R package (intercept, for panels with genuine level differences).")
    lrvar_lag: Optional[int] = Field(default=None, description="Forward-selection only: Bartlett-kernel truncation lag for the Newey-West long-run variance in the post-selection t-test. None (default) uses the released fsPDA package's rule floor(T2**(1/4)); a supplied value must be a non-negative integer no larger than floor(sqrt(T2)).")
    l2_standardize: bool = Field(default=True, description="L2-relaxation only: standardise (demean + unit-variance scale) the treated and control series before solving, matching the authors' released L2relax (the default). The penalty is scale-sensitive, so standardisation is recommended; set False for the raw-scale variant.")
