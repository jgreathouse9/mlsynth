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
    method: str = Field(default="fs", description="Type of PDA to use: 'LASSO', 'l2', 'fs', or 'hcw' (original Hsiao-Ching-Wan best-subset).", pattern="^(LASSO|l2|fs|hcw)$")
    methods: Optional[List[str]] = Field(default=None, description="Optional list of PDA variants to run; overrides `method` when set.")
    hcw_criterion: str = Field(default="AICc", pattern="^(AICc|AIC|BIC)$", description="HCW only: model-selection criterion for the best-subset donor search ('AICc' default, matching pampe / HCW Table XVI; also 'AIC' or 'BIC').")
    hcw_nvmax: Optional[int] = Field(default=None, gt=0, description="HCW only: largest donor-subset size searched (pampe's nvmax). None searches up to all donors (bounded by the pre-period OLS df). Cap it for large donor pools, since best-subset is combinatorial.")
    hcw_backend: str = Field(default="fw", pattern="^(fw|scip)$", description="HCW only: best-subset search engine. 'fw' (default) is the exact Furnival-Wilson branch-and-bound, which certifies the optimum for small pools and otherwise returns the best incumbent with an optimality gap. 'scip' uses the optional SCIP mixed-integer solver (requires pyscipopt) to certify the optimum at larger pool sizes.")
    tau: Optional[float] = Field(default=None, description="User-specified treatment effect value (used as tau_l2 for 'l2' method).")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Significance level for confidence intervals and ATE inference.")
    fs_intercept: bool = Field(default=False, description="Forward-selection only: include a constant in the donor regression. False (default) matches Shi & Huang's simulation (no intercept, valid size on mean-zero factor data); True matches the released fsPDA R package (intercept, for panels with genuine level differences).")
    lrvar_lag: Optional[int] = Field(default=None, description="Forward-selection only: Bartlett-kernel truncation lag for the Newey-West long-run variance in the post-selection t-test. None (default) uses the released fsPDA package's rule floor(T2**(1/4)); a supplied value must be a non-negative integer no larger than floor(sqrt(T2)).")
    l2_standardize: bool = Field(default=True, description="L2-relaxation only: standardise (demean + unit-variance scale) the treated and control series before solving, matching the authors' released L2relax (the default). The penalty is scale-sensitive, so standardisation is recommended; set False for the raw-scale variant.")
    prediction_intervals: bool = Field(default=False, description="Attach Jiang, Li, Shen & Zhou (2025) bootstrap prediction intervals for the per-period treatment effect and counterfactual to every fitted PDA variant. Equal-tailed and symmetric intervals are returned; each variant reports whether the post-selection OLS HAC sandwich studentization was used or the sigma^2-only fallback (e.g. for the dense L2-relaxation in high dimensions).")
    pi_n_boot: int = Field(default=999, ge=2, description="Number of bootstrap replications for the prediction intervals (only used when prediction_intervals is True).")
    pi_seed: Optional[int] = Field(default=0, description="Seed for the prediction-interval bootstrap RNG (reproducible by default; set None for a fresh draw).")
