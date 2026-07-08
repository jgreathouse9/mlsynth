"""Configuration for the PPSCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union
from pydantic import Field, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class PPSCMConfig(BaseEstimatorConfig):
    """Configuration for the Partially Pooled SCM (PPSCM) estimator.

    Implements Ben-Michael, Feller & Rothstein (2022, *JRSS-B*
    84(2):351-381). Targets staggered-adoption designs by minimizing a
    weighted average of the per-treated-unit imbalance ``q_sep`` and
    the average-treated imbalance ``q_pool``, with weighting hyper-
    parameter ``nu``.
    """

    nu: Union[float, Literal["auto"]] = Field(
        default="auto",
        description=(
            "Pooling parameter. Small nu approaches a separate SCM per treated "
            "unit, large nu a fully pooled SCM (nu weights the pooled balance "
            "term). 'auto' (default) uses the triangle-inequality ratio "
            "global_l2 * sqrt(d) / avg_l2 of the separate fit, matching "
            "augsynth's heuristic."
        ),
    )
    fixedeff: bool = Field(
        default=True,
        description=(
            "Include two-way fixed effects (time effect from never-treated "
            "units + per-cohort unit pre-mean) and balance the residuals, as "
            "in augsynth (force=3). False removes only the control time means."
        ),
    )
    n_leads: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of post-treatment horizons (relative time) to estimate. "
            "None defaults to the number of post-treatment periods of the last "
            "treated unit."
        ),
    )
    n_lags: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of pre-treatment periods to balance. None balances all "
            "pre-treatment periods."
        ),
    )
    time_cohort: bool = Field(
        default=False,
        description=(
            "If True, collapse units sharing an adoption time into one "
            "fully-pooled cohort (one synthetic control per cohort)."
        ),
    )
    lam: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization on the donor weights.",
    )
    solver: Any = Field(
        default=None,
        description="CVXPY solver. None falls back to OSQP.",
    )
    run_inference: bool = Field(
        default=True,
        description=(
            "Whether to run inference (see ``inference_method``); refits or "
            "reweights the estimator, can be slow for the jackknife."
        ),
    )
    inference_method: str = Field(
        default="jackknife",
        description=(
            "Inference procedure: 'jackknife' (delete-one, refit per unit) or "
            "'bootstrap' (augsynth's default Mammen wild/multiplier bootstrap; "
            "reweights the single fit, no refit). The augsynth multisynth "
            "vignette prints the bootstrap SEs."
        ),
    )
    n_boot: int = Field(
        default=1000, ge=1,
        description="Bootstrap replications when ``inference_method='bootstrap'``.",
    )
    seed: int = Field(
        default=0, description="RNG seed for the bootstrap multipliers.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Wald confidence band.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Auxiliary covariates to balance alongside the pre-treatment "
            "outcomes (augsynth::multisynth Sec 5.2). Each covariate is "
            "z-scored against the never-treated controls and rescaled to the "
            "control-outcome standard deviation, so covariate and outcome "
            "imbalance share a scale; the covariate imbalance is then stacked "
            "into the pooled and separate QP terms. Time-varying covariates "
            "are aggregated to their mean over periods before the first "
            "adoption. None (default) balances outcomes only."
        ),
    )

    @model_validator(mode="after")
    def _check_inference_method(self):
        if self.inference_method not in ("jackknife", "bootstrap"):
            raise MlsynthConfigError(
                "inference_method must be 'jackknife' or 'bootstrap'; got "
                f"{self.inference_method!r}.")
        return self
