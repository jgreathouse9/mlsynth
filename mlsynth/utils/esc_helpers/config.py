"""Configuration for the ESC (Ensemble Synthetic Control) estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class ESCConfig(BaseEstimatorConfig):
    """Configuration for Ensemble Synthetic Control (Lian & Pu 2026).

    ESC targets short pre-treatment panels, where a single synthetic-control
    fit extrapolates unstably. It aggregates ``n_learners`` low-complexity
    LASSO base learners, each fit on a block-subsampled slice of the
    pre-period (respecting temporal dependence) crossed with a random donor
    subspace, and reads pointwise prediction intervals off the empirical
    quantiles of the ensemble counterfactuals.
    """

    n_learners: int = Field(
        default=200, ge=2,
        description="Number of perturbed LASSO base learners in the ensemble (B/m).",
    )
    block_length: int = Field(
        default=2, ge=1,
        description="Block length b for the time-block subsampling that respects "
                    "serial dependence in the pre-treatment window.",
    )
    subspace_ratio: float = Field(
        default=0.5, gt=0.0, le=1.0,
        description="Donor-subspace ratio rho: each learner sees a random "
                    "floor(rho*J) donors (model-space perturbation).",
    )
    block_keep: float = Field(
        default=0.8, gt=0.0, le=1.0,
        description="Fraction of pre-treatment blocks each learner keeps "
                    "(time-block subsample size; at least two blocks are kept).",
    )
    alpha: float = Field(
        default=0.10, gt=0.0, lt=1.0,
        description="Prediction-interval miscoverage; the pointwise band is the "
                    "[alpha/2, 1-alpha/2] ensemble quantiles (0.10 -> 90% band).",
    )
    lambda_rule: Literal["1se", "min"] = Field(
        default="1se",
        description="LASSO penalty rule from the per-fit CV path: '1se' (the "
                    "conservative one-standard-error rule, the paper's stability "
                    "default) or 'min' (CV-minimizing lambda).",
    )
    lambda_scope: Literal["local", "global"] = Field(
        default="local",
        description="Where the penalty is chosen: 'local' (per-learner CV on each "
                    "subsample, the recommended Local rule for short T0) or "
                    "'global' (one CV on the full pre-period, reused by every learner).",
    )
    aggregate: Literal["mean", "median", "trimmed"] = Field(
        default="mean",
        description="Ensemble point aggregation of the base counterfactuals "
                    "(eq. 5): 'mean', 'median', or 'trimmed' mean.",
    )
    trim: float = Field(
        default=0.10, ge=0.0, lt=0.5,
        description="Two-sided trimming fraction for aggregate='trimmed'.",
    )
    n_alphas: int = Field(
        default=30, ge=5,
        description="Length of the LASSO penalty path searched by cross-validation.",
    )
    cv_folds: int = Field(
        default=3, ge=2,
        description="Cross-validation folds for the LASSO penalty selection.",
    )
    seed: int = Field(
        default=0,
        description="RNG seed for the block/subspace perturbation draws (reproducibility).",
    )
    inference: bool = Field(
        default=True,
        description="Populate the scalar ATT interval from the ensemble ATT "
                    "quantiles (the per-period band is always produced).",
    )

    @model_validator(mode="after")
    def _check_ensemble(self) -> "ESCConfig":
        if self.aggregate == "trimmed" and self.trim <= 0.0:
            raise MlsynthConfigError(
                "aggregate='trimmed' needs trim > 0; use 'mean' for no trimming."
            )
        return self
