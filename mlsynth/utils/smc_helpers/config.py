"""Configuration for the SMC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SMCConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Matching Control estimator (Zhu 2023).

    SMC matches each donor to the treated unit by a univariate regression, then
    synthesises the matched controls with box-``[0, 1]`` weights chosen by a
    Mallows/Cp unbiased-risk criterion. The combined donor coefficients may be
    negative (controlled extrapolation), and the Cp penalty -- not a predictor
    (``V``) search -- identifies the weights, so the estimator is deterministic.

    References
    ----------
    Zhu, Rong J. B. (2023). *Synthetic Matching Control Method.*
    arXiv:2306.02584.
    """

    ridge: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Tikhonov stabiliser added to the Cp Gram so the box QP is strictly "
            "convex (the reference's 0.001*I). Not part of the paper's criterion; "
            "keep small. Larger values shrink the synthesis weights toward zero."
        ),
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional predictor columns (Algorithm 3). Each covariate's "
            "pre-treatment mean is standardised to the outcome rows' scale and "
            "added as one extra matching row. Predictor weights are held equal "
            "(V = I): the Cp penalty does the identification, so no -- inherently "
            "non-reproducible -- V search is run."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> "SMCConfig":
        if self.covariates is not None and len(self.covariates) == 0:
            raise MlsynthConfigError(
                "SMC 'covariates' must be a non-empty list or None."
            )
        return self
