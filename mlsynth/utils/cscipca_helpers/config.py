"""Configuration for the CSCIPCA estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class CSCIPCAConfig(BaseEstimatorConfig):
    """Configuration for the CSC-IPCA estimator.

    Implements Wang (2024), *"Counterfactual and Synthetic Control Method:
    Causal Inference with Instrumented Principal Component Analysis"*. CSC-IPCA
    is a factor-model counterfactual imputer whose factor loadings are a linear
    projection of observed covariates, ``Lambda_it = X_it Gamma`` (instrumented
    principal component analysis, Kelly-Pruitt-Su 2019/2020). The mapping matrix
    ``Gamma`` and the latent factors ``F_t`` are estimated by alternating least
    squares on the control units; the treated unit's mapping is re-estimated on
    its pre-treatment periods, and the counterfactual is
    ``hat Y_it(0) = X_it hat Gamma hat F_t``.

    Beyond the common fields (``df``, ``outcome``, ``treat``, ``unitid``,
    ``time``, ``display_graphs``, ``save``, colours), CSC-IPCA reads:

    Parameters
    ----------
    covariates : list of str
        Column names of the time-varying covariates that instrument the factor
        loadings. At least one is required -- this is the ``X_it`` cube the
        method is built on; with no covariates the model degenerates to plain
        PCA/IFE and one of the outcome-only factor estimators (``CFM``,
        ``FMA``) is the right tool instead.
    n_factors : int
        Number of latent common factors ``K``. Defaults to 2 (the paper's
        empirical choice for the Brexit application).
    max_iter : int
        Maximum alternating-least-squares iterations.
    tol : float
        Convergence tolerance on the max absolute change in ``Gamma`` and
        ``F`` between ALS iterations.
    alpha : float
        Two-sided significance level for the conformal confidence band
        (``0.05`` -> 95% band). The paper uses ``0.10`` for the Brexit study.
    inference : bool
        Whether to run the moving-block conformal inference (Chernozhukov et
        al. 2021) for a per-period confidence band. When ``False`` only the
        point counterfactual + ATT are returned (much faster).
    n_nulls : int
        Number of candidate null values per post-period on the conformal grid.
    null_grid_scale : float
        Half-width of the per-period conformal grid, in multiples of the
        pre-treatment fit RMSE around the point effect.
    """

    covariates: List[str] = Field(
        ...,
        min_length=1,
        description="Time-varying covariate columns that instrument the loadings.",
    )
    n_factors: int = Field(
        default=2, ge=1,
        description="Number of latent common factors K.",
    )
    max_iter: int = Field(
        default=100, ge=1,
        description="Maximum alternating-least-squares iterations.",
    )
    tol: float = Field(
        default=1e-6, gt=0.0,
        description="ALS convergence tolerance on max|Delta Gamma|, max|Delta F|.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the conformal band.",
    )
    inference: bool = Field(
        default=True,
        description="Run moving-block conformal inference for a per-period band.",
    )
    n_nulls: int = Field(
        default=201, ge=3,
        description="Candidate null values per post-period on the conformal grid.",
    )
    null_grid_scale: float = Field(
        default=6.0, gt=0.0,
        description="Conformal grid half-width in multiples of the pre-fit RMSE.",
    )

    @field_validator("covariates")
    @classmethod
    def _no_blank_covariates(cls, v: List[str]) -> List[str]:
        if any((not isinstance(c, str)) or (not c.strip()) for c in v):
            raise MlsynthConfigError(
                "covariates must be a list of non-empty column-name strings."
            )
        if len(set(v)) != len(v):
            raise MlsynthConfigError("covariates must not contain duplicates.")
        return v

    @model_validator(mode="after")
    def _check_cscipca_params(cls, values: Any) -> Any:
        if values.outcome in values.covariates:
            raise MlsynthConfigError(
                f"outcome column '{values.outcome}' must not also be listed as "
                "a covariate."
            )
        # The mapping Gamma is L x K; identifying K factors from the loadings
        # Lambda_it = X_it Gamma needs at least as many covariates as factors,
        # otherwise (X Gamma) is rank-deficient and the factor solve is singular.
        if len(values.covariates) < values.n_factors:
            raise MlsynthConfigError(
                f"CSCIPCA needs at least n_factors={values.n_factors} "
                f"covariates to identify the loadings; got "
                f"{len(values.covariates)}. Add covariates or reduce n_factors."
            )
        return values
