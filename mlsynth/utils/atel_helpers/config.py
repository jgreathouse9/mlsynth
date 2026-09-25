"""Configuration for the ATEL estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class ATELConfig(BaseEstimatorConfig):
    """Configuration for Average Treatment Effect Localization (ATEL).

    Implements Lee, R.-C. (2026), *"Average Treatment Effect Localization:
    Projection Methods in Synthetic Control"*, Econometric Theory. ATEL reports
    a kernel-weighted average effect localized at the adoption date, so early
    post-periods count for more than late ones. The counterfactual comes from a
    time-varying factor model whose factors are estimated by diversified
    projection and whose loading is fit by local linear regression.

    Parameters
    ----------
    covariates : list of str
        Time-varying covariate columns. The sieve basis in these columns builds
        the diversified weights, so at least one is required and there is no
        outcomes-only fallback.
    n_factors : int
        Number of factors, which is also the sieve width. Required: the
        information criterion in the reference implementation returns an
        endpoint of its candidate grid on the paper's own panel, so there is no
        data-driven default here. Must be at least 2.

        With ``P`` covariates the projection consumes the first ``n_factors`` of
        the ``n_factors * P`` weight blocks, and that set is a complete set of
        (basis, covariate) pairs only when ``n_factors`` is a multiple of ``P``.
        Otherwise the estimate depends on the order the covariates are named:
        on the paper's Arizona panel, swapping two covariates moves the estimate
        by 22.6 at ``n_factors = 3`` and by 71.4 at ``n_factors = 5``, against
        standard errors near 13. Such a configuration is refused.
    basis : {"bspline", "trigonometric", "polynomial"}
        Sieve basis. ``"bspline"`` is the paper's recommendation.
    bandwidth : float or None
        Smoothing bandwidth as a fraction of the sample. ``None`` selects one by
        leave-one-out cross-validation over 0.30 to 0.95 in steps of 0.05, and
        warns when the criterion lands on an endpoint of that grid.
    alpha : float
        Two-sided significance level for the interval on the localized estimate.
    """

    covariates: List[str] = Field(
        ...,
        min_length=1,
        description="Time-varying covariate columns building the sieve weights.",
    )
    n_factors: int = Field(
        ...,
        ge=2,
        description="Number of factors, and the sieve width. At least 2.",
    )
    basis: Literal["bspline", "trigonometric", "polynomial"] = Field(
        default="bspline",
        description="Sieve basis for the diversified weights.",
    )
    bandwidth: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Bandwidth as a fraction of the sample; None cross-validates.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Two-sided significance level for the interval.",
    )

    @model_validator(mode="after")
    def check_atel_params(self) -> Any:
        missing = [c for c in self.covariates if c not in self.df.columns]
        if missing:
            raise MlsynthConfigError(
                f"Covariate column(s) {missing} are not in the panel."
            )
        n_cov = len(self.covariates)
        if n_cov > 1 and self.n_factors % n_cov != 0:
            raise MlsynthConfigError(
                f"n_factors={self.n_factors} is not a multiple of the "
                f"{n_cov} covariates. The projection would take the sieve's "
                f"basis {self.n_factors // n_cov + 1} for some covariates and "
                f"not others, which makes the estimate depend on the order "
                f"the covariates are named. Choose a multiple of {n_cov}."
            )
        return self
