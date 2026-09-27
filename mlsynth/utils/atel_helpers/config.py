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
    weight_source : {"covariates", "initial", "hadamard"}
        Where the diversified weights come from, following the constructions
        Fan and Liao (2022) Section 4 recommends.

        ``"covariates"`` (default) is their Section 4.1: a sieve basis in the
        observed time-varying covariates, which is what the ATEL paper uses and
        what requires ``covariates``.

        ``"initial"`` is their Section 4.3: a sieve basis in each unit's first
        observed outcome, ``w_ik = phi_k(x_i0)``, which correlates with the
        loadings through ``x_0 = B f_0 + u_0``. The first period is held out of
        the estimation sample, since the weights are built from it. No
        covariates are needed.

        ``"hadamard"`` is their Section 4.4: deterministic sign columns, which
        satisfy the independence requirement by construction and use no data at
        all. No covariates are needed.

        Their Section 4.2, weights from trimmed principal-component loadings on
        an earlier sample split, is not offered: it needs a split and serial
        independence of the errors, which is a different assumption burden.
    covariates : list of str
        Time-varying covariate columns. Required, and at least one, when
        ``weight_source`` is ``"covariates"``, since the sieve basis in these
        columns is what builds the weights. Refused for the other sources,
        which cannot use them.
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

    weight_source: Literal["covariates", "initial", "hadamard"] = Field(
        default="covariates",
        description=(
            "Construction for the diversified weights: a sieve in the "
            "covariates (Fan-Liao 4.1), in each unit's first outcome "
            "(4.3), or deterministic sign columns (4.4)."
        ),
    )
    covariates: List[str] = Field(
        default_factory=list,
        description=(
            "Time-varying covariate columns building the sieve weights; "
            "required for weight_source='covariates' and refused otherwise."
        ),
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
        if self.weight_source == "covariates":
            if not self.covariates:
                raise MlsynthConfigError(
                    "weight_source='covariates' builds the diversified weights "
                    "from a sieve basis in the covariates, so at least one "
                    "covariate column is required. For a panel without "
                    "covariates use weight_source='initial' or 'hadamard'."
                )
        elif self.covariates:
            raise MlsynthConfigError(
                f"weight_source={self.weight_source!r} does not read covariates, "
                f"but {list(self.covariates)} were supplied. Drop them, or use "
                "weight_source='covariates' to build the weights from them."
            )

        missing = [c for c in self.covariates if c not in self.df.columns]
        if missing:
            raise MlsynthConfigError(
                f"Covariate column(s) {missing} are not in the panel."
            )
        n_cov = len(self.covariates)
        # The block slice only creates an ordering to permute when the weights
        # come from more than one covariate; the other sources emit one series
        # per basis function, so any factor count is well defined.
        if n_cov > 1 and self.n_factors % n_cov != 0:
            raise MlsynthConfigError(
                f"n_factors={self.n_factors} is not a multiple of the "
                f"{n_cov} covariates. The projection would take the sieve's "
                f"basis {self.n_factors // n_cov + 1} for some covariates and "
                f"not others, which makes the estimate depend on the order "
                f"the covariates are named. Choose a multiple of {n_cov}."
            )
        return self
