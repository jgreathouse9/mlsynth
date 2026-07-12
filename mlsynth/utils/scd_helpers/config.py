"""Configuration for the SCD estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SCDConfig(BaseEstimatorConfig):
    """Configuration for Synthetic Control with Differencing (SCD).

    SCD (Rincon & Song 2026, arXiv:2510.26106; repeated-cross-section
    inference from Canen & Song 2025) fits a simplex synthetic control on
    *within-group-differenced* survey-weighted group means, and reports a
    per-period effect path with confidence bands built from individual
    influence functions (a :math:`\\sqrt{n}`, fixed-``T`` asymptotic).

    Unlike the aggregate-panel estimators, SCD expects grouped microdata:
    each ``(unit, time)`` cell carries many individual observations, supplied
    as one row per individual, optionally with a survey weight. Treatment is
    still applied at the unit-time level via the ``treat`` column.

    Parameters
    ----------
    differencing : {"did", "uniform", "sc"}
        Within-group differencing scheme (the "D" in SCD). ``"did"``
        (default) differences every group series off its last pre-period
        mean; ``"uniform"`` differences off the pre-period average;
        ``"sc"`` applies no differencing (classical synthetic control on
        levels).
    weight_col : str, optional
        Column with individual survey weights used to form the group means.
        If ``None`` the group means are unweighted (equal weight per row).
    compute_inference : bool
        If ``True`` (default), compute the repeated-cross-section variance
        and the weight confidence set, producing per-period confidence
        bands. If ``False``, only the point estimator runs.
    alpha : float
        Overall (one minus coverage) level for the confidence bands.
        Default ``0.10`` (90% bands).
    kappa : float
        Bonferroni split allotted to the weight confidence set; the
        pointwise term uses the remaining ``alpha - kappa``. Must satisfy
        ``0 < kappa < alpha``. Default ``0.05``.
    n_grid : int
        Number of candidate weight vectors sampled per family (a Gaussian
        cloud around the fitted weights and a Dirichlet cloud) when sweeping
        the weight confidence set. Default ``3000``.
    grid_radius : float
        Standard deviation of the Gaussian cloud around the fitted weights.
        Default ``0.05``.
    tolerance : float
        Numerical tolerance for the confidence-set active-constraint count.
        Default ``1e-6``.
    random_state : int
        Seed for the confidence-set grid sampler. Default ``0``.
    """

    differencing: Literal["did", "uniform", "sc"] = Field(
        default="did",
        description="Within-group differencing scheme: 'did' (off last pre-period), "
                    "'uniform' (off pre-period average), or 'sc' (no differencing).",
    )
    weight_col: Optional[str] = Field(
        default=None,
        description="Column with individual survey weights; unweighted if None.",
    )
    compute_inference: bool = Field(
        default=True,
        description="Compute the repeated-cross-section variance and weight "
                    "confidence set, producing per-period confidence bands.",
    )
    alpha: float = Field(
        default=0.10, gt=0.0, lt=1.0,
        description="Overall level (one minus coverage) for the confidence bands.",
    )
    kappa: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Bonferroni share for the weight confidence set (0 < kappa < alpha).",
    )
    n_grid: int = Field(
        default=3000, ge=2,
        description="Candidate weight vectors sampled per family for the confidence-set sweep.",
    )
    grid_radius: float = Field(
        default=0.05, gt=0.0,
        description="Std of the Gaussian cloud around the fitted weights.",
    )
    tolerance: float = Field(
        default=1e-6, gt=0.0,
        description="Numerical tolerance for the confidence-set active-constraint count.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for the confidence-set grid sampler.",
    )

    @model_validator(mode="after")
    def _check_bonferroni_split(self):
        if not (self.kappa < self.alpha):
            raise MlsynthConfigError(
                f"SCD requires 0 < kappa < alpha; got kappa={self.kappa}, "
                f"alpha={self.alpha}."
            )
        return self
