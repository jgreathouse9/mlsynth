"""Configuration for the NSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class NSCConfig(BaseEstimatorConfig):
    """Configuration for the Nonlinear Synthetic Control (NSC) estimator.

    Implements Tian (2023), *"The Synthetic Control Method with
    Nonlinear Outcomes"* (arXiv:2306.01967). The estimator generalises
    Abadie-Diamond-Hainmueller (2010) synthetic control to nonlinear
    outcome functions by

    * dropping the non-negativity restriction on donor weights,
    * adding a pairwise-distance-weighted L1 penalty plus an L2
      penalty in the weight-fitting objective, and
    * scaling the tuning parameters by the eigenvalues of
      :math:`Z_0 Z_0'` so they can be cross-validated on ``[0, 1]``.

    Parameters
    ----------
    a : float or None
        Dimensionless L1-discrepancy tuning parameter on ``[0, 1]``.
        Higher values concentrate weight on units close to the
        treated one in pretreatment matching variables (paper eq.
        (7)). ``None`` triggers coordinate-descent CV.
    b : float or None
        Dimensionless L2 tuning parameter on ``[0, 1]``. Higher
        values spread weights more evenly across donors. ``None``
        triggers coordinate-descent CV.
    cv_grid_size : float
        Step of the CV grid on ``[0, 1]``. Paper default is 0.1.
    cv_target : {"controls", "treated"}
        CV target. ``"controls"`` (paper default) leaves each donor
        out in turn and predicts it from the others; ``"treated"``
        scores on the treated unit's pretreatment fit.
    cv_max_iterations : int
        Hard cap on coordinate-descent iterations for the CV sweep.
    covariates : list of str, optional
        Optional covariate columns to use as additional matching
        variables alongside the pretreatment outcomes; collapsed to
        per-unit pretreatment means before being stacked into ``Z_0``.
    alpha : float
        Two-sided significance level for the Doudchenko-Imbens
        confidence intervals.
    run_inference : bool
        Whether to compute the Doudchenko-Imbens variance estimator
        and the per-period / ATT CIs.
    display_graphs : bool
        Whether to render the diagnostic NSC plot.
    """

    a: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Dimensionless L1-discrepancy tuning parameter; "
        "None triggers CV.",
    )
    b: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Dimensionless L2 tuning parameter; None triggers CV.",
    )
    cv_grid_size: float = Field(
        default=0.1, gt=0.0, le=0.5,
        description="CV grid step on [0, 1].",
    )
    cv_target: Literal["controls"] = Field(
        default="controls",
        description=(
            "CV target. Only the R-faithful 'controls' target is supported: "
            "for each donor, fit weights on its pre-period using the other "
            "donors and score on its held-out post-period MSPE."
        ),
    )
    cv_max_iterations: int = Field(
        default=3, ge=1, le=20,
        description="Coordinate-descent iterations for the CV sweep.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional covariate columns stacked into the matching matrix.",
    )
    standardize: bool = Field(
        default=True,
        description=(
            "Centre each matching-variable column to mean 0 and rescale to "
            "sample sd 1 (R's scale()). Default True matches the reference "
            "NSC implementation; set False only for back-compat."
        ),
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for Doudchenko-Imbens CIs.",
    )
    run_inference: bool = Field(
        default=True,
        description="Whether to run Doudchenko-Imbens variance estimation.",
    )
    seed: int = Field(
        default=123,
        description=(
            "Seed for the extra-donor draws in the R-faithful CV and "
            "leave-one-control inference loops. Matches the reference "
            "R script's set.seed(123)."
        ),
    )

    @model_validator(mode='after')
    def check_nsc_params(cls, values: Any) -> Any:
        if values.covariates is not None:
            unknown = [c for c in values.covariates if c not in values.df.columns]
            if unknown:
                raise MlsynthConfigError(
                    f"covariates references unknown columns: {unknown}"
                )
        return values
