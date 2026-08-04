"""Configuration for the GSYNTH estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class GSYNTHConfig(BaseEstimatorConfig):
    """Configuration for the generalized synthetic control (GSYNTH) estimator.

    Implements Xu, Y. (2017), *Generalized Synthetic Control Method: Causal
    Inference with Interactive Fixed Effects Models*, Political Analysis
    25(1):57-76. An interactive fixed effects model is fit to the never-treated
    units, each treated unit's factor loadings are recovered from its own
    pre-treatment periods, and the untreated potential outcome is imputed from
    the two.

    Parameters
    ----------
    covariates : list of str or None
        Time-varying covariates entering the outcome equation with a common
        coefficient vector. ``None`` fits the outcome on fixed effects and
        factors alone.
    r : int or None
        Number of unobserved factors. ``None`` selects it by the paper's
        Algorithm 1, a leave-one-period-out cross-validation over the treated
        units' pre-treatment periods.
    r_max : int
        Largest rank the cross-validation considers. The estimator lowers this
        further when the shortest treated pre-period history cannot identify
        that many loadings.
    two_way : bool
        Include additive unit and period effects alongside the factors. This is
        the specification Table 2 reports and the default everywhere in the
        paper.
    inference : bool
        Run Algorithm 2, the parametric bootstrap.
    n_bootstrap : int
        Bootstrap draws, used for both the prediction-error pool and the
        resampling loop. The paper uses 2,000.
    alpha : float
        Two-sided significance level for the percentile intervals.
    seed : int
        Seed for the bootstrap.
    tol : float
        Convergence tolerance on the coefficient vector in the alternating
        least squares that fits the control-unit model.
    max_iter : int
        Iteration cap for the same loop.
    """

    covariates: Optional[List[str]] = Field(
        default=None,
        description="Time-varying covariates with a common coefficient vector.",
    )
    r: Optional[int] = Field(
        default=None, ge=0,
        description="Number of factors; None selects it by Algorithm 1.",
    )
    r_max: int = Field(
        default=5, ge=0,
        description="Largest rank the cross-validation considers.",
    )
    two_way: bool = Field(
        default=True,
        description="Include additive unit and period effects.",
    )
    inference: bool = Field(
        default=True,
        description="Run the Algorithm 2 parametric bootstrap.",
    )
    n_bootstrap: int = Field(
        default=200, ge=1,
        description="Bootstrap draws for the prediction-error and resampling loops.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the percentile intervals.",
    )
    seed: int = Field(
        default=0, ge=0, description="Seed for the bootstrap.",
    )
    tol: float = Field(
        default=1e-5, gt=0.0,
        description="Convergence tolerance for the alternating least squares.",
    )
    max_iter: int = Field(
        default=500, ge=1,
        description="Iteration cap for the alternating least squares.",
    )

    @model_validator(mode="after")
    def check_gsynth_params(cls, values: Any) -> Any:
        if values.r is not None and values.r > values.r_max:
            raise MlsynthConfigError(
                f"r ({values.r}) must not exceed r_max ({values.r_max}); raise "
                f"r_max to fit that many factors."
            )
        if values.covariates is not None:
            if not values.covariates:
                raise MlsynthConfigError(
                    "covariates must name at least one column, or be None."
                )
            dupes = {c for c in values.covariates
                     if values.covariates.count(c) > 1}
            if dupes:
                raise MlsynthConfigError(
                    f"covariates repeats {sorted(dupes)}; a repeated column "
                    f"makes the coefficient vector unidentified."
                )
            clash = set(values.covariates) & {values.outcome, values.treat,
                                              values.unitid, values.time}
            if clash:
                raise MlsynthConfigError(
                    f"covariates may not name the outcome, treatment, unit or "
                    f"time column; got {sorted(clash)}."
                )
        return values
