"""Configuration for the GSYNTH estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Literal, Optional

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
    force : {"none", "unit", "time", "two-way"}
        Which additive effects accompany the factors, named and coded as in
        gsynth and fect: ``"none"`` keeps the grand mean alone, ``"unit"`` adds
        unit effects, ``"time"`` adds period effects, ``"two-way"`` adds both.
        Two-way is the specification Xu (2017) Table 2 reports and the default
        here. An effect that is switched off is not removed from the panel, so
        the factors absorb it; the choice moves the estimate, and applied work
        varies it deliberately.
    two_way : bool or None
        Superseded by ``force``, and kept so code written against the first
        release keeps running: ``True`` resolves to ``"two-way"`` and ``False``
        to ``"time"``, with a :class:`DeprecationWarning`. Supplying both is an
        error.
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
        description=(
            "Number of factors. None selects it by Xu (2017) Algorithm 1: "
            "leave-one-pre-treatment-period-out prediction error at each "
            "candidate rank, taking a larger rank only when it beats the "
            "running minimum by more than 1%. That is the criterion gsynth "
            "implements through 1.3.x. From gsynth 1.4.0 the R package "
            "forwards to fect, whose cross-validation masks a random share of "
            "observed cells over k rounds and weights the error across "
            "relative periods, so it can choose a different rank on the same "
            "panel. Fix r to compare the estimators themselves."
        ),
    )
    r_max: int = Field(
        default=5, ge=0,
        description=(
            "Largest rank the cross-validation considers. Lowered further when "
            "the shortest treated pre-period history cannot identify that many "
            "loadings."
        ),
    )
    force: Literal["none", "unit", "time", "two-way"] = Field(
        default="two-way",
        description=(
            "Which additive effects accompany the factors, as in gsynth: none, "
            "unit, time, or two-way. Reach for a setting other than two-way "
            "when the specification you are matching used one."
        ),
    )
    two_way: Optional[bool] = Field(
        default=None,
        description=(
            "Superseded by force; True maps to two-way and False to time, with "
            "a DeprecationWarning."
        ),
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
    def resolve_force_alias(cls, values: Any) -> Any:
        if values.two_way is not None:
            if "force" in values.model_fields_set:
                raise MlsynthConfigError(
                    "force and two_way both given; two_way is the superseded "
                    "spelling, so pass force alone."
                )
            resolved = "two-way" if values.two_way else "time"
            warnings.warn(
                f"two_way is superseded by force; two_way={values.two_way} "
                f"means force={resolved!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(values, "force", resolved)
        return values

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
