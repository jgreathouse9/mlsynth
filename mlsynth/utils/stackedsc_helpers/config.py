"""Configuration for STACKEDSC (Wiltshire 2023).

Four inputs beyond the usual columns, and each corresponds to a choice the
paper makes explicitly rather than a convenience.

``agg_weights`` because the event-time average is over treated units and the
paper weights it by 1990 population, "so each tau_e is not overly-affected by
large percentage effects in small counties" (fn 28). Equal weighting is a
different estimand, not a default worth hiding.

``bias_correct`` because the paper's headline is the corrected figure, and the
correction roughly doubles it.

``covariates`` and ``backend`` because the synthetic control is fitted on
predictors, and which predictor-weight rule runs determines the answer by more
than the estimand itself on this data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig


class STACKEDSCConfig(BaseEstimatorConfig):
    """Typed configuration for :class:`mlsynth.STACKEDSC`."""

    agg_weights: Optional[str] = Field(
        default=None,
        description=(
            "Column holding the per-treated-unit aggregation weight gamma_i, "
            "constant within a unit. None gives equal weights. Wiltshire uses "
            "1990 population; the pattern of results holds under equal "
            "weighting with different magnitudes, so this changes the "
            "estimand and is worth stating rather than defaulting silently."),
    )
    normalize: bool = Field(
        default=True,
        description=(
            "Index the outcome for each treated unit and its donors to 100 at "
            "that unit's final pre-treatment period, so effects read as "
            "percent changes and are comparable across units of very "
            "different size. This is not a display choice: with weights "
            "summing to one it constrains the synthetic control to reproduce "
            "the treated unit's base-period level exactly, which is why the "
            "gap at e = -1 is identically zero. Turning it off gives a "
            "level-scale stacked SCM, which is a different estimator."),
    )
    n_leads: Optional[int] = Field(
        default=None, gt=0,
        description=(
            "Post-treatment horizons to report. None uses the largest window "
            "every treated unit can supply, which is what `balanced` does in "
            "the reference implementation."),
    )
    n_lags: Optional[int] = Field(
        default=None, gt=0,
        description=(
            "Pre-treatment horizons to report, counting back from e = -1. "
            "None uses the largest window every treated unit can supply."),
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Predictor columns matched alongside the pre-treatment outcome "
            "path. None matches on the outcome alone."),
    )
    covariate_windows: Optional[Dict[Any, Tuple[Any, Any]]] = Field(
        default=None,
        description=(
            "Per-covariate inclusive (start, end) averaging window, as "
            "{column: (start, end)}. The paper averages its ten covariates "
            "over the five years before any unit is treated."),
    )
    backend: Literal["auto", "outcome-only", "regression", "malo", "mscmt"] = (
        Field(
            default="auto",
            description=(
                "Predictor-weight rule. 'auto' uses 'outcome-only' without "
                "covariates and 'regression' with them -- the latter because "
                "Stata's `synth`, which the reference implementation wraps, "
                "runs its regression rule unless the caller asks for the "
                "nested search."),
        )
    )
    bias_correct: bool = Field(
        default=False,
        description=(
            "Apply the Abadie-L'Hour correction for residual predictor "
            "imbalance to each unit's gap before aggregating. Requires "
            "`covariates`. On the paper's data this roughly doubles the "
            "estimate, and the corrected figure is the one it reports."),
    )
    bias_correct_ridge: float = Field(
        default=1e-2, ge=0.0,
        description=(
            "Ridge penalty on the bias-correction slope; ignored unless "
            "`bias_correct`. Large values shrink the correction toward zero."),
    )
    donor_predicate: Optional[Any] = Field(
        default=None,
        description=(
            "Callable (treated_unit_id, donor_unit_id) -> bool restricting "
            "each treated unit's donor pool, mirroring the reference "
            "implementation's `donorif`. None uses every never-treated unit "
            "for every treated unit. Note that a predicate which actually "
            "binds forces a per-unit design matrix and forfeits the "
            "per-cohort batching."),
    )

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _bias_correction_needs_predictors(self):
        if self.bias_correct and not self.covariates:
            raise ValueError(
                "bias_correct=True needs covariates: the correction subtracts "
                "(X1 - X0 W)' beta from the gap, and with no predictors that "
                "term does not exist."
            )
        return self

    @model_validator(mode="after")
    def _windows_name_real_covariates(self):
        if self.covariate_windows:
            named = set(self.covariates or [])
            unknown = [c for c in self.covariate_windows if c not in named]
            if unknown:
                raise ValueError(
                    f"covariate_windows names {unknown}, which are not in "
                    f"covariates={sorted(named)}."
                )
        return self
