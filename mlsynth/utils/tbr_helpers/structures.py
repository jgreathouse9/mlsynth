"""Typed results for TBR.

The standardized sub-models carry what every effect estimator reports; the
fields below carry what is TBR's alone. A cooldown split and a cost-design rank
belong here and not on :class:`~mlsynth.config_models.MethodDetailsResults`,
which is every estimator's.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...config_models import BaseEstimatorResults
from .diagnostics import AssumptionChecks


class TBRFit(BaseModel):
    """The fitted pretest relation, Kerman et al. (2017) eqn 1."""

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(..., description="Intercept of the pretest relation.")
    beta: float = Field(..., description="Slope on the control group aggregate.")
    sigma_sq: float = Field(
        ..., description="Classical residual variance estimate; its square root "
                         "is the ``s`` of eqn 6.")
    df: int = Field(
        ..., description="Degrees of freedom of the posterior, ``n - 2`` for "
                         "``n`` pretest periods.")
    n_pretest: int = Field(..., description="Pretest periods used for the fit.")
    rank_deficient: bool = Field(
        ..., description="True when the pretest design is rank deficient, which "
                         "is section 3.4's case of a regressor that is constant "
                         "through the pretest. The fit is then the zero fit and "
                         "the counterfactual is zero with certainty.")


class CumulativeEffect(BaseModel):
    """The posterior of ``Delta(T)`` at every horizon in the window."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    estimate: List[float] = Field(
        ..., description="Median cumulative effect at each horizon (eqn 4).")
    scale: List[float] = Field(
        ..., description="Scale of the t-distribution at each horizon (eqn 6). "
                         "The scale and not the standard deviation, which is "
                         "undefined for four or fewer pretest periods.")
    lower: List[float] = Field(..., description="Lower posterior bound.")
    upper: List[float] = Field(..., description="Upper posterior bound.")
    level: float = Field(..., description="Two-sided interval level.")
    df: int = Field(..., description="Degrees of freedom of the posterior.")
    periods: List = Field(
        ..., description="Time labels of the horizons, in order.")


class PointwiseEffect(BaseModel):
    r"""The posterior of the per-period effect :math:`\phi_t`, over the panel.

    The cumulative effect answers what the campaign did by time :math:`t`; this
    answers what it did in the period. Equation 6 gives it with no new algebra:
    at a horizon of one period the cumulative effect is the per-period effect
    and :math:`\bar{x}_1` is :math:`x_t`, so the scale reduces to
    :math:`s\sqrt{v_a + 2 x_t v_{ab} + v_b x_t^2 + 1}`.

    Reported over the whole panel and not only the test window. The pretest
    half is the fitted model's residuals, which section 3.3's figure draws as a
    visual diagnostic: on a well-specified fit they centre on zero and their
    intervals cover it at about the nominal rate.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    estimate: List[float] = Field(
        ..., description="Median per-period effect, the prediction error "
                         "``y_t - (alpha + beta x_t)``.")
    scale: List[float] = Field(
        ..., description="Scale of the t-distribution at each period: eqn 6 at "
                         "a horizon of one.")
    lower: List[float] = Field(..., description="Lower posterior bound.")
    upper: List[float] = Field(..., description="Upper posterior bound.")
    level: float = Field(..., description="Two-sided interval level.")
    df: int = Field(..., description="Degrees of freedom of the posterior.")
    periods: List = Field(
        ..., description="Time labels, the whole panel, in order.")


class IROASResult(BaseModel):
    """Incremental return on ad spend, section 3.4."""

    model_config = ConfigDict(frozen=True)

    estimate: float = Field(
        ..., description="Point estimate of ``iROAS(T)``, the posterior median.")
    lower: float = Field(..., description="Lower posterior bound.")
    upper: float = Field(..., description="Upper posterior bound.")
    level: float = Field(..., description="Two-sided interval level.")
    total_incremental_cost: float = Field(
        ..., description="``Delta_cost(T)`` at the final horizon.")
    total_incremental_response: float = Field(
        ..., description="``Delta_resp(T)`` at the final horizon.")
    fixed_cost: bool = Field(
        ..., description="True when there is no spend outside the treated test "
                         "cells, so the cost counterfactual is zero with "
                         "certainty and the ratio is a rescaled t. False when "
                         "the denominator carries uncertainty and the ratio is "
                         "simulated from both posteriors.")


class TBREstimate(BaseEstimatorResults):
    """What TBR estimates once the groups are fixed.

    An :class:`~mlsynth.config_models.EffectResult`, and what ``fit()`` puts on
    ``report`` in both modes. The design half sits on the enclosing
    :class:`~mlsynth.utils.tbr_helpers.design_structures.TBRResults`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tbr_fit: Optional[TBRFit] = Field(
        default=None, description="The fitted response relation.")
    cost_fit: Optional[TBRFit] = Field(
        default=None, description="The fitted cost relation, when a cost column "
                                  "was supplied.")
    cumulative: Optional[CumulativeEffect] = Field(
        default=None, description="Posterior of the cumulative response effect.")
    cumulative_cost: Optional[CumulativeEffect] = Field(
        default=None, description="Posterior of the cumulative cost effect.")
    pointwise: Optional[PointwiseEffect] = Field(
        default=None,
        description="Posterior of the per-period response effect, over the "
                    "whole panel. The test-window estimates sum to the final "
                    "cumulative one; the pretest estimates are the fit's own "
                    "residuals.")
    iroas: Optional[IROASResult] = Field(
        default=None, description="Incremental return on ad spend; None unless a "
                                  "cost column was supplied.")

    intervention_periods: int = Field(
        default=0, description="Post-treatment periods before the cooldown "
                               "begins. Equals the whole window when no "
                               "cooldown flag was supplied.")
    cooldown_periods: int = Field(
        default=0, description="Post-treatment periods flagged as cooldown. "
                               "Zero when no cooldown flag was supplied.")
    effect_at_intervention_end: Optional[float] = Field(
        default=None, description="Cumulative effect at the last period before "
                                  "the cooldown. Section 3.5 reads this against "
                                  "the cooldown-end figure to decide whether the "
                                  "cooldown was needed.")
    effect_at_cooldown_end: Optional[float] = Field(
        default=None, description="Cumulative effect at the final horizon.")

    treated_units: List = Field(
        default_factory=list, description="Units in the treatment group.")
    control_units: List = Field(
        default_factory=list, description="Units in the control group.")
    unassigned_units: List = Field(
        default_factory=list, description="Units in neither group, which enter "
                                          "neither aggregate.")
    assumptions: Optional[AssumptionChecks] = Field(
        default=None,
        description="What the panel says about the assumptions that are "
                    "checkable from it: the pretest residuals, and the "
                    "completeness and stability of the geo set. A check that "
                    "passed is not a licence for the estimate -- the half of "
                    "the identifying assumption that reaches into the test "
                    "window cannot be tested at all.")
    filled_cells: int = Field(
        default=0, description="Unit-period cells absent from the panel and "
                               "filled with zero before aggregation. TBR sums "
                               "across geos, so an absent cell enters the total "
                               "as a zero; the count makes the assumption "
                               "visible.")
