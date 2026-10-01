"""Typed results for TBRMM.

A design is what the advertiser chooses between, so each recommended pair
carries its own groups, its own score and the climb that produced it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ...config_models import DesignResult


class TBRMMMarketEffect(BaseModel):
    """One treated geo's realized effect, from its own augmented DiD fit."""

    model_config = ConfigDict(frozen=True)

    unit: Any = Field(..., description="The treated geo.")
    att: float = Field(
        ..., description="Mean per-period effect over the post window: the "
                         "average of this geo's outcome minus its counterfactual.")
    att_percent: Optional[float] = Field(
        default=None,
        description="`att` as a percent of this geo's mean counterfactual over "
                    "the post window. None when that baseline is ~0.")
    total_effect: float = Field(
        ..., description="Summed effect over the post window for this geo.")
    delta1: float = Field(
        ..., description="Fitted intercept of equation (2.4) for this geo.")
    delta2: float = Field(
        ..., description="Fitted control scale. Forcing it to 1 would reduce the "
                         "estimator to difference-in-differences, so its distance "
                         "from 1 is how much the augmentation did.")
    rmse_fit: float = Field(
        ..., description="Pretest residual RMSE of this geo's regression. The "
                         "precision cost of reading one geo instead of the group "
                         "shows up here.")


class TBRMMEffect(BaseModel):
    """A design's realized effect: the pooled number and its market-level parts."""

    model_config = ConfigDict(frozen=True)

    att: float = Field(
        ..., description="Pooled mean per-period effect. This is the average of "
                         "`market_effects`, following Li and Van den Bulte's "
                         "Appendix C, so the headline and the breakdown agree by "
                         "construction.")
    att_percent: Optional[float] = Field(
        default=None,
        description="`att` as a percent of the treated markets' mean "
                    "counterfactual over the post window.")
    total_effect: float = Field(
        ..., description="Summed effect across every treated geo and post period "
                         "-- the program's incremental total, so this is a sum of "
                         "`market_effects` totals and not their average.")
    n_post: int = Field(..., description="Post-window length in periods.")
    n_treated: int = Field(..., description="Treated geos measured.")
    market_effects: List[TBRMMMarketEffect] = Field(
        default_factory=list,
        description="One entry per treated geo, in the design's treatment order.")


class TBRMMDesign(BaseModel):
    """One recommended treatment group and its matching control group."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(..., description="Size of the treatment group.")
    treatment_units: List[Any] = Field(
        ..., description="Geos to treat, in the order the climb added them.")
    control_units: List[Any] = Field(
        ..., description="The matching control group, sorted.")
    unassigned_units: List[Any] = Field(
        ..., description="Geos in neither group, sorted. These are held out of "
                         "the experiment and enter neither aggregate.")
    objective_value: float = Field(
        ..., description="The objective at this design. Under 'reference' this is "
                         "the inverse of the smallest detectable impact, the last "
                         "element of the score; the gates are in `detail`.")
    detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="The objective's components: the three terms under 'paper', "
                    "the four gates plus the correlation and the required impact "
                    "under 'reference'.")
    matching_trace: List[Tuple[float, ...]] = Field(
        default_factory=list,
        description="The score at each accepted step of the matching climb for "
                    "this treatment size, oldest first, as the tuple the search "
                    "compares. Non-decreasing by construction, which is what "
                    "makes the search a hill climb. The tuple and not "
                    "`objective_value`, because under 'reference' the climb "
                    "maximises a lexicographic key and its last element can fall "
                    "on a step that gains a gate.")
    matching_converged: bool = Field(
        default=False,
        description="True when the climb stopped because no single toggle of "
                    "control-group membership improved the objective, which is "
                    "Algorithm 1's stopping rule.")
    n_candidates_evaluated: int = Field(
        default=0,
        description="Objective evaluations spent reaching this design.")
    effect: Optional[TBRMMEffect] = Field(
        default=None,
        description="This design's realized effect, present only when the panel "
                    "carried a post window. Every candidate is measured, not only "
                    "the recommendation, so the menu can be compared after the "
                    "fact as well as before it.")


class TBRMMResults(DesignResult):
    """TBRMM's :class:`~mlsynth.config_models.DesignResult`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    designs: List[TBRMMDesign] = Field(
        default_factory=list,
        description="One recommended pair per treatment size, smallest first.")
    recommended: Optional[TBRMMDesign] = Field(
        default=None,
        description="The design with the best objective across sizes. A larger "
                    "treatment group is not automatically better, so this is the "
                    "argmax and not the last entry.")
    objective: str = Field(
        default="reference",
        description="Which objective was climbed.")
    forced_treatment_units: List[Any] = Field(
        default_factory=list,
        description="Geos eligible for treatment alone, which every design "
                    "contains. Au's k0 is this list's length.")
    eligible_treatment_units: List[Any] = Field(
        default_factory=list,
        description="Geos the advertiser allows in the treatment group.")
    eligible_control_units: List[Any] = Field(
        default_factory=list,
        description="Geos the advertiser allows in the control group.")
    n_periods_scored: int = Field(
        default=0,
        description="Periods the objective was computed over.")
