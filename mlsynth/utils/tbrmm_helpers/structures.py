"""Typed results for TBRMM.

A design is what the advertiser chooses between, so each recommended pair
carries its own groups, its own score and the climb that produced it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...config_models import DesignResult


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
    matching_trace: List[float] = Field(
        default_factory=list,
        description="The objective at each accepted step of the matching climb "
                    "for this treatment size, oldest first. Non-decreasing by "
                    "construction, which is what makes the search a hill climb.")
    matching_converged: bool = Field(
        default=False,
        description="True when the climb stopped because no single toggle of "
                    "control-group membership improved the objective, which is "
                    "Algorithm 1's stopping rule.")
    n_candidates_evaluated: int = Field(
        default=0,
        description="Objective evaluations spent reaching this design.")


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
