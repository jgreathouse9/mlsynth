"""Typed results for TBRMM.

A design is what the advertiser chooses between, so each recommended pair
carries its own groups, its own score and the climb that produced it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ...config_models import DesignResult


class TBRMMPosterior(BaseModel):
    """TBR's posterior for a measured effect (Kerman, Wang and Vaver 2017).

    Under the paper's flat prior on :math:`(\alpha, \beta, \log\sigma)` the
    cumulative effect :math:`\Delta(T)` has a t posterior on :math:`n-2` degrees
    of freedom whose scale is their eqn 6,

    .. math::

       T s \sqrt{v_a + 2\bar{x}_T v_{ab} + v_b \bar{x}_T^2 + 1/T},

    which is algebraically the OLS prediction-error standard deviation, so the
    interval reads as a credible interval or a prediction interval alike.

    The cumulative effect is the primitive. The ATT is that divided by a known
    constant, so ``att_lower`` and ``att_upper`` are ``total_lower`` and
    ``total_upper`` over the same constant and ``df`` does not move. The constant
    is the geo count times the period count for a group, and the period count
    alone for a single market.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale: float = Field(
        ..., description="Scale of the cumulative effect's t posterior: eqn 6. "
                         "Its two terms grow as T^2 (the shared coefficient "
                         "uncertainty) and T (the test window's own noise), so "
                         "it is not recoverable by adding per-period variances.")
    df: int = Field(
        ..., description="Degrees of freedom: pretest periods minus two.")
    level: float = Field(
        ..., description="Two-sided interval level the bounds were cut at.")
    total_lower: float = Field(
        ..., description="Lower bound on the cumulative effect.")
    total_upper: float = Field(
        ..., description="Upper bound on the cumulative effect.")
    group_lower: float = Field(
        default=0.0,
        description="The total averaged over the test periods: the cumulative "
                    "effect on the treatment group per period. This is the "
                    "scale report.inference is on, so an interval and the att "
                    "beside it agree. total = group * n_periods.")
    group_upper: float = Field(
        default=0.0,
        description="Upper bound of the per-period group effect.")
    att_lower: float = Field(
        ..., description="Lower bound on the mean per-period effect, the "
                         "cumulative bound rescaled.")
    att_upper: float = Field(
        ..., description="Upper bound on the mean per-period effect.")
    prob_direction: float = Field(
        ..., description="Posterior mass on the side of zero the point estimate "
                         "sits on; 0.5 when the posterior straddles zero evenly "
                         "and approaches 1 as the effect separates from it.")
    variance: Literal["iid", "hac"] = Field(
        default="iid",
        description="Which variance the scale came from. 'iid' is equation 6 as "
                    "published, whose noise term is T sigma^2. 'hac' is Li and "
                    "Van den Bulte's Proposition 3.4, which replaces both terms "
                    "with Newey-West truncated sums and so prices serially "
                    "correlated residuals.")
    bandwidth: Optional[int] = Field(
        default=None,
        description="Newey-West truncation lag behind a 'hac' scale, and None "
                    "under 'iid'. Zero keeps only the diagonal, which is the "
                    "independent case.")


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
    posterior: Optional["TBRMMPosterior"] = Field(
        default=None,
        description="TBR's posterior for this geo's effect, fitted on its own "
                    "pretest regression. Wider than the group's, which is the "
                    "precision given up by reading one geo instead of the sum.")
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
    delta1: float = Field(
        default=0.0,
        description="Intercept of the group's own augmented DiD fit, on the "
                    "summed treated series.")
    delta2: float = Field(
        default=0.0,
        description="Free control scale of the group's fit. It is the sum of the "
                    "implied donor weights, so its distance from one is how far "
                    "the design sits from a convex average of the controls.")
    n_post: int = Field(..., description="Post-window length in periods.")
    n_treated: int = Field(..., description="Treated geos measured.")
    posterior: Optional["TBRMMPosterior"] = Field(
        default=None,
        description="TBR's posterior for the group's effect, fitted on the "
                    "summed treated series. Not assembled from the market "
                    "posteriors: treated geos co-move, so their residuals are "
                    "correlated and only a regression spanning them prices that "
                    "correlation.")
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


class TBRResults(DesignResult):
    """What :meth:`mlsynth.TBR.fit` returns, in both modes.

    A :class:`~mlsynth.config_models.DesignResult`. ``report`` carries the
    estimate as a :class:`~mlsynth.utils.tbr_helpers.structures.TBREstimate`,
    built by the same code in both modes. The design fields below are populated
    only when the split was searched for; a named split has nothing to choose
    between, so ``designs`` is empty and ``recommended`` is ``None``.
    """

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


#: The pre-merge name. TBRMM was the design half of TBR and is now its searched
#: mode; this alias keeps the structures module importable under either name
#: while the rest of the merge lands.
TBRMMResults = TBRResults
