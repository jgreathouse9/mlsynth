"""Configuration for the TBR estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError, MlsynthDataError


#: The constraint fields whose default is None; the boolean and the threshold
#: are read separately.
_CONSTRAINT_FIELDS = (
    "to_be_treated", "not_to_be_treated", "size_col", "min_size", "max_size",
    "cluster_col", "adjacency", "stratum_col", "min_per_stratum",
    "max_per_stratum", "market_cost_col", "max_total_cost", "min_controls",
    "max_controls")


class TBRConfig(BaseEstimatorConfig):
    """Configuration for TBR - Time-Based Regression (Kerman et al. 2017).

    TBR aggregates geos into a treatment and a control group and regresses one
    group's series on the other over the pretest period. A geo experiment has
    three groups and not two, so ``control_col`` is always required: a geo may
    be held out of both.

    Two ways to name the window and the groups, because designing an experiment
    and estimating from one are separate jobs, as they are for LEXSCM and
    SYNDES.

    Estimation. ``treat`` names the treated units and the period they were
    treated in, which is the ordinary case of an experiment that ran.

    Design. ``post_col`` names the window and ``treatment_col`` the group, with
    no ``treat`` at all. This is how a candidate split is scored before anything
    is treated, and how an A/A test is run on history where the truth is zero.
    Au (2018) needs both: his search fits TBR on pretest data under a
    hypothetical split, and his Example 3 measures TBR's error rate on the
    control geos of a finished experiment.

    Supply ``treat``, or ``post_col`` together with ``treatment_col``.
    """

    treat: Optional[str] = Field(
        default=None,
        description="Column marking treated unit-periods (0/1), naming both the "
                    "treated units and when treatment began. Omit it for design "
                    "mode and supply post_col and treatment_col instead.",
    )

    post_col: Optional[str] = Field(
        default=None,
        description="Design mode: 0/1 indicator for the post-treatment window. "
                    "Block assigned, so constant across units within a period "
                    "and once 1 always 1. Supplied with treatment_col in place "
                    "of treat, it lets TBR run on a panel where nothing was "
                    "treated, which is what scoring a candidate design and "
                    "running an A/A test both need.",
    )

    treatment_col: Optional[str] = Field(
        default=None,
        description="Design mode: column marking units in the treatment group "
                    "(boolean or 0/1, constant within unit). Required when "
                    "treat is omitted; ignored when treat is supplied, which "
                    "already names the treated units.",
    )

    control_col: Optional[str] = Field(
        default=None,
        description="REQUIRED: column marking units in the control group "
                    "(boolean or 0/1, constant within unit). Units that are "
                    "neither treated nor flagged here are unassigned and enter "
                    "neither aggregate, which is how a geo is held out of the "
                    "experiment without being dropped from the panel.",
    )

    cooldown_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 indicator for the cooldown period, the window "
                    "after the intervention stops over which its effect is "
                    "still accumulating. Block assigned: the flag is a property "
                    "of the period, so it is constant across units within a "
                    "period, and once it turns 1 it stays 1. Supplying it "
                    "extends the window the cumulative posterior covers to the "
                    "end of the cooldown, which is the reference "
                    "implementation's default. Omitted, the window ends where "
                    "the cooldown would have begun.",
    )

    cost_col: Optional[str] = Field(
        default=None,
        description="Optional per-unit spend column. Supplying it computes the "
                    "incremental return on ad spend, Section 3.4's "
                    "iROAS(T) = Delta_resp(T) / Delta_cost(T), as a second "
                    "estimand. Omitted, only the response effect is reported.",
    )

    level: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Two-sided posterior interval level for the cumulative "
                    "effect. 0.9 is the paper's own reporting level.",
    )

    # ---- searched mode: the design half (Au 2018 section 3)
    max_treatment_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Searched mode: the largest treatment group the advertiser will "
                    "allow, Au's K. One recommended design is returned per size "
                    "from the number of forced treatment geos (or one) up to K, "
                    "which is what the advertiser chooses between.",
    )

    n_test: Optional[int] = Field(
        default=None,
        gt=0,
        description="Searched mode: how many periods the planned experiment will run. "
                    "This enters the objective through the power term, since the "
                    "smallest detectable impact depends on the test's length, and "
                    "through the A/A test's window.",
    )

    objective: Literal["reference", "paper", "relative"] = Field(
        default="reference",
        description="Which objective the hill climb maximises. 'reference' "
                    "(default) is the four assumption gates, then the group "
                    "correlation, then the inverse of the smallest detectable "
                    "impact -- the score Google's implementation uses. 'paper' is "
                    "Au section 3.1's stated min(CUSUM p, Breusch-Godfrey p, "
                    "R-squared). The default is not the paper's literal text "
                    "because that text carries no power term while its own prose "
                    "asks for one: over 400 random splits of the GeoLift markets "
                    "the two rank designs at Spearman +0.54 with no overlap in "
                    "their top five, and the reference's ten best are 2.5 times "
                    "better on detectable impact. Reach for 'paper' to reproduce "
                    "the paper. 'relative' drops the gates and the correlation "
                    "and ranks designs by relative MDE, the smallest detectable "
                    "cumulative effect as a share of the treated group's level "
                    "over the test (the key is that level over the required "
                    "impact). Reach for it when the treatment groups on offer "
                    "differ in size and a lift is judged as a percentage: the "
                    "reference's power term is in outcome units and favours "
                    "groups with low sales. In scratch measurements over 300 "
                    "random treatment groups per window the correlation gate "
                    "never bound (every winner had correlation of at least "
                    "0.97), and ranking by relative MDE picked groups with 5 "
                    "times (monthly beer shipments by state) and about 10 times "
                    "(simulated panels) the sales, with intervals 1.6 to 3.2 "
                    "times narrower as a share of sales and the same calibrated "
                    "coverage; on GeoLift the picks were similar, and the "
                    "calibrated coverage was 67% against the reference's 80% "
                    "over 84 overlapping windows. Every treatment-eligible geo "
                    "needs a positive mean on the scoring window.",
    )

    control_start: Literal["carried", "pool", "best"] = Field(
        default="carried",
        description="Where the control group search starts at each treatment "
                    "size. 'carried' (default) hands the previous size's matched "
                    "control group to the next size, which is Algorithm 1 and "
                    "what Google's implementation does, so it is what the "
                    "benchmark reproduces. 'pool' re-derives the starting group "
                    "from every control-eligible geo the treatment group leaves "
                    "free. Matching is a single-toggle climb and its answer "
                    "depends on where it starts: on the GeoLift panel the same "
                    "treatment group lands on a different control group from a "
                    "different start in 16 of 19 cases. 'pool' selects different "
                    "markets, so a design chosen under it does not match the "
                    "reference. Neither start dominates: over twelve simulated "
                    "panels at three sizes each, 'pool' wins six of the thirty "
                    "six comparisons, 'carried' wins seven and the rest tie. "
                    "'best' runs both and keeps whichever design scores higher "
                    "at each size, which costs about two and a half times the "
                    "default and is never worse than it, since a tie returns the "
                    "reference walk's design.",
    )

    search: Literal["greedy", "local"] = Field(
        default="greedy",
        description="Searched mode: what runs after Algorithm 1. 'greedy' "
                    "(default) returns the hill climb's designs, which is what "
                    "Google's implementation returns and what the benchmark "
                    "reproduces. 'local' then descends from each size's greedy "
                    "design and from local_restarts random designs of the same "
                    "size, moving one control geo in, out or for another, or one "
                    "treated geo for another, and keeps a move only when the "
                    "direct fit confirms it improves the objective, so a design is "
                    "never worse than the greedy's. On the GeoLift markets it "
                    "shrinks the smallest detectable impact by 17% at one treated "
                    "geo. Requires objective='reference' or 'relative'.",
    )

    local_restarts: int = Field(
        default=10,
        ge=0,
        description="Searched mode with search='local': random designs per "
                    "treatment size the local search also descends from, besides "
                    "the greedy's.",
    )

    local_seed: int = Field(
        default=0,
        description="Searched mode with search='local': seed for the random "
                    "restarts, so a design can be reproduced. The restarts at "
                    "treatment size k draw from local_seed + k, so each size's "
                    "draws do not depend on how many sizes are searched.",
    )

    treatment_eligible_col: Optional[str] = Field(
        default=None,
        description="Column marking geos the advertiser allows in the treatment "
                    "group (boolean or 0/1, constant within unit). Omitted, every "
                    "geo is eligible. Together with the other two columns this is "
                    "Au's A_i: a geo eligible for treatment alone is forced into "
                    "the treatment group and counts toward k0.",
    )

    control_eligible_col: Optional[str] = Field(
        default=None,
        description="Column marking geos the advertiser allows in the control "
                    "group (boolean or 0/1, constant within unit). Omitted, every "
                    "geo is eligible.",
    )

    unassigned_eligible_col: Optional[str] = Field(
        default=None,
        description="Column marking geos the advertiser allows to be held out of "
                    "the experiment (boolean or 0/1, constant within unit). "
                    "Omitted, every geo may be held out. A geo ineligible here "
                    "has to end up in one group or the other.",
    )

    # ---- searched mode: business constraints, the vocabulary SYNDES, MAREX
    # and LEXSCM share (#1046). Forced, forbidden and out-of-band geos are
    # folded into each geo's eligibility; the rest bind the groups as a whole.
    to_be_treated: Optional[List[Any]] = Field(
        default=None,
        description="Searched mode: geos forced into every treatment group. A "
                    "forced geo is eligible for treatment alone, so it counts "
                    "toward k0 like a geo treatment_eligible_col alone admits. "
                    "At most max_treatment_size geos, disjoint from "
                    "not_to_be_treated.",
    )

    not_to_be_treated: Optional[List[Any]] = Field(
        default=None,
        description="Searched mode: geos kept out of every treatment group (a "
                    "regulated market, one already in another test). They lose "
                    "the treatment role and stay control candidates.",
    )

    size_col: Optional[str] = Field(
        default=None,
        description="Searched mode: per-geo constant size column. Geos outside "
                    "[min_size, max_size] lose treatment eligibility and stay "
                    "control candidates. Requires min_size or max_size.",
    )

    min_size: Optional[float] = Field(
        default=None,
        description="Searched mode: lower bound of the treatment size band, "
                    "inclusive. Requires size_col.",
    )

    max_size: Optional[float] = Field(
        default=None,
        description="Searched mode: upper bound of the treatment size band, "
                    "inclusive. Requires size_col.",
    )

    cluster_col: Optional[str] = Field(
        default=None,
        description="Searched mode: per-geo constant column; two geos sharing a "
                    "value conflict and are never both treated, because a "
                    "campaign in one reaches the other. Combined with adjacency "
                    "by logical OR.",
    )

    adjacency: Optional[pd.DataFrame] = Field(
        default=None,
        description="Searched mode: square spillover matrix indexed and columned "
                    "by geo label. Off-diagonal entries strictly above "
                    "spillover_threshold mark a conflicting pair, never both "
                    "treated. Combined with cluster_col by logical OR.",
    )

    spillover_threshold: float = Field(
        default=0.0,
        description="Searched mode: adjacency entries strictly above this mark a "
                    "conflicting pair. Requires adjacency.",
    )

    stratum_col: Optional[str] = Field(
        default=None,
        description="Searched mode: per-geo constant column defining strata for "
                    "coverage quotas on the treatment group via min_per_stratum "
                    "and max_per_stratum.",
    )

    min_per_stratum: Optional[int] = Field(
        default=None,
        description="Searched mode: the fewest treated geos in every stratum that "
                    "has a treatment-eligible geo. Requires stratum_col. A size "
                    "too small to meet every stratum's minimum has no design and "
                    "is not reported.",
    )

    max_per_stratum: Optional[int] = Field(
        default=None,
        description="Searched mode: the most treated geos in any stratum. "
                    "Requires stratum_col.",
    )

    market_cost_col: Optional[str] = Field(
        default=None,
        description="Searched mode: per-geo constant column holding each geo's "
                    "cost of being treated (the media held back, the fees to run "
                    "it). With max_total_cost, the treated geos' costs sum to at "
                    "most the budget. Not cost_col, which is the per-period spend "
                    "series behind the iROAS estimate.",
    )

    max_total_cost: Optional[float] = Field(
        default=None,
        description="Searched mode: the budget on the treated geos' total "
                    "market_cost_col. A positive number; requires "
                    "market_cost_col. Unrelated to cost_col, which feeds iROAS.",
    )

    exclude_bordering_donors: bool = Field(
        default=False,
        description="Searched mode: keep a treated geo's conflict neighbours "
                    "(from cluster_col / adjacency) out of the control group, so "
                    "a geo partly treated by spillover never sits in the "
                    "counterfactual. Requires cluster_col or adjacency.",
    )

    min_controls: Optional[int] = Field(
        default=None,
        description="Searched mode: the smallest control group a design may "
                    "have, at least 1.",
    )

    max_controls: Optional[int] = Field(
        default=None,
        description="Searched mode: the largest control group a design may "
                    "have, at least 1 and not below min_controls.",
    )

    variance: Literal["iid", "hac"] = Field(
        default="iid",
        description="Variance behind a measured effect's interval. 'iid' is "
                    "equation 6 as published. Reach for 'hac' when the pretest "
                    "residuals are serially correlated, which weekly or daily "
                    "sales usually are: it swaps both terms for Li and Van den "
                    "Bulte's Proposition 3.4 Newey-West sums, so the interval "
                    "prices the dependence instead of assuming it away. Read "
                    "for only when ``post_col`` marks realized periods.",
    )

    hac_bandwidth: Optional[int] = Field(
        default=None,
        ge=0,
        description="Newey-West truncation lag under ``variance='hac'``. The "
                    "default follows the paper, ceil(T_pre ** 0.25). Zero keeps "
                    "only the diagonal, which reproduces equation 6's "
                    "test-window error term exactly and leaves a "
                    "heteroskedasticity-robust sandwich in place of its "
                    "coefficient term, so the scale is not equation 6's unless "
                    "the residual variance is constant across the pretest.",
    )

    inference: Literal["posterior", "calibrated"] = Field(
        default="posterior",
        description="How the cumulative effect's interval is priced. "
                    "'posterior' (default) is the paper's conjugate posterior, "
                    "the interval the replication against google/matched_markets "
                    "pins. 'calibrated' keeps the same point estimate and "
                    "replaces only the interval by a Dirichlet-process "
                    "predictive calibrated on honest blank windows: the "
                    "regression's standardised errors on the last "
                    "blank_periods pre-periods, pooled over the real design and "
                    "placebo designs and trended to the experiment's distance "
                    "from the design period. In searched mode the design search "
                    "then runs on the pre-period minus those periods. Reach for "
                    "it when the residuals are serially correlated or the "
                    "design was searched for, where the posterior covers well "
                    "below its level.",
    )

    blank_periods: Optional[int] = Field(
        default=None,
        description="Calibrated inference: B, the last pre-periods held back "
                    "from the design search and used as honest blank windows. "
                    "Defaults to three test lengths, the test length being the "
                    "realised post window, or n_test in a design-only run. "
                    "Refused below two test lengths.",
    )

    n_placebos: int = Field(
        default=24,
        description="Calibrated inference: placebo treatment groups whose blank "
                    "errors are pooled with the real design's, at least 8. "
                    "When the eligibility and design rules admit fewer, all of "
                    "them are used and the result records how many.",
    )

    calibration_prior_weight: float = Field(
        default=2.0,
        description="Calibrated inference: n0, the Dirichlet-process prior's "
                    "weight on the posterior against the pooled blank errors, "
                    "in units of independent blank windows. Zero uses the blank "
                    "errors alone; a large value returns the posterior band.",
    )

    calibration_seed: int = Field(
        default=0,
        description="Calibrated inference: seed for the placebo draws, so a "
                    "band can be reproduced. Non-negative.",
    )

    @field_validator("treat", "control_col", "cooldown_col", "cost_col",
                     "post_col", "treatment_col")
    @classmethod
    def _non_empty(cls, v):
        if v is not None and not str(v).strip():
            raise ValueError("column name must not be blank")
        return v

    @model_validator(mode="after")
    def check_df_and_columns(self):
        """Shadow the base column check, which requires ``treat`` to be a name.

        Design mode has no ``treat``, so the base's required-column set cannot
        be reused verbatim. The same checks run here over the columns TBR
        actually needs, which keeps the shared validator untouched for the
        other configs that inherit it.
        """
        required = {self.outcome, self.unitid, self.time}
        for optional in (self.treat, self.post_col, self.treatment_col,
                         self.control_col, self.cooldown_col, self.cost_col,
                         self.treatment_eligible_col, self.control_eligible_col,
                         self.unassigned_eligible_col, self.size_col,
                         self.cluster_col, self.stratum_col,
                         self.market_cost_col):
            if optional is not None:
                required.add(optional)
        if self.df.empty:
            raise MlsynthDataError("Input DataFrame 'df' cannot be empty.")
        missing = required - set(self.df.columns)
        if missing:
            raise MlsynthDataError(
                f"Missing required columns in DataFrame 'df': "
                f"{', '.join(sorted(missing))}")
        # The base's column check is shadowed here, and this invariant came
        # with it. TBR sums across geos to form each aggregate, so a repeated
        # (unit, period) cell has no one reading -- a duplicate row and two
        # records meant to be added are the same input -- and the ambiguity has
        # to be refused before anything reshapes.
        repeated = int(self.df.duplicated(subset=[self.unitid, self.time]).sum())
        if repeated:
            raise MlsynthDataError(
                f"the panel repeats {repeated} (unit, period) cell(s): duplicate "
                f"rows. TBR sums across geos to form each aggregate, so a "
                f"repeated cell has no one reading and the intended total "
                f"cannot be guessed.")

        blank = {c: int(self.df[c].isna().sum())
                 for c in (self.unitid, self.time) if self.df[c].isna().any()}
        if blank:
            raise MlsynthDataError(
                f"Rows with a missing unit or period are not observations: "
                f"{blank}")
        return self

    @property
    def design_constraint_fields(self) -> List[str]:
        """The business-constraint fields set away from their defaults."""
        out = [name for name in _CONSTRAINT_FIELDS if getattr(self, name) is not None]
        if self.exclude_bordering_donors:
            out.append("exclude_bordering_donors")
        if self.spillover_threshold != 0.0:
            out.append("spillover_threshold")
        return out

    @property
    def mode(self) -> str:
        """``"named"`` when the split is given, ``"searched"`` when it is found."""
        return "searched" if self.max_treatment_size is not None else "named"

    @model_validator(mode="after")
    def _one_way_to_name_the_groups(self):
        """The groups are either given or searched for, and never both.

        TBR regresses the treatment aggregate on the control aggregate, so it
        needs two groups. They arrive one of two ways: named by
        ``treatment_col`` and ``control_col``, or found by the hill climb, which
        ``max_treatment_size`` turns on. The eligibility flags only narrow the
        pools the climb may draw from and default to every geo, so they cannot
        tell the modes apart.
        """
        named = self.treatment_col is not None or self.control_col is not None \
            or self.treat is not None
        searched = self.max_treatment_size is not None

        if named and searched:
            raise MlsynthConfigError(
                "the groups are named and searched for at the same time: "
                "treatment_col/control_col fix a split, max_treatment_size asks "
                "for one to be found. Supply one or the other."
            )
        if not named and not searched:
            raise MlsynthConfigError(
                "no groups to regress: supply treatment_col and control_col for "
                "a split you already have, or max_treatment_size and n_test to "
                "search for one."
            )

        local_settings = {"local_restarts", "local_seed"} & self.model_fields_set
        if self.search != "local" and local_settings:
            raise MlsynthConfigError(
                f"{sorted(local_settings)} tune the local search, which runs only "
                f"with search='local'; set it, or drop them.")
        if self.search == "local" and not searched:
            raise MlsynthConfigError(
                "search='local' improves a searched design and a named split has "
                "nothing to search; drop it, or drop treatment_col/control_col to "
                "design instead.")
        if self.search == "local" and self.objective == "paper":
            raise MlsynthConfigError(
                "search='local' scores its moves with the reference or relative "
                "objective, whose terms are quadratic forms in the group "
                "memberships; the paper's p-values are not, so use "
                "objective='reference' or 'relative'.")

        if searched:
            if self.n_test is None:
                raise MlsynthConfigError(
                    "a searched design needs n_test, how many periods the "
                    "planned experiment runs: the objective's power term is a "
                    "function of the test's length."
                )
            return self

        if self.treat is None:
            if self.post_col is None:
                raise MlsynthConfigError(
                    "a named split needs a post-treatment window: supply treat "
                    "for an experiment that ran, or post_col to backtest one."
                )
            if self.treatment_col is None:
                raise MlsynthConfigError(
                    "post_col names the window but not the groups; supply "
                    "treatment_col to say which units form the treatment group, "
                    "or use treat instead."
                )
        if self.control_col is None:
            raise MlsynthConfigError(
                "a named split needs control_col: TBR regresses the treatment "
                "aggregate on the control aggregate, and half a split is not a "
                "split."
            )
        if self.n_test is not None:
            raise MlsynthConfigError(
                "n_test sizes a search and a named split has nothing to search; "
                "drop it, or drop treatment_col/control_col to design instead."
            )
        return self

    @model_validator(mode="after")
    def _calibration_settings(self):
        """Refuse a calibration setting that is idle or out of range."""
        tuning = {"blank_periods", "n_placebos", "calibration_prior_weight",
                  "calibration_seed"} & self.model_fields_set
        if self.inference != "calibrated":
            if tuning:
                raise MlsynthConfigError(
                    f"{sorted(tuning)} tune calibrated inference, which runs "
                    f"only with inference='calibrated'; set it, or drop them.")
            return self
        if self.variance == "hac":
            raise MlsynthConfigError(
                "variance='hac' and inference='calibrated' both answer serially "
                "correlated residuals, and the calibration standardises by the "
                "iid scale; choose one.")
        if self.n_placebos < 8:
            raise MlsynthConfigError(
                f"n_placebos must be at least 8; got {self.n_placebos}.")
        n0 = self.calibration_prior_weight
        if not (math.isfinite(n0) and n0 >= 0.0):
            raise MlsynthConfigError(
                f"calibration_prior_weight must be finite and non-negative; "
                f"got {n0!r}.")
        if self.calibration_seed < 0:
            raise MlsynthConfigError(
                f"calibration_seed must be non-negative; got "
                f"{self.calibration_seed}.")
        B = self.blank_periods
        if B is not None and B < 2:
            raise MlsynthConfigError(
                f"blank_periods must be at least 2, two periods of a one-period "
                f"test; got {B}.")
        # A design-only search has no realised window, so its test length is
        # n_test and the rule is decidable here. Otherwise the fit checks it.
        if B is not None and self.mode == "searched" and self.post_col is None \
                and B < 2 * self.n_test:
            raise MlsynthConfigError(
                f"blank_periods is {B} and n_test is {self.n_test}; the blank "
                f"window has to be at least 2 x the test length.")
        return self

    @model_validator(mode="after")
    def _design_constraints(self):
        """Refuse a contradictory or idle constraint before anything is read.

        Every check here is decidable from the config and the per-geo columns
        alone. Whether some design satisfies the constraints together depends
        on the eligibility and on K, and the search's audit answers that.
        """
        active = self.design_constraint_fields
        if not active:
            return self
        if self.mode == "named":
            raise MlsynthConfigError(
                f"{sorted(active)} constrain the design search, and a named "
                f"split has nothing to search; drop them, or drop "
                f"treatment_col/control_col and set max_treatment_size to design "
                f"under them.")

        df, unit = self.df, self.unitid
        geos = set(df[unit].unique())
        forced = list(self.to_be_treated or [])
        forbidden = list(self.not_to_be_treated or [])
        unknown = sorted(map(str, {g for g in forced + forbidden if g not in geos}))
        if unknown:
            raise MlsynthConfigError(
                f"to_be_treated/not_to_be_treated name geos not in the panel: "
                f"{unknown}.")
        clash = sorted(map(str, set(forced) & set(forbidden)))
        if clash:
            raise MlsynthConfigError(
                f"geos cannot be both to_be_treated and not_to_be_treated: {clash}.")
        if len(set(forced)) > self.max_treatment_size:
            raise MlsynthConfigError(
                f"to_be_treated forces {len(set(forced))} geos into every "
                f"treatment group but max_treatment_size is "
                f"{self.max_treatment_size}.")

        # The size band.
        bounds = (self.min_size, self.max_size)
        if self.size_col is None and bounds != (None, None):
            raise MlsynthConfigError("min_size / max_size require size_col.")
        if self.size_col is not None and bounds == (None, None):
            raise MlsynthConfigError(
                "size_col bounds nothing without min_size or max_size.")
        if None not in bounds and self.min_size > self.max_size:
            raise MlsynthConfigError(
                f"min_size ({self.min_size}) is above max_size ({self.max_size}).")
        out_of_band = set()
        if self.size_col is not None:
            size = _per_geo(df, unit, self.size_col)
            lo = -math.inf if self.min_size is None else self.min_size
            hi = math.inf if self.max_size is None else self.max_size
            out_of_band = {g for g, v in size.items() if not lo <= v <= hi}
            outside = sorted(map(str, set(forced) & out_of_band))
            if outside:
                raise MlsynthConfigError(
                    f"to_be_treated geos {outside} fall outside the size band "
                    f"[{self.min_size}, {self.max_size}], which bars them from "
                    f"treatment.")

        # Stratum quotas.
        quota = (self.min_per_stratum, self.max_per_stratum)
        if self.stratum_col is None and quota != (None, None):
            raise MlsynthConfigError(
                "min_per_stratum / max_per_stratum require stratum_col.")
        if self.stratum_col is not None and quota == (None, None):
            raise MlsynthConfigError(
                "stratum_col sets no quota without min_per_stratum or "
                "max_per_stratum.")
        for name, value in (("min_per_stratum", self.min_per_stratum),
                            ("max_per_stratum", self.max_per_stratum)):
            if value is not None and value < 1:
                raise MlsynthConfigError(f"{name} must be at least 1; got {value}.")
        if None not in quota and self.min_per_stratum > self.max_per_stratum:
            raise MlsynthConfigError(
                f"min_per_stratum ({self.min_per_stratum}) is above "
                f"max_per_stratum ({self.max_per_stratum}).")
        if self.min_per_stratum is not None:
            stratum = _per_geo(df, unit, self.stratum_col)
            barred = set(forbidden) | out_of_band
            eligible: dict = {}
            for g, s in stratum.items():
                eligible[s] = eligible.get(s, 0) + (g not in barred)
            short = sorted(str(s) for s, n in eligible.items()
                           if 0 < n < self.min_per_stratum)
            if short:
                raise MlsynthConfigError(
                    f"min_per_stratum is {self.min_per_stratum} but stratum(s) "
                    f"{short} have fewer geos left eligible for treatment.")

        # Conflicts.
        if "spillover_threshold" in active and self.adjacency is None:
            raise MlsynthConfigError(
                "spillover_threshold reads adjacency; supply adjacency or drop it.")
        if self.exclude_bordering_donors and self.cluster_col is None \
                and self.adjacency is None:
            raise MlsynthConfigError(
                "exclude_bordering_donors keeps a treated geo's neighbours out of "
                "the control group and needs cluster_col or adjacency to say who "
                "they are.")

        # The budget.
        col, cap = self.market_cost_col, self.max_total_cost
        if col is not None and cap is None:
            raise MlsynthConfigError(
                "market_cost_col needs max_total_cost, the budget the costs are "
                "summed against.")
        if cap is not None and col is None:
            raise MlsynthConfigError(
                "max_total_cost needs market_cost_col, the column of each geo's "
                "cost.")
        if cap is not None:
            if not (math.isfinite(cap) and cap > 0):
                raise MlsynthConfigError(
                    f"max_total_cost must be a positive number; got {cap!r}.")
            cost = np.asarray(list(_per_geo(df, unit, col).values()), dtype=float)
            if not (np.all(np.isfinite(cost)) and np.all(cost >= 0)):
                raise MlsynthDataError(
                    f"market_cost_col {col!r} must hold a finite, non-negative "
                    f"cost for every geo.")

        # The control band.
        for name, value in (("min_controls", self.min_controls),
                            ("max_controls", self.max_controls)):
            if value is not None and value < 1:
                raise MlsynthConfigError(f"{name} must be at least 1; got {value}.")
        if self.min_controls is not None and self.max_controls is not None \
                and self.min_controls > self.max_controls:
            raise MlsynthConfigError(
                f"min_controls ({self.min_controls}) is above max_controls "
                f"({self.max_controls}).")
        return self


def _per_geo(df: pd.DataFrame, unit: str, col: str) -> dict:
    """A per-geo constant column as ``{geo: value}``, refused if it varies."""
    grouped = df.groupby(unit, observed=True)[col]
    varying = grouped.nunique(dropna=False)
    if (varying > 1).any():
        raise MlsynthDataError(
            f"{col!r} must be constant within each geo; it varies for "
            f"{list(varying[varying > 1].index)[:5]}.")
    return dict(grouped.first().items())
