"""Configuration for the TBR estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError, MlsynthDataError


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

    objective: Literal["reference", "paper"] = Field(
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
                    "the paper.",
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
                    "only the diagonal and so reproduces the iid scale.",
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
                         self.unassigned_eligible_col):
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
