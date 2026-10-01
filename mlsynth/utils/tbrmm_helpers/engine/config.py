"""Configuration for the TBR estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator, model_validator

from ....config_models import BaseEstimatorConfig
from ....exceptions import MlsynthDataError


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

    control_col: str = Field(
        ...,
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
        required = {self.outcome, self.unitid, self.time, self.control_col}
        for optional in (self.treat, self.post_col, self.treatment_col,
                         self.cooldown_col, self.cost_col):
            if optional is not None:
                required.add(optional)
        if self.df.empty:
            raise MlsynthDataError("Input DataFrame 'df' cannot be empty.")
        missing = required - set(self.df.columns)
        if missing:
            raise MlsynthDataError(
                f"Missing required columns in DataFrame 'df': "
                f"{', '.join(sorted(missing))}")
        blank = {c: int(self.df[c].isna().sum())
                 for c in (self.unitid, self.time) if self.df[c].isna().any()}
        if blank:
            raise MlsynthDataError(
                f"Rows with a missing unit or period are not observations: "
                f"{blank}")
        return self

    @model_validator(mode="after")
    def _one_way_to_name_the_window(self):
        """Either ``treat``, or ``post_col`` with ``treatment_col``."""
        if self.treat is None and self.post_col is None:
            raise ValueError(
                "TBR needs a post-treatment window: supply treat for an "
                "experiment that ran, or post_col together with treatment_col "
                "to design or backtest one")
        if self.treat is None and self.treatment_col is None:
            raise ValueError(
                "post_col names the window but not the groups; supply "
                "treatment_col to say which units form the treatment group, or "
                "use treat instead")
        return self
