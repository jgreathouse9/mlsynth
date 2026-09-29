"""Configuration for the TBRMM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from ...config_models import BaseMAREXConfig
from ...exceptions import MlsynthDataError


class TBRMMConfig(BaseMAREXConfig):
    """Configuration for TBRMM - matched markets for a geo experiment (Au 2018).

    TBRMM chooses which geos to treat before the experiment runs, so the panel
    it takes carries no treatment: every period is history it scores candidate
    splits against. It inherits :class:`BaseMAREXConfig` for that reason, as
    LEXSCM and MAREX do.
    """

    max_treatment_size: int = Field(
        ...,
        gt=0,
        description="REQUIRED: the largest treatment group the advertiser will "
                    "allow, Au's K. One recommended design is returned per size "
                    "from the number of forced treatment geos (or one) up to K, "
                    "which is what the advertiser chooses between.",
    )

    n_test: int = Field(
        ...,
        gt=0,
        description="REQUIRED: how many periods the planned experiment will run. "
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

    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 indicator marking periods to exclude from "
                    "scoring, for a panel that already carries a later window. "
                    "Block assigned: constant across units within a period, and "
                    "once 1 always 1. Omitted, every period is scored.",
    )

    @field_validator("treatment_eligible_col", "control_eligible_col",
                     "unassigned_eligible_col", "post_col")
    @classmethod
    def _non_empty(cls, v):
        if v is not None and not str(v).strip():
            raise ValueError("column name must not be blank")
        return v

    @model_validator(mode="after")
    def _columns_present(self):
        named = [c for c in (self.treatment_eligible_col,
                             self.control_eligible_col,
                             self.unassigned_eligible_col, self.post_col)
                 if c is not None]
        missing = [c for c in named if c not in self.df.columns]
        if missing:
            raise MlsynthDataError(
                f"column(s) {missing} named in the configuration are absent "
                f"from the panel; it has {sorted(self.df.columns)}")
        return self
