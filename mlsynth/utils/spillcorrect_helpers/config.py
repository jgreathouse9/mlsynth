"""Configuration for SPILLCORRECT.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseMAREXConfig, DesignResult
from ...exceptions import MlsynthConfigError

#: Keys of ``design_config`` the restricted re-solve sets itself.
SET_BY_THE_CORRECTION = ("df", "outcome", "unitid", "time", "to_be_treated",
                         "not_to_be_treated", "adjacency", "exclude_bordering_donors")


class SPILLCORRECTConfig(BaseMAREXConfig):
    """Configuration for correcting known contaminated controls in a design.

    ``df`` is the realized panel: the design's units over its fitting, blank
    and post windows, long format, sorted by unit and time. The design is
    read through its typed faces (``windows``, ``composite``,
    ``inference_settings``), so it is not re-solved, except for the restricted
    comparison.
    """

    design: DesignResult = Field(
        ..., description="The fitted design, e.g. a MAREX result. Its windows and "
                         "composite faces must be filled.")
    affected_units: List[Any] = Field(
        ..., description="Control units known to carry a spillover from the "
                         "treatment. Those with control weight are corrected; "
                         "those without carry no bias and are kept out of the donors.")
    method: Literal["iterative", "inclusive", "restricted"] = Field(
        default="iterative",
        description="'iterative' (Melnychuk 2024): rebuild each contaminated "
                    "control from the clean controls. 'inclusive' (Di Stefano and "
                    "Mellace 2024): solve the cross-weight system for the effect "
                    "and every spillover; it can be singular. 'restricted': report "
                    "the design re-solved without the affected controls, which "
                    "needs design_config.")
    scale: bool = Field(
        default=True,
        description="Fit each rebuilding synthetic control on series divided by "
                    "their means over the fitting window, so a control larger than "
                    "every clean one can be rebuilt. Turn it off only for outcomes "
                    "that are not proportional to a unit's size.")
    window: Literal["blank_and_post", "post"] = Field(
        default="blank_and_post",
        description="'blank_and_post': fit on the design's fitting window and "
                    "correct its blank and post periods, so the placebo periods "
                    "that calibrate the band and the test carry the correction's "
                    "own error. 'post': fit on the whole pre-period and correct "
                    "the post periods only, as the published methods do.")
    global_test: Optional[Literal["subsets", "cyclic"]] = Field(
        default=None,
        description="The global test of the recomputed inference. None keeps the "
                    "design's.")
    design_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The MAREX configuration the design was fit with, without df, "
                    "outcome, unitid and time. When given, the design is re-solved "
                    "with the affected controls out of its control pool and the "
                    "restricted estimate is reported beside the correction.")

    @model_validator(mode="after")
    def _check_design(self) -> "SPILLCORRECTConfig":
        d = self.design
        if d.windows is None or d.composite is None:
            raise MlsynthConfigError(
                "SPILLCORRECT: the design carries no windows or composite face; "
                "it needs a design estimator that fills them (MAREX does)."
            )
        members = d.composite.members or (d.composite.treated_labels
                                          + d.composite.control_labels)
        labels = [str(u) for u in self.affected_units]
        if not labels:
            raise MlsynthConfigError("SPILLCORRECT: affected_units needs at least one unit.")
        twice = sorted({u for u in labels if labels.count(u) > 1})
        if twice:
            raise MlsynthConfigError(f"SPILLCORRECT: affected_units lists {twice} twice.")
        unknown = [u for u in labels if u not in members]
        if unknown:
            raise MlsynthConfigError(
                f"SPILLCORRECT: affected_units {unknown} are not among the design's units.")
        treated = [u for u in labels if u in d.composite.treated]
        if treated:
            raise MlsynthConfigError(
                f"SPILLCORRECT: affected_units {treated} are treated in the design; "
                "a spillover reaches controls, and a treated unit's outcome is the "
                "effect being measured."
            )
        if self.method == "restricted" and self.design_config is None:
            raise MlsynthConfigError(
                "SPILLCORRECT: method='restricted' re-solves the design, which needs "
                "design_config, the configuration it was fit with."
            )
        if self.design_config is not None:
            clash = [k for k in SET_BY_THE_CORRECTION if k in self.design_config]
            if clash:
                raise MlsynthConfigError(
                    f"SPILLCORRECT: design_config sets {clash}, which the restricted "
                    "re-solve takes from this configuration."
                )
        if self.global_test is not None and d.inference_settings is None:
            raise MlsynthConfigError(
                "SPILLCORRECT: global_test changes the design's inference, and this "
                "design was fit without inference."
            )
        return self
