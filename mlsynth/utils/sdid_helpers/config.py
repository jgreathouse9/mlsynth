"""Configuration for the SDID estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SDIDConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Difference-in-Differences (SDID) estimator.

    Implements Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)'s SDID
    with the event-study aggregation of Ciccia (2024, arXiv:2407.09565).
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` panel-data interface from :class:`BaseEstimatorConfig`.

    Setting ``subgroup`` switches on the synthetic triple-difference (SC-DDD)
    mode of Zhuang (2024, arXiv:2409.12353): the outcome is demeaned by the
    non-target subgroup within each treatment-group-by-time cell, reducing the
    triple difference to a difference-in-differences that SDID then estimates.
    """

    vce: Literal["placebo", "jackknife", "bootstrap", "noinference"] = Field(
        default="placebo",
        description=(
            "Variance estimator for the ATT, following Arkhangelsky et al. "
            "(2021): 'placebo' (Algorithm 4, the default and the only method "
            "defined for a single treated unit), 'jackknife' (Algorithm 3, "
            "fixed-weights leave-one-out), 'bootstrap' (Algorithm 2, clustered "
            "resampling with weights re-fit), or 'noinference' to skip variance "
            "estimation. 'jackknife' and 'bootstrap' are implemented for the "
            "block (single adoption period) design and return NaN for a single "
            "treated unit, matching the synthdid R package."
        ),
    )
    B: int = Field(
        default=500,
        ge=0,
        description=(
            "Number of resamples for the variance estimator (placebo iterations "
            "or bootstrap replications; ignored by 'jackknife' and "
            "'noinference'). Set to 0 to skip resample-based inference. The "
            "paper uses B = 500."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed used for the placebo / bootstrap resampling.",
    )
    intercept_adjust: bool = Field(
        default=False,
        description=(
            "Whether the counterfactual exposed in ``time_series`` is shifted by "
            "the SDID intercept (the constant level difference between the treated "
            "unit and its weighted donors). When False (default), the raw "
            "weighted-donor series is reported, as in Arkhangelsky et al. (2021) "
            "Figure 1; when True it is level-matched to the treated unit over the "
            "pre-period. The point estimate and inference are unaffected either way."
        ),
    )
    subgroup: Optional[str] = Field(
        default=None,
        description=(
            "Synthetic triple-difference (SC-DDD) mode, Zhuang (2024, "
            "arXiv:2409.12353). Column naming a within-unit subgroup dimension "
            "(e.g. an age band). When set, the outcome is demeaned by the "
            "non-target subgroup within each treatment-group-by-time cell -- "
            "reducing the triple difference to a difference-in-differences -- and "
            "SDID is run on the transformed outcome over the target subgroup. "
            "Requires ``target_subgroup``. None (default) -> ordinary SDID."
        ),
    )
    target_subgroup: Optional[Any] = Field(
        default=None,
        description=(
            "SC-DDD mode: the value of the ``subgroup`` column identifying the "
            "policy-exposed (target) subgroup whose effect is estimated; all "
            "other subgroup values are the non-target controls that are demeaned "
            "out. Required when ``subgroup`` is set."
        ),
    )

    @model_validator(mode="after")
    def _check_ddd(self) -> "SDIDConfig":
        if self.subgroup is None:
            if self.target_subgroup is not None:
                raise MlsynthConfigError(
                    "target_subgroup is set but subgroup is not; provide both to "
                    "enable SC-DDD mode, or neither for ordinary SDID.")
            return self
        if self.subgroup not in self.df.columns:
            raise MlsynthConfigError(
                f"subgroup column '{self.subgroup}' is not in df.")
        if self.target_subgroup is None:
            raise MlsynthConfigError(
                "SC-DDD mode requires target_subgroup (the exposed subgroup value) "
                "when subgroup is set.")
        col = self.df[self.subgroup]
        if not (col == self.target_subgroup).any():
            raise MlsynthConfigError(
                f"target_subgroup {self.target_subgroup!r} does not appear in "
                f"column '{self.subgroup}'.")
        if (col != self.target_subgroup).sum() == 0:
            raise MlsynthConfigError(
                f"column '{self.subgroup}' has no non-target rows to demean by; "
                "SC-DDD needs at least one non-target subgroup value.")
        return self
