"""Configuration for the SDID estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SDIDConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Difference-in-Differences (SDID) estimator.

    Implements Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)'s SDID
    with the event-study aggregation of Ciccia (2024, arXiv:2407.09565).
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` panel-data interface from :class:`BaseEstimatorConfig`.
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
