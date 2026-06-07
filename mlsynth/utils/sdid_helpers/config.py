"""Configuration for the SDID estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SDIDConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Difference-in-Differences (SDID) estimator.

    Implements Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)'s SDID
    with the event-study aggregation of Ciccia (2024, arXiv:2407.09565).
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` panel-data interface from :class:`BaseEstimatorConfig`.
    """

    B: int = Field(
        default=500,
        ge=0,
        description=(
            "Number of placebo iterations for the variance estimator. "
            "Set to 0 to skip placebo inference (att_se / p_value will be NaN). "
            "The paper uses B = 500."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed used for the placebo resampling.",
    )
