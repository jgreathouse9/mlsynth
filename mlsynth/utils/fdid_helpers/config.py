"""Configuration for the Forward Difference-in-Differences (FDID) estimator.

Co-located with the FDID helper package. The shared
:class:`~mlsynth.config_models.BaseEstimatorConfig` remains central; only the
per-estimator config lives here. Re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from pydantic import Field

from ...config_models import BaseEstimatorConfig


class FDIDConfig(BaseEstimatorConfig):
    """
    Configuration for the Forward Difference-in-Differences (FDID) estimator.
    Inherits all common configuration parameters from BaseEstimatorConfig.

    Additional Parameters
    ---------------------
    verbose : bool, default=True
        Whether to save intermediary Forward Selection results.
    """

    verbose: bool = Field(
        default=True,
        description="Whether to save intermediary Forward Selection Results.",
    )
