"""Configuration for the Forward Difference-in-Differences (FDID) estimator.

Co-located with the FDID helper package. The shared
:class:`~mlsynth.config_models.BaseEstimatorConfig` remains central; only the
per-estimator config lives here. Re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class FDIDConfig(BaseEstimatorConfig):
    """
    Configuration for the Forward Difference-in-Differences (FDID) estimator.
    Inherits all common configuration parameters from BaseEstimatorConfig.

    Additional Parameters
    ---------------------
    verbose : bool, default=True
        Whether to save intermediary Forward Selection results.
    inference : {"analytic", "hac"}, default="analytic"
        Which standard error to report. ``"analytic"`` is Li (2023)
        Proposition 2.1, exact when the parallel-trends residual is serially
        uncorrelated. ``"hac"`` estimates the residual's autocovariances on
        the pre-period and prices them into the variance of the pre- and
        post-period block means, which restores coverage when the residual is
        dependent and costs nothing when it is not. See
        :mod:`mlsynth.utils.fdid_helpers.inference`.
    lrvar_lag : int, optional
        Truncation lag for ``inference="hac"``. Defaults to
        :func:`~mlsynth.utils.fdid_helpers.inference.hac_lag`, which is
        ``min(T1 - 1, T0 // 10)``.
    """

    verbose: bool = Field(
        default=True,
        description="Whether to save intermediary Forward Selection Results.",
    )
    inference: Literal["analytic", "hac"] = Field(
        default="analytic",
        description=(
            "Standard error to report: 'analytic' (Li 2023, Proposition 2.1) "
            "or 'hac' (serial-correlation robust)."
        ),
    )
    lrvar_lag: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Truncation lag for inference='hac'. Defaults to "
            "min(post_periods - 1, pre_periods // 10)."
        ),
    )

    @model_validator(mode="after")
    def _lag_requires_the_hac_method(self) -> "FDIDConfig":
        """A lag under the analytic formula would have no effect at all."""
        if self.lrvar_lag is not None and self.inference != "hac":
            raise MlsynthConfigError(
                "lrvar_lag applies only to inference='hac'; got "
                f"inference={self.inference!r}."
            )
        return self
