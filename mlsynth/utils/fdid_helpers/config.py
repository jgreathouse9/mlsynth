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
            "or 'hac' (serial-correlation robust). Left unset, a single "
            "treated unit takes 'analytic' and a staggered panel takes 'hac': "
            "the staggered aggregate averages over event horizons, so an "
            "unpriced autocorrelation divides the variance by the horizon "
            "count as though the horizons were independent draws. Setting it "
            "explicitly is honoured on both paths."
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

    selection: Literal["unit", "pooled", "partial"] = Field(
        default="unit",
        description=(
            "Whose pre-treatment fit forward selection optimises when the "
            "panel has several treated units: each unit's own ('unit', Li's "
            "Web Appendix C), its cohort's mean ('pooled'), or a convex "
            "combination ('partial'). Ignored with one treated unit."
        ),
    )
    pooling_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Weight on the cohort criterion under selection='partial'. Zero "
            "reproduces 'unit', one reproduces 'pooled'. Defaults to 0.5."
        ),
    )
    anticipation: int = Field(
        default=0,
        ge=0,
        description=(
            "Pre-periods dropped from the end of each treated unit's "
            "pre-window, for effects that begin before the recorded adoption "
            "date. Shortens the window selection and the intercept both use."
        ),
    )
    max_horizon: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Highest event time reported on a staggered panel. Defaults to "
            "the largest horizon every cohort supports, keeping the event "
            "study balanced."
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

    @model_validator(mode="after")
    def _pooling_weight_requires_partial_selection(self) -> "FDIDConfig":
        """Under 'unit' or 'pooled' the weight is fixed, so setting it is a
        silent no-op the caller would read as having taken effect."""
        if self.pooling_weight is not None and self.selection != "partial":
            raise MlsynthConfigError(
                "pooling_weight applies only to selection='partial'; got "
                f"selection={self.selection!r}."
            )
        return self

    @property
    def resolved_pooling_weight(self) -> float:
        """The weight actually used, with the default filled in."""
        return 0.5 if self.pooling_weight is None else float(self.pooling_weight)
