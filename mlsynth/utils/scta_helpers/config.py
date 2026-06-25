"""Configuration for the SCTA estimator (Synthetic Control with Temporal Aggregation).

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SCTAConfig(BaseEstimatorConfig):
    """Configuration for Synthetic Control with Temporal Aggregation (SCTA).

    SCTA (Sun, Ben-Michael & Feller 2024) fits a single-treated-unit synthetic
    control that jointly balances the disaggregated high-frequency pre-period
    outcomes and their temporal aggregates (block means of ``block_length``
    consecutive periods). A single weight ``nu`` on a fixed diagonal ``V``
    trades the two off: ``nu = 0`` recovers the conventional disaggregated SC,
    larger ``nu`` shifts balance onto the aggregated series.
    """

    block_length: int = Field(
        ...,
        description="Number K of consecutive high-frequency pre-periods per temporal aggregate (e.g. 12 months per year). The pre-period length T0 must be divisible by K.",
    )
    nu: float = Field(
        default=0.5,
        description="Weight on the aggregated objective relative to the disaggregated one. Enters V as diag(K*nu on the aggregate rows, 1 on the disaggregated rows). nu=0 -> disaggregated SC; the paper's default heuristic is 0.5 with sensitivity over a range.",
    )
    augment: Optional[Literal["ridge"]] = Field(
        default=None,
        description="If 'ridge', augment the simplex SC with the bilevel ridge-augmented solver (Ben-Michael Augmented SCM). None -> plain simplex at the true optimum of the temporal-aggregation objective.",
    )
    ridge_lambda: Optional[float] = Field(
        default=None,
        description="Fixed ridge penalty for augment='ridge'; None selects it by leave-one-period-out cross-validation (augsynth's 1-SE rule).",
    )
    demean: bool = Field(
        default=True,
        description="Intercept-shift / unit fixed effect: the counterfactual is the treated unit's pre-period mean plus the weighted donor deviations (Doudchenko-Imbens; intrinsic to the SCTA estimator form).",
    )
    conformal_alpha: float = Field(
        default=0.1,
        description="Miscoverage rate for the CWZ conformal prediction interval on the headline ATT (e.g. 0.1 for a 90% interval).",
        gt=0, lt=1,
    )
    frontier: Optional[List[float]] = Field(
        default=None,
        description="Optional grid of nu values to trace the imbalance frontier (disaggregated vs aggregated pre-treatment RMSE). When given, results.frontier reports one point per nu; the headline fit still uses the scalar `nu`.",
    )

    @model_validator(mode="after")
    def _check_scta(self) -> "SCTAConfig":
        if self.block_length < 2:
            raise MlsynthConfigError(
                f"block_length must be >= 2; got {self.block_length}.")
        if self.nu < 0:
            raise MlsynthConfigError(f"nu must be >= 0; got {self.nu}.")
        if self.ridge_lambda is not None:
            if self.augment != "ridge":
                raise MlsynthConfigError(
                    "ridge_lambda is only used with augment='ridge'.")
            if self.ridge_lambda <= 0:
                raise MlsynthConfigError(
                    f"ridge_lambda must be > 0; got {self.ridge_lambda}.")
        if self.frontier is not None and any(v < 0 for v in self.frontier):
            raise MlsynthConfigError("frontier nu values must all be >= 0.")
        return self
