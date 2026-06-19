"""Configuration for the SHC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class SHCConfig(BaseEstimatorConfig):
    m: int = Field(default=1, description="Length of the evaluation window.")
    bandwidth_grid: Optional[List[float]] = Field(default=None, description="Bandwidth grid for LOOCV.")
    use_augmented: bool = Field(default=False, description="Use Augmented SHC (ASHC) variant.")
    inference_method: str = Field(
        default="bootstrap",
        description=(
            "Conformal inference variant: 'bootstrap' (Chen-Yang-Yang 2024 "
            "with-replacement residual resampling, the default) or 'exact' "
            "(Chernozhukov-Wuthrich-Zhu 2021 permutation test)."
        ),
    )
    permutation_scheme: str = Field(
        default="moving_block",
        description=(
            "Permutation family for inference_method='exact': 'moving_block' "
            "(cyclic shifts, for stationary weakly-dependent errors) or 'iid' "
            "(random permutations, exact under exchangeability)."
        ),
    )
    num_permutations: Optional[int] = Field(
        default=None,
        description=(
            "Permutation count for inference_method='exact' with "
            "permutation_scheme='iid' (>= 2). Ignored for 'moving_block' "
            "(always T). Defaults to 1000 for 'iid'."
        ),
    )

    @model_validator(mode="after")
    def check_shc_params(self) -> "SHCConfig":
        if not isinstance(self.use_augmented, bool):
            raise MlsynthConfigError("'use_augmented' must be a boolean.")

        if self.m <= 0:
            raise MlsynthConfigError("'m' must be a positive integer.")

        if self.bandwidth_grid is not None:
            if not self.bandwidth_grid:
                raise MlsynthConfigError("'bandwidth_grid' cannot be an empty list.")
            if not all(isinstance(h, (int, float)) for h in self.bandwidth_grid):
                raise MlsynthConfigError("All elements in 'bandwidth_grid' must be numeric.")
            if not all(h > 0 for h in self.bandwidth_grid):
                raise MlsynthConfigError("All bandwidth values must be strictly positive.")

        if self.inference_method not in ("bootstrap", "exact"):
            raise MlsynthConfigError(
                "'inference_method' must be 'bootstrap' or 'exact'."
            )
        if self.permutation_scheme not in ("moving_block", "iid"):
            raise MlsynthConfigError(
                "'permutation_scheme' must be 'moving_block' or 'iid'."
            )
        if self.num_permutations is not None and self.num_permutations < 2:
            raise MlsynthConfigError("'num_permutations' must be >= 2.")

        return self
