"""Configuration for the SHC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

import numbers

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

    reference_pool: str = Field(
        default="block_oos",
        description=(
            "Which pre-period residuals calibrate the conformal test. "
            "'block_oos' (default) refits each historical block against the "
            "blocks that share no observation with it and takes the residual "
            "over its own post-window, so the reference residuals are the same "
            "object as the tested statistic. 'smoother' is the paper's literal "
            "reading, y_t minus the fitted latent trend; it is retained for "
            "reproducing published numbers and it over-rejects."
        ),
    )
    reference_stride: Optional[int] = Field(
        default=None,
        description=(
            "With reference_pool='block_oos', evaluate every stride-th block. "
            "Each block costs one matching solve. None (the default) picks the "
            "smallest stride keeping the count at or under 60 blocks, so the "
            "cost does not grow with the panel; 1 evaluates every block, which "
            "is what the 1% level needs and what a few hundred blocks charges "
            "a few tens of seconds for."
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
            # numbers.Real: numpy scalars are numbers too (issue #320).
            if not all(isinstance(h, numbers.Real) and not isinstance(h, bool)
                       for h in self.bandwidth_grid):
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
        if self.reference_pool not in ("block_oos", "smoother"):
            raise MlsynthConfigError(
                "'reference_pool' must be 'block_oos' or 'smoother'."
            )
        if self.reference_stride is not None and self.reference_stride < 1:
            raise MlsynthConfigError("'reference_stride' must be >= 1.")

        return self
