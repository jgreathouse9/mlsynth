"""Configuration for the TASC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class TASCConfig(BaseEstimatorConfig):
    """Configuration for the Time-Aware Synthetic Control (TASC) estimator.

    Implements the state-space model of Rho, Illick, Narasipura, Abadie, Hsu,
    and Misra (2026, arXiv:2601.03099) with EM learning (Kalman filter + RTS
    smoother on the E-step, closed-form MLE on the M-step) and Kalman-with-
    infinite-variance counterfactual inference.
    """

    d: int = Field(
        ...,
        ge=1,
        description=(
            "Hidden state dimension. Should satisfy d << min(n_donors, T) so "
            "that H X retains a low-rank signal structure."
        ),
    )
    n_em_iter: int = Field(
        default=50,
        ge=1,
        description="Number of EM iterations N1 in Algorithm 2 (EM_pre).",
    )
    em_tol: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Optional convergence tolerance on the maximum absolute change in "
            "(A, H) between successive EM iterations. If None, EM runs for the "
            "full ``n_em_iter`` iterations."
        ),
    )
    diagonal_Q: bool = Field(
        default=True,
        description=(
            "If True, the M-step constrains the state-noise covariance Q to be "
            "diagonal (the paper's default in Algorithm 7). If False, the full "
            "symmetric covariance is updated."
        ),
    )
    diagonal_R: bool = Field(
        default=True,
        description=(
            "If True, the M-step constrains the observation-noise covariance R "
            "to be diagonal (the paper's default in Algorithm 7). If False, the "
            "full symmetric covariance is updated."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description=(
            "Significance level for the posterior-based counterfactual "
            "confidence intervals (computed from h_1' P_t^s h_1)."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed used for EM initialization tie-breaks.",
    )

    @model_validator(mode="after")
    def _check_tasc_dim(self) -> "TASCConfig":
        if self.d < 1:
            raise MlsynthConfigError("'d' (hidden state dimension) must be >= 1.")
        return self
