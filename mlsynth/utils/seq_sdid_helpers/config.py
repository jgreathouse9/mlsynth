"""Configuration for the SequentialSDID estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SequentialSDIDConfig(BaseEstimatorConfig):
    """Configuration for the Sequential Synthetic Difference-in-Differences estimator.

    Implements Arkhangelsky & Samkov (2025, arXiv:2404.00164v2). Operates on
    cohort-level aggregates and is robust to violations of parallel trends
    induced by interactive fixed effects. Inherits the standard ``df`` /
    ``outcome`` / ``treat`` / ``unitid`` / ``time`` panel interface from
    :class:`BaseEstimatorConfig`.
    """

    eta: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Non-negative regularization for the two QPs. Larger values shrink "
            "the unit weights toward ``omega_j proportional to pi_j`` (the "
            "stacked-DiD imputation limit of Remark 2.2)."
        ),
    )
    mode: Literal["ssdid", "sdid_imputation"] = Field(
        default="ssdid",
        description=(
            "Estimator mode. 'ssdid' is the paper's main estimator with a "
            "finite ``eta``. 'sdid_imputation' forces the limit ``eta -> "
            "infinity``, recovering the imputation-style sequential DiD of "
            "Remark 2.2 (Borusyak et al. 2024-style)."
        ),
    )
    K: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Maximum event-time horizon to estimate. ``None`` (default) "
            "auto-sets ``K = T - a_max`` so every estimable effect fits "
            "inside the panel."
        ),
    )
    a_min: Optional[int] = Field(
        default=None,
        description=(
            "Earliest treated cohort (1-based time index) to include. "
            "``None`` (default) uses the earliest adopting cohort."
        ),
    )
    a_max: Optional[int] = Field(
        default=None,
        description=(
            "Latest treated cohort (1-based time index) to include. ``None`` "
            "(default) uses the latest finitely-adopting cohort."
        ),
    )
    n_bootstrap: int = Field(
        default=500,
        ge=0,
        description=(
            "Number of Bayesian-bootstrap iterations for SE/CI (Section 2.3 "
            "of the paper). Set to 0 to skip inference."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Wald confidence band.",
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the bootstrap.",
    )
