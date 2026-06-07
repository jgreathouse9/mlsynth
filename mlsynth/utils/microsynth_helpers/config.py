"""Configuration for the MicroSynth estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class MicroSynthConfig(BaseEstimatorConfig):
    """Configuration for the MicroSynth estimator.

    Implements Robbins & Davenport (2021, *J. Stat. Software*),
    "microsynth: Synthetic Control Methods for Disaggregated and
    Micro-Level Data in R". A user-level balancing estimator: solve a
    constrained QP for non-negative simplex weights on control users
    that exactly balance covariate moments against the treated group's
    moments, then read off the ATT as the weighted-mean outcome
    difference.

    Unlike aggregate-unit SCM estimators in :mod:`mlsynth`, MicroSynth
    operates at the individual-user level with many treated units and
    a large donor pool of controls. The dual ascent solver scales with
    the number of balancing constraints (``d + 1``), not with the
    number of controls, making it tractable for ``N_C`` in the
    millions on a single machine.
    """

    covariates: List[str] = Field(
        ...,
        description=(
            "Column names in ``df`` to use as balancing covariates. "
            "These must be time-invariant per unit (a single value "
            "per user); time-varying features should be collapsed by "
            "the caller (e.g., to pre-treatment means) before passing."
        ),
    )
    outcome_lag_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional list of pre-treatment time labels (as found in "
            "the ``time`` column) whose outcome values become "
            "additional balancing constraints -- the canonical lagged-"
            "outcome predictors. Appended after ``covariates``."
        ),
    )
    standardize_covariates: bool = Field(
        default=True,
        description=(
            "Standardize each covariate to unit SD across all units "
            "before fitting. Improves numerical conditioning of the "
            "dual problem; does not change the final weights."
        ),
    )
    balance_tol: float = Field(
        default=1e-4,
        gt=0.0,
        description=(
            "Maximum absolute standardized mean difference per "
            "covariate accepted as 'balanced' after weighting. Used "
            "for the feasibility diagnostic."
        ),
    )
    max_iter: int = Field(
        default=500,
        ge=10,
        description="L-BFGS-B maximum iterations for the dual problem.",
    )
    gtol: float = Field(
        default=1e-8,
        gt=0.0,
        description="L-BFGS-B gradient tolerance.",
    )
    run_inference: bool = Field(
        default=True,
        description="Whether to compute a bootstrap confidence interval.",
    )
    n_bootstrap: int = Field(
        default=500,
        ge=2,
        description="Bootstrap replications for CI.",
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the bootstrap.",
    )
