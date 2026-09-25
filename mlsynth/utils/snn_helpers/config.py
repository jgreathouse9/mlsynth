"""Configuration for the SNN estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SNNConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Nearest Neighbors (SNN) estimator.

    Agarwal, Dahleh, Shah & Shen (2021), *"Causal Matrix Completion"*
    (arXiv:2109.15154). Imputes treated units' untreated potential
    outcomes by MNAR matrix completion (anchor submatrix + principal
    component regression), generalising the Synthetic Interventions /
    synthetic-control approach. Inherits the standard ``df`` / ``outcome``
    / ``treat`` / ``unitid`` / ``time`` interface.

    Parameters
    ----------
    n_neighbors : int
        Number of synthetic neighbours (anchor-row groups) to average.
    max_rank : int, optional
        Fixed PCR truncation rank; overrides the spectral/universal rule.
    spectral_energy : float
        Singular-value energy threshold for spectral rank selection
        (used when ``max_rank`` is None and ``universal_rank`` is False).
    universal_rank : bool
        Use the Donoho-Gavish (2014) universal hard-threshold rank.
        Default True -- well-calibrated for small low-rank panels (e.g.
        Prop 99); set False to use the spectral-energy threshold.
    clip : bool
        Clip imputations to the observed value range.
    linear_span_eps : float
        Threshold for the linear span test: an entry passes when the target
        row's normalized reconstruction error is at or below this. The
        reference implementation's default is 0.1.
    subspace_eps : float
        Threshold for the subspace inclusion test: an entry passes when the
        share of the target column's energy outside the fitted subspace is at
        or below this. The reference implementation's default is 0.1.
    inference : bool
        Run a leave-one-control jackknife for the ATT SE / CI.
    alpha : float
        Two-sided level for the jackknife confidence interval.
    random_state : int
        Seed for anchor-row splitting.
    """

    n_neighbors: int = Field(
        default=1, ge=1,
        description="Number of synthetic neighbours (anchor-row groups).",
    )
    max_rank: Optional[int] = Field(
        default=None, ge=1,
        description="Fixed PCR truncation rank (overrides spectral rule).",
    )
    spectral_energy: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Singular-value energy threshold for rank selection.",
    )
    universal_rank: bool = Field(
        default=True,
        description="Use the Donoho-Gavish universal hard-threshold rank "
                    "(default; well-calibrated for small low-rank panels). "
                    "Set False to use the spectral-energy threshold instead.",
    )
    clip: bool = Field(
        default=True,
        description="Clip imputations to the observed value range.",
    )
    linear_span_eps: float = Field(
        default=0.1, gt=0.0,
        description="Linear span test threshold on the target row's normalized "
                    "reconstruction error (Assumption 3's empirical "
                    "counterpart). Reported, never gated on.",
    )
    subspace_eps: float = Field(
        default=0.1, gt=0.0,
        description="Subspace inclusion test threshold on the target column's "
                    "out-of-subspace energy share (Assumption 7's empirical "
                    "counterpart). Reported, never gated on.",
    )
    inference: bool = Field(
        default=False,
        description="Run a leave-one-control jackknife for the ATT SE/CI.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the jackknife confidence interval.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for anchor-row splitting.",
    )
