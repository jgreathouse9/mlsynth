"""Configuration for the RMSI estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class RMSIConfig(BaseEstimatorConfig):
    """Configuration for the RMSI estimator.

    Agarwal, Choi & Yuan (2026), *"Robust Matrix Estimation with Side
    Information"* (arXiv:2603.24833). Imputes the treated counterfactual of a
    block-adoption causal panel by a four-component sieve + nuclear-norm matrix
    estimator that exploits unit-level (row) and time-level (column) covariates.
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` interface.

    Parameters
    ----------
    unit_covariates : list of str
        Columns that vary across units and are (approximately) constant within a
        unit; averaged per unit to form the row feature matrix ``X``. May be
        empty (then the row projection captures only the mean).
    time_covariates : list of str
        Columns that vary across periods and are constant across units within a
        period; averaged per period to form the column feature matrix ``Z``.
        May be empty.
    sieve_order : int
        Polynomial sieve order ``J`` (default 2, matching the paper).
    rank : int, optional
        Factor rank for the block recombination; chosen by a relative-magnitude
        singular-value threshold when None.
    """

    unit_covariates: List[str] = Field(
        default_factory=list,
        description="Unit-level covariate columns (row side information).")
    time_covariates: List[str] = Field(
        default_factory=list,
        description="Time-level covariate columns (column side information).")
    sieve_order: int = Field(
        default=2, ge=1, description="Polynomial sieve order J.")
    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Factor rank; relative-threshold estimate when None.")
