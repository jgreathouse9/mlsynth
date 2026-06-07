"""Configuration for the MCNNM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from pydantic import Field
from ...config_models import BaseEstimatorConfig


class MCNNMConfig(BaseEstimatorConfig):
    """Configuration for the MC-NNM estimator.

    Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *"Matrix
    Completion Methods for Causal Panel Data Models"* (JASA). Imputes the
    treated cells of the outcome matrix via nuclear-norm-regularised
    low-rank matrix completion with unregularised two-way fixed effects
    (SOFT-IMPUTE, threshold chosen by cross-validation). Inherits the
    standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` / ``time``
    interface.

    Parameters
    ----------
    estimate_unit_fe : bool
        Estimate (unregularised) unit fixed effects. Default True.
    estimate_time_fe : bool
        Estimate (unregularised) time fixed effects. Default True.
    n_lambda : int
        Number of candidate singular-value thresholds in the CV grid.
    n_folds : int
        Cross-validation folds over the observed cells.
    inference : bool
        Run a leave-one-control jackknife for the ATT SE / CI. Default
        False (it refits the model once per control unit).
    alpha : float
        Two-sided level for the jackknife confidence interval.
    random_state : int
        Seed for the CV fold assignment.
    """

    estimate_unit_fe: bool = Field(
        default=True, description="Estimate unregularised unit fixed effects.")
    estimate_time_fe: bool = Field(
        default=True, description="Estimate unregularised time fixed effects.")
    n_lambda: int = Field(
        default=40, ge=2,
        description="Number of candidate thresholds in the CV grid.")
    n_folds: int = Field(
        default=5, ge=2, description="Cross-validation folds over observed cells.")
    inference: bool = Field(
        default=False,
        description="Run a leave-one-control jackknife for the ATT SE/CI.")
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the jackknife confidence interval.")
    random_state: int = Field(
        default=0, description="Seed for the CV fold assignment.")
