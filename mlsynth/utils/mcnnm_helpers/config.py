"""Configuration for the MCNNM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


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
    lam : float or None
        The singular-value threshold SOFT-IMPUTE shrinks against. ``None``
        (default) selects it by cross-validation over the observed cells.
        Giving it fixes the fit, which is what reproduces a published one and
        what lets a comparison against another implementation separate the
        estimator from its tuning. The scale is the outcome's: ``lam`` is an
        absolute threshold on the singular values of the demeaned outcome
        matrix, so a value another package reports on its own normalised grid
        does not transfer. Read ``result.best_lambda`` off a
        cross-validated fit to replay it.
    n_lambda : int
        Number of candidate singular-value thresholds in the CV grid. Applies
        only when ``lam`` is None.
    n_folds : int
        Cross-validation folds over the observed cells. Applies only when
        ``lam`` is None.
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
    lam: Optional[float] = Field(
        default=None, gt=0.0,
        description=(
            "Singular-value threshold for SOFT-IMPUTE. None cross-validates "
            "it. The scale is the outcome's, so a value another package "
            "reports on its own normalised grid does not transfer; read "
            "result.best_lambda off a cross-validated fit to replay it. Must "
            "be strictly positive: at zero SOFT-IMPUTE's shrinkage is the "
            "identity and the low-rank term never propagates to the "
            "unobserved cells, so the fit collapses to the fixed effects "
            "alone."))
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

    @model_validator(mode="after")
    def check_tuning(cls, values: Any) -> Any:
        if values.lam is None:
            return values
        given = [f for f in ("n_lambda", "n_folds") if f in values.model_fields_set]
        if given:
            raise MlsynthConfigError(
                f"lam fixes the threshold, so the cross-validation that "
                f"{' and '.join(given)} configures never runs; pass lam alone, "
                f"or drop it to search over a grid of {values.n_lambda} "
                f"thresholds in {values.n_folds} folds."
            )
        return values
