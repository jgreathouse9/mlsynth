"""Configuration for the MSQRT estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class MSQRTConfig(BaseEstimatorConfig):
    """Configuration for the MSQRT estimator.

    Shen, Song & Abadie (2025), *"Efficiently Learning Synthetic Control
    Models for High-dimensional Disaggregated Data"* (arXiv:2510.22828).
    Stacks all treated units into one matrix regression ``Y = X Theta + E``
    and fits the donor-weight matrix by Multivariate Square-root Lasso
    (nuclear-norm loss + element-wise L1), with the L1 penalty chosen by
    rolling-origin cross-validation on the pre-period. Assumes a block design
    (all treated units adopt at the same period) and inherits the standard
    ``df`` / ``outcome`` / ``treat`` / ``unitid`` / ``time`` interface.

    Parameters
    ----------
    lambda_ : float, optional
        Fixed L1 penalty. If None (default), selected by cross-validation.
    n_lambda : int
        Number of log-spaced candidate penalties in the CV grid.
    cv_initial_train, cv_val_window, cv_step, cv_folds : int, optional
        Rolling-origin CV schedule overrides. Adaptive defaults
        (scaled to the pre-period length) are used when left None.
    inference : bool
        Attach CFPT/scpi prediction intervals (Cattaneo, Feng, Palomba &
        Titiunik 2025) for all four predictands plus simultaneous bands. For
        MSQRT only the out-of-sample error is modelled. Default False.
    alpha : float
        Total miscoverage level for the intervals (default 0.1 -> 90%).
    time_dependence : {"iid", "general"}
        Time-averaging assumption for the time-averaged predictands (TAUS,
        TAUA). ``"iid"`` (default) shrinks the band by ``sqrt(L)``;
        ``"general"`` makes no serial-dependence assumption.
    """

    lambda_: Optional[float] = Field(
        default=None, gt=0.0,
        description="Fixed L1 penalty; None selects it by cross-validation.")
    n_lambda: int = Field(
        default=15, ge=2,
        description="Number of log-spaced candidate penalties in the CV grid.")
    cv_initial_train: Optional[int] = Field(
        default=None, ge=2,
        description="Initial training window for rolling-origin CV.")
    cv_val_window: Optional[int] = Field(
        default=None, ge=1,
        description="Validation window length for rolling-origin CV.")
    cv_step: Optional[int] = Field(
        default=None, ge=1, description="Step between successive CV folds.")
    cv_folds: Optional[int] = Field(
        default=None, ge=1, description="Maximum number of CV folds.")
    inference: bool = Field(
        default=False,
        description="Attach CFPT/scpi prediction intervals to the predictands.")
    alpha: float = Field(
        default=0.1, gt=0.0, lt=1.0,
        description="Total miscoverage level for the prediction intervals.")
    time_dependence: Literal["iid", "general"] = Field(
        default="iid",
        description="Time-averaging assumption for time-averaged predictands.")
    weight_col: Optional[str] = Field(
        default=None,
        description=(
            "Per-unit-constant column of size weights (e.g. market population / "
            "TV homes). When given, the unit-averaged predictands (TSUA/TAUA) are "
            "additionally reported as a size-weighted convex combination of the "
            "treated units -- a population-weighted aggregate ATT with valid SCPI "
            "intervals -- alongside the equal-weight ones. ``None`` (default) "
            "reports only the equal-weight aggregate."),
    )
