"""Configuration for the DSCAR estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class DSCARConfig(BaseEstimatorConfig):
    """Configuration for the Dynamic Synthetic Control AR (DSCAR) estimator.

    Zheng & Chen (2024), JRSS-B 86(1):155-176, "Dynamic synthetic
    control method for evaluating treatment effects in auto-regressive
    processes." Extends Abadie-Diamond-Hainmueller (2010) to settings
    with time-varying confounders, an auto-regressive outcome, spatial
    dependence in the residuals, and possibly multiple treated units
    that all turn on at a common intervention time.

    The DSC weights are computed **per post-period** via empirical-
    likelihood maximisation under per-period matching constraints
    (equations 2.7-2.9 of the paper), allowing exact matches as
    :math:`N_{co}, N_{tr} \\to \\infty` with :math:`T` fixed -- the
    asymptotic regime that suits a typical air-pollution / hourly
    panel.

    Parameters
    ----------
    exog_covariates : list of str, optional
        Time-varying exogenous covariate columns to include in the
        per-period matching. ``None`` skips the covariate match (DSC
        then matches on the lagged outcome only).
    lagged_outcome : str, optional
        Column name supplying the externally-computed
        :math:`Y_{i, t-1}` value at the **first** sample period. For
        later periods the lag is read off the panel itself. ``None``
        drops the lag constraint at ``t = 1``.
    placebo_reps : int
        Number of normalised-placebo replications for the SE on the
        DSC ATT (Section 3.2). ``0`` (default) skips placebo inference.
    el_tolerance : float
        Threshold for the QP residual ``mean|Z_1 - Z_0 w|`` at which
        the EL refinement step is attempted; matches the R reference's
        default ``0.01``. Smaller values fall back to QP weights more
        often.
    fdr_alpha : float
        Significance level for the BY-adjusted pre-period
        unconfoundedness test (Section 3.1).
    seed : int
        RNG seed for the placebo draw.

    References
    ----------
    Zheng, X., & Chen, S. X. (2024). *Dynamic synthetic control method
    for evaluating treatment effects in auto-regressive processes.*
    Journal of the Royal Statistical Society Series B, 86(1):155-176.
    """

    exog_covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Time-varying exogenous covariate column names. None to "
            "skip the covariate match (lagged-outcome match only)."
        ),
    )
    lagged_outcome: Optional[str] = Field(
        default=None,
        description=(
            "Column carrying Y_{i, t-1} at the first sample period. "
            "Required to include the lag match at t = 1."
        ),
    )
    placebo_reps: int = Field(
        default=0,
        description=(
            "Number of normalised-placebo replications for the DSC "
            "ATT standard error. 0 disables placebo inference."
        ),
    )
    el_tolerance: float = Field(
        default=1e-2,
        description=(
            "Mean-absolute-mismatch threshold for triggering the EL "
            "refinement step. R reference default: 0.01."
        ),
    )
    fdr_alpha: float = Field(
        default=0.05,
        description="Significance level for the BY pre-period test.",
    )
    seed: int = Field(default=0, description="RNG seed for placebo draws.")
