"""Configuration for the SBC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SBCConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Business Cycle (SBC) estimator.

    Implements:

        Shi, Z., Xi, J., & Xie, H. (2025). "A Synthetic Business Cycle
        Approach to Counterfactual Analysis with Nonstationary
        Macroeconomic Data." arXiv:2505.22388.

    Parameters
    ----------
    h : int
        Hamilton-filter forecasting horizon (paper's recommendation:
        roughly two to four years; default ``h=2``).
    p : int
        Number of self-lags used by the Hamilton filter (paper default
        ``p=2``).
    weights_mode : {"simplex", "unrestricted"}
        Synthetic-control variant for the cycle imputation step.
        ``"simplex"`` (default) matches the paper's Eq. (3): non-negative
        weights summing to 1, no intercept. ``"unrestricted"`` runs an
        OLS with intercept (Doudchenko-Imbens vertical regression style).
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    """

    h: int = Field(default=2, ge=1, description=(
        "Hamilton-filter forecasting horizon. Default 2 follows Hamilton "
        "(2018) and Shi/Xi/Xie (2025)."
    ))
    p: int = Field(default=2, ge=1, description=(
        "Number of self-lags used by the Hamilton filter."
    ))
    weights_mode: Literal["simplex", "unrestricted"] = Field(
        default="simplex",
        description=(
            "Weighting scheme for the SCM step on cycles. 'simplex' is "
            "the paper's default; 'unrestricted' is the vertical-regression "
            "alternative discussed in Section 2.1."
        ),
    )
    display_graphs: bool = Field(default=False, description=(
        "Show the SBC counterfactual plot after fitting."
    ))
