"""Configuration for the HSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Union
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class HSCConfig(BaseEstimatorConfig):
    """Configuration for the Harmonic Synthetic Control (HSC) estimator.

    Implements:

        Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method."

    HSC matches donors under a frequency-dependent metric and absorbs the
    treated unit's low-frequency residual into a smooth component that is
    forecast forward. A single allocation parameter ``rho in [0, 1]``,
    selected by rolling-origin cross-validation, interpolates between
    synthetic control on ``q``-th differences (``rho -> 0``) and synthetic
    control on levels with an intercept/trend (``rho -> 1``).

    Parameters
    ----------
    q : int
        Smoothness order of the difference operator (1 or 2). ``q=1``
        controls a stochastic trend / random-walk drift; ``q=2`` controls a
        local-linear trend. Default 1.
    rho_grid : list of float
        Candidate allocation values in ``[0, 1]`` searched by CV. Default
        ``[0.0, 0.2, 0.5, 0.8, 0.97]``.
    cv_splits : int
        Number of rolling-origin folds (sklearn ``TimeSeriesSplit``). Default 3.
    ridge : float
        Relative ridge coefficient on the donor weights for QP conditioning.
        Default ``1e-6``.
    forecaster : {"arima110", "last"}
        Forecaster for the smooth residual. ``"arima110"`` (default) is a
        closed-form ARIMA(1, 1, 0); ``"last"`` carries the last value forward.
    display_graphs : bool
        Show the observed-vs-counterfactual plot after fitting.

    Notes
    -----
    Following Liu & Xu (2026), HSC ships the *point* estimator only; uncertainty
    quantification is deliberately out of scope, so there are no inference
    options here.
    """

    q: Literal[1, 2] = Field(
        default=1,
        description="Smoothness order of the difference operator (1 or 2).",
    )
    rho_grid: List[float] = Field(
        default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 0.97],
        description="Candidate allocation values in [0, 1] searched by CV.",
    )
    cv_splits: int = Field(
        default=3, ge=2,
        description="Rolling-origin CV folds (sklearn TimeSeriesSplit).",
    )
    ridge: Union[float, Literal["sdid"]] = Field(
        default=1e-6,
        description="Donor-weight ridge. A float is a relative coefficient "
                    "(ridge * trace(X'WX)/N). The string 'sdid' uses the "
                    "data-driven SDID-style penalty zeta^2 T0 with "
                    "zeta = T_post^{1/4} sigma_dX (Liu & Xu 2026 sec. 7), "
                    "which diversifies the donor weights.",
    )
    forecaster: Literal["arima110", "last"] = Field(
        default="arima110",
        description="Smooth-residual forecaster: ARIMA(1,1,0) or last-value.",
    )
    display_graphs: bool = Field(
        default=False,
        description="Show the HSC counterfactual plot after fitting.",
    )

    @model_validator(mode="after")
    def _check_hsc_params(cls, values: Any) -> Any:
        grid = values.rho_grid
        if not grid:
            raise MlsynthConfigError("rho_grid must contain at least one value.")
        if any((r < 0.0 or r > 1.0) for r in grid):
            raise MlsynthConfigError("All rho_grid values must lie in [0, 1].")
        if not isinstance(values.ridge, str) and values.ridge < 0.0:
            raise MlsynthConfigError("ridge must be non-negative (or 'sdid').")
        return values
