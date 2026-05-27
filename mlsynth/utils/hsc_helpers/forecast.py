"""Forecasters for the HSC smooth residual ``E``.

In post-treatment periods the treated-unit smooth component ``E`` is
extrapolated by a time-series forecaster and added to the donor-matched
component to form the counterfactual. The default is an ARIMA(1, 1, 0)
forecaster -- the correctly specified model for the paper's idiosyncratic
ARIMA(1, 1, 0) stochastic-trend DGP.

The ARIMA(1, 1, 0) forecast is computed in closed form (no iterative MLE):
the first differences follow a zero-mean AR(1), so future increments decay
geometrically and the level path integrates them forward. This is fully
deterministic, which keeps HSC results exactly reproducible.
"""

from __future__ import annotations

import numpy as np


def arima110_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """Closed-form ARIMA(1, 1, 0) forecast of a level series ``x``.

    Fits a zero-mean AR(1) to ``diff(x)`` by least squares, then forecasts
    ``h`` future increments ``d_{t+k} = phi^k d_t`` and integrates to levels.

    Parameters
    ----------
    x : np.ndarray
        Observed level series.
    h : int
        Forecast horizon.

    Returns
    -------
    np.ndarray
        Length-``h`` forecast of the level path.
    """

    x = np.asarray(x, dtype=float)
    if h <= 0:
        return np.empty(0)
    dx = np.diff(x)
    if dx.size < 3:
        return np.repeat(float(x[-1]) if x.size else 0.0, h)
    den = float(dx[:-1] @ dx[:-1])
    phi = float(dx[1:] @ dx[:-1]) / den if den > 1e-12 else 0.0
    phi = float(np.clip(phi, -0.98, 0.98))
    out = np.empty(h)
    level = float(x[-1])
    increment = float(dx[-1])
    for k in range(h):
        increment *= phi
        level += increment
        out[k] = level
    return out


def last_value_forecast(x: np.ndarray, h: int) -> np.ndarray:
    """Carry the last observed value forward as a constant."""
    x = np.asarray(x, dtype=float)
    if h <= 0:
        return np.empty(0)
    return np.repeat(float(x[-1]) if x.size else 0.0, h)


FORECASTERS = {
    "arima110": arima110_forecast,
    "last": last_value_forecast,
}


def forecast_smooth(E: np.ndarray, h: int, method: str = "arima110") -> np.ndarray:
    """Dispatch a forecaster by name over the smooth residual ``E``."""
    if method not in FORECASTERS:
        raise ValueError(
            f"Unknown HSC forecaster '{method}'. Options: {sorted(FORECASTERS)}."
        )
    return FORECASTERS[method](E, h)
