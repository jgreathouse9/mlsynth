"""Step 2 of the SBC procedure: extrapolate the treated unit's trend.

From Shi, Xi, Xie (2025), Section 3.1:

    tau_hat_{1, t} = alpha_hat_{1, 1} Y_{1, t - h}
                   + ... + alpha_hat_{1, p} Y_{1, t - h - p + 1},
                   T_0 + 1 <= t <= T_0 + h.

The intercept ``alpha_0`` from the pre-treatment fit is NOT applied here:
the extrapolation uses only the slope coefficients on the treated unit's
own lags. This matches the paper's display equation.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError
from .structures import HamiltonFit


def forecast_treated_trend(
    y_target: np.ndarray,
    treated_fit: HamiltonFit,
    T0: int,
    horizon: int,
) -> np.ndarray:
    """Forecast the treated unit's trend ``horizon`` periods past ``T_0``.

    Parameters
    ----------
    y_target : np.ndarray
        Length-``T`` observed treated series.
    treated_fit : HamiltonFit
        Hamilton filter fit on ``y_target[:T0]``.
    T0 : int
        End of pre-treatment window (exclusive).
    horizon : int
        Number of post-treatment periods to forecast. Typically
        ``T - T0``.

    Returns
    -------
    np.ndarray
        Length-``horizon`` projected trend ``tau_hat_{1, T0+1..T0+horizon}``.
    """

    if horizon <= 0:
        return np.empty(0, dtype=float)

    h = treated_fit.h
    p = treated_fit.p
    coefs = treated_fit.coefficients  # (p+1,)
    if coefs.shape[0] != p + 1:
        raise MlsynthEstimationError(
            f"Hamilton coefficients have length {coefs.shape[0]}; "
            f"expected p+1={p + 1}."
        )

    slope = coefs[1:]   # alpha_1 .. alpha_p

    out = np.empty(horizon, dtype=float)
    for step in range(horizon):
        t = T0 + step  # zero-indexed; corresponds to paper's T_0 + step + 1
        # Lag j (j=0..p-1) addresses y[t - h - j]
        idx = np.array([t - h - j for j in range(p)], dtype=int)
        if np.any(idx < 0):
            raise MlsynthEstimationError(
                "Trend forecast requires y_target indices that fall before "
                "the start of the panel; reduce h or extend the pre-window."
            )
        out[step] = float(slope @ y_target[idx])
    return out
