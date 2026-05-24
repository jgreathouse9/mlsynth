"""HAC variance machinery for proximal GMM inference.

The three proximal estimators (:mod:`.estimation`) all close with a GMM
sandwich variance for the ATT. The "meat" of that sandwich is a
Heteroskedasticity- and Autocorrelation-Consistent (HAC) estimate of the
long-run variance of the stacked moment conditions, formed with a
Bartlett kernel. This mirrors the reference implementation of

    Shi, X., Li, K., Miao, W., Hu, M., & Tchetgen Tchetgen, E. (2023).
    "Theory for identification and Inference with Synthetic Controls: A
    Proximal Causal Inference Framework." arXiv:2108.13935.

and was validated value-for-value against the authors' code
(``freshtaste/proximal``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bartlett(lag_order: int, truncation_lag: int) -> float:
    """Bartlett kernel weight ``1 - |lag| / (lag_trunc + 1)``.

    Parameters
    ----------
    lag_order : int
        Current lag order (``0`` returns weight ``1``).
    truncation_lag : int
        Bandwidth; lags beyond it receive weight ``0``.

    Returns
    -------
    float
        The kernel weight for ``lag_order``.
    """

    if np.abs(lag_order) <= truncation_lag:
        return 1 - np.abs(lag_order) / (truncation_lag + 1)
    return 0.0


def hac(
    moment_conditions: np.ndarray,
    truncation_lag: int,
    kernel: Callable[[int, int], float] = bartlett,
) -> np.ndarray:
    """HAC long-run covariance of stacked moment conditions.

    Parameters
    ----------
    moment_conditions : np.ndarray
        Moment matrix of shape ``(n_obs, n_moments)`` (rows are time
        periods, columns are moment conditions).
    truncation_lag : int
        Kernel bandwidth (number of autocovariance lags to include).
    kernel : Callable[[int, int], float], optional
        Lag-weighting kernel. Defaults to :func:`bartlett`.

    Returns
    -------
    np.ndarray
        The ``(n_moments, n_moments)`` HAC covariance estimate.
    """

    n_obs, n_moments = moment_conditions.shape
    omega = np.zeros((n_moments, n_moments))

    # Lag 0.
    omega += (moment_conditions.T @ moment_conditions) / n_obs

    # Lags 1..truncation_lag (symmetrized).
    for lag in range(1, min(truncation_lag, n_obs - 1) + 1):
        weight = kernel(lag, truncation_lag)
        autocov = (moment_conditions[:-lag].T @ moment_conditions[lag:]) / n_obs
        omega += weight * (autocov + autocov.T)

    return omega
