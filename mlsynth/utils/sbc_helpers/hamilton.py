"""Hamilton (2018) filter for trend / cycle decomposition.

Implements Eq. (2) of Shi, Xi, Xie (2025):

    tau_t = alpha_0 + alpha_1 * Y_{t-h} + alpha_2 * Y_{t-h-1}
                    + ... + alpha_p * Y_{t-h-p+1}
    c_t   = Y_t - tau_t

Coefficients ``(alpha_0, ..., alpha_p)`` are estimated by OLS of ``Y_t``
on a constant and ``p`` lags of ``Y_{t-h}``. The trend is the in-sample
fit and the cycle is the residual.

The first ``h + p - 1`` observations are unavailable as targets because
the rightmost lag ``Y_{t-h-p+1}`` is undefined; those entries of
``trend_pre`` and ``cycle_pre`` are returned as ``np.nan``.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthDataError, MlsynthEstimationError
from .structures import HamiltonFit


def _build_lag_design(y: np.ndarray, h: int, p: int):
    """Build the design matrix and target vector for the Hamilton OLS.

    Returns
    -------
    X : np.ndarray
        Shape ``(T_eff, p + 1)`` design matrix with leading ones column.
    y_target : np.ndarray
        Shape ``(T_eff,)`` targets ``y_t`` for ``t = h + p - 1 .. T - 1``.
    valid_idx : np.ndarray
        Indices into ``y`` corresponding to the rows of ``X``.
    """

    T = y.shape[0]
    first = h + p - 1
    if first >= T:
        raise MlsynthDataError(
            f"Hamilton filter requires T >= h + p; got T={T}, h+p={h + p}."
        )
    valid_idx = np.arange(first, T)
    T_eff = valid_idx.size

    X = np.empty((T_eff, p + 1), dtype=float)
    X[:, 0] = 1.0
    for j in range(p):
        # alpha_{j+1} multiplies Y_{t - h - j}
        X[:, j + 1] = y[valid_idx - h - j]
    y_target = y[valid_idx]
    return X, y_target, valid_idx


def fit_hamilton_filter(
    y_pre: np.ndarray, h: int = 2, p: int = 2
) -> HamiltonFit:
    """Fit the Hamilton filter on a univariate pre-treatment series.

    Parameters
    ----------
    y_pre : np.ndarray
        Length-``T0`` pre-treatment series.
    h : int
        Forecasting horizon (paper default 2).
    p : int
        Number of self-lags (paper default 2).

    Returns
    -------
    HamiltonFit
        Coefficients plus the fitted trend / cycle over the pre-treatment
        window (with leading NaNs where lags are unavailable).
    """

    if h < 1 or p < 1:
        raise MlsynthEstimationError("Hamilton filter requires h >= 1 and p >= 1.")

    y_pre = np.asarray(y_pre, dtype=float)
    T0 = y_pre.shape[0]

    X, target, valid_idx = _build_lag_design(y_pre, h, p)

    # OLS via lstsq (handles potential rank deficiency on tiny windows).
    coefs, *_ = np.linalg.lstsq(X, target, rcond=None)

    trend = np.full(T0, np.nan)
    trend[valid_idx] = X @ coefs

    cycle = np.full(T0, np.nan)
    cycle[valid_idx] = y_pre[valid_idx] - trend[valid_idx]

    return HamiltonFit(
        coefficients=coefs,
        trend_pre=trend,
        cycle_pre=cycle,
        h=int(h),
        p=int(p),
    )


def cycle_matrix_pre(
    Y_full: np.ndarray, T0: int, h: int, p: int
):
    """Apply the Hamilton filter to every column of the pre-treatment block.

    Returns
    -------
    fits : list of HamiltonFit
        One per column of ``Y_full`` (target as ``fits[0]``, donors after).
    cycles : np.ndarray
        Shape ``(T0, N)`` matrix of pre-treatment cycles (NaN in the first
        ``h + p - 1`` rows).
    """

    Y_pre = Y_full[:T0]
    N = Y_pre.shape[1]
    fits = [fit_hamilton_filter(Y_pre[:, i], h=h, p=p) for i in range(N)]
    cycles = np.column_stack([f.cycle_pre for f in fits])
    return fits, cycles
