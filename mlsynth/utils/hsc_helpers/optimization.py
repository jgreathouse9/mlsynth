"""HSC fit at a fixed allocation, and rolling-origin CV over the allocation.

``fit_at_rho`` solves the profiled donor QP and returns the smooth residual;
``select_rho_by_cv`` chooses ``rho`` by sklearn's rolling-origin
:class:`~sklearn.model_selection.TimeSeriesSplit`, scoring each candidate by
its out-of-sample prediction error (donor match + forecast of the smooth
component) on held-out pre-period blocks.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .forecast import forecast_smooth
from .formulation import fit_donor_weights, smoother_and_metric


def fit_at_rho(
    X_pre: np.ndarray,
    Y_pre: np.ndarray,
    rho: float,
    q: int,
    ridge: float = 1e-6,
    solver: Optional[object] = None,
    ridge_abs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the profiled HSC problem at a fixed ``rho``.

    ``ridge_abs`` (e.g. the SDID-style coefficient) overrides the relative
    ``ridge`` when supplied.

    Returns
    -------
    omega : np.ndarray
        Donor weights on the simplex, shape ``(N,)``.
    E_pre : np.ndarray
        Fitted smooth residual ``S_{rho,q} (Y_pre - X_pre omega)``, ``(T0,)``.
    """

    T0 = X_pre.shape[0]
    S, W = smoother_and_metric(T0, q, rho)
    omega = fit_donor_weights(
        X_pre, Y_pre, W, ridge=ridge, ridge_abs=ridge_abs, solver=solver
    )
    E_pre = S @ (Y_pre - X_pre @ omega)
    return omega, E_pre


def select_rho_by_cv(
    X_pre: np.ndarray,
    Y_pre: np.ndarray,
    q: int,
    rho_grid: Sequence[float],
    n_splits: int = 3,
    ridge: float = 1e-6,
    forecaster: str = "arima110",
    solver: Optional[object] = None,
    ridge_abs: Optional[float] = None,
) -> Tuple[float, Dict[float, float]]:
    """Select ``rho`` by rolling-origin cross-validation.

    For each candidate ``rho`` and each expanding-window
    ``TimeSeriesSplit`` fold, HSC is fit on the training block; the held-out
    block is predicted by ``X_val @ omega + forecast(E)`` and scored by mean
    squared error. The ``rho`` with the lowest average fold error wins.

    Returns
    -------
    best_rho : float
        Selected allocation parameter.
    cv_curve : dict
        ``{rho: mean CV error}``.
    """

    indices = np.arange(len(Y_pre))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    cv_curve: Dict[float, float] = {}
    best_rho, best_score = float(rho_grid[0]), np.inf

    for rho in rho_grid:
        fold_errors = []
        for train_idx, val_idx in splitter.split(indices):
            if len(train_idx) <= q + 2:
                continue
            omega, E = fit_at_rho(
                X_pre[train_idx], Y_pre[train_idx], rho, q, ridge, solver,
                ridge_abs=ridge_abs,
            )
            prediction = X_pre[val_idx] @ omega + forecast_smooth(
                E, len(val_idx), forecaster
            )
            fold_errors.append(float(np.mean((Y_pre[val_idx] - prediction) ** 2)))
        score = float(np.mean(fold_errors)) if fold_errors else np.inf
        cv_curve[float(rho)] = score
        if score < best_score:
            best_score, best_rho = score, float(rho)

    return best_rho, cv_curve
