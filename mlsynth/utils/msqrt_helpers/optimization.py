"""Multivariate Square-root Lasso solve + rolling-origin lambda selection.

Estimator (Shen, Song & Abadie 2025, eq. 5)::

    Theta_hat = argmin_Theta  (1/sqrt(T0)) * ||Y - X Theta||_*  +  lambda * ||Theta||_1

where ``||.||_*`` is the nuclear norm (the "square-root"/pivotal loss; tuning
does not depend on the unknown noise variance) and ``||.||_1`` is the
element-wise L1 penalty driving donor selection in the high-dimensional regime.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cvxpy as cp
import numpy as np


def fit_msqrt_weights(Y: np.ndarray, X: np.ndarray, lambd: float, *, tol: float = 1e-2):
    """Solve eq. (5) for the donor-weight matrix ``Theta``.

    Parameters
    ----------
    Y : (T, m) treated outcomes; X : (T, n) donor outcomes; lambd : L1 penalty.

    Returns
    -------
    Theta_hat : (n, m) weight matrix
    Y_hat : (T, m) fitted treated outcomes ``X @ Theta_hat``
    nonzero_per_col : (m,) count of active donors per treated unit
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, m = Y.shape
    n = X.shape[1]

    Theta = cp.Variable((n, m))
    loss = (1.0 / np.sqrt(T)) * cp.norm(Y - X @ Theta, "nuc")
    reg = lambd * cp.norm(Theta, 1)
    prob = cp.Problem(cp.Minimize(loss + reg))
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MSQRT solve did not converge (status={prob.status}).")

    Theta_hat = np.asarray(Theta.value, dtype=float)
    Y_hat = X @ Theta_hat
    nonzero_per_col = np.sum(np.abs(Theta_hat) > tol, axis=0)
    return Theta_hat, Y_hat, nonzero_per_col


def _cv_schedule(T0: int, initial_train, val_window, step):
    """Adapt the rolling-origin CV schedule to the available pre-period."""
    val_window = val_window or max(1, T0 // 5)
    initial_train = initial_train or max(2, int(round(0.6 * T0)))
    step = step or val_window
    initial_train = min(initial_train, max(2, T0 - val_window))
    return int(initial_train), int(val_window), int(step)


def select_lambda_cv(
    Y_pre: np.ndarray,
    X_pre: np.ndarray,
    lambdas: Sequence[float],
    *,
    initial_train: Optional[int] = None,
    val_window: Optional[int] = None,
    step: Optional[int] = None,
    n_folds: Optional[int] = None,
) -> float:
    """Pick ``lambda`` by rolling-origin (expanding-window) cross-validation.

    Falls back to the smallest grid value if no fold can be formed (very short
    pre-period). Returns the lambda minimising mean validation MSE.
    """
    Y_pre = np.asarray(Y_pre, dtype=float)
    X_pre = np.asarray(X_pre, dtype=float)
    T0 = Y_pre.shape[0]
    lambdas = list(lambdas)

    init, val, stp = _cv_schedule(T0, initial_train, val_window, step)
    fold_starts = list(range(init, T0 - val + 1, stp))
    if n_folds is not None:
        fold_starts = fold_starts[:n_folds]
    if not fold_starts:
        return float(min(lambdas))

    best_lambda, best_score = float(lambdas[0]), np.inf
    for lam in lambdas:
        scores = []
        for tr_end in fold_starts:
            Theta, _, _ = fit_msqrt_weights(Y_pre[:tr_end], X_pre[:tr_end], lam)
            val_hat = X_pre[tr_end:tr_end + val] @ Theta
            err = Y_pre[tr_end:tr_end + val] - val_hat
            scores.append(float(np.mean(err ** 2)))
        s = float(np.mean(scores))
        if s < best_score:
            best_score, best_lambda = s, float(lam)
    return best_lambda
