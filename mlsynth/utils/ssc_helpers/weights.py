"""Synthetic-control weights for SSC (intercept + simplex, batch over units).

Each unit's untreated outcome is modelled as ``a_i + Y_t' b_i`` where ``b_i``
lies on the simplex (non-negative, sums to one) with ``b_ii = 0`` -- i.e. a
demeaned synthetic control of unit ``i`` on *all other* units (Cao, Lu & Wu
2026, eq. 2.1). Fitting every unit in turn yields the intercept vector ``a`` and
the weight matrix ``B`` used throughout the estimator and its inference.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sc_weights_one(y: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
    """Demeaned simplex synthetic control of one unit on the others.

    Solves ``min_b || (y - mean y) - (X - mean X) b ||^2`` subject to
    ``b >= 0`` and ``sum(b) = 1``, then recovers the intercept
    ``a = mean(y) - mean(X) b``.

    Parameters
    ----------
    y : np.ndarray, shape (T0,)
        Treated unit's pre-treatment series.
    X : np.ndarray, shape (T0, N-1)
        Donor units' pre-treatment series (columns).

    Returns
    -------
    a : float
        Intercept.
    b : np.ndarray, shape (N-1,)
        Simplex weights on the donors.
    """
    import cvxpy as cp

    yd = y - y.mean()
    Xd = X - X.mean(axis=0, keepdims=True)
    n = X.shape[1]
    b = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(yd - Xd @ b)),
                      [b >= 0, cp.sum(b) == 1])
    prob.solve(solver=cp.CLARABEL)
    bv = np.asarray(b.value, dtype=float)
    bv = np.clip(bv, 0.0, None)
    s = bv.sum()
    bv = bv / s if s > 0 else np.full(n, 1.0 / n)
    a = float(y.mean() - X.mean(axis=0) @ bv)
    return a, bv


def synthetic_control_batch(Y_pre: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit :func:`sc_weights_one` for every unit (each treated, others donors).

    Parameters
    ----------
    Y_pre : np.ndarray, shape (N, T0)
        Pre-treatment outcomes (rows are units, columns are periods).

    Returns
    -------
    a_hat : np.ndarray, shape (N,)
        Per-unit intercepts.
    B_hat : np.ndarray, shape (N, N)
        Weight matrix; row ``i`` holds unit ``i``'s donor weights with a zero
        on the diagonal.
    """
    N, _ = Y_pre.shape
    a_hat = np.zeros(N)
    B_hat = np.zeros((N, N))
    for i in range(N):
        others = [j for j in range(N) if j != i]
        a_i, b_i = sc_weights_one(Y_pre[i], Y_pre[others].T)
        a_hat[i] = a_i
        B_hat[i, others] = b_i
    return a_hat, B_hat
