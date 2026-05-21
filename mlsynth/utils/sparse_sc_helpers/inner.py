"""Inner W-weight QP for SparseSC.

Given V-weights ``v`` (a vector of predictor importance), the donor
weights ``w`` solve the canonical SCM simplex QP

    min over w >= 0, sum(w) = 1:
        w' X0' diag(v) X0 w  -  2 X1' diag(v) X0 w.

This is the same QP MATLAB's ``quadprog`` solves inside
``loss_function.m``; we use cvxpy with OSQP (with SCS fallback).
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np


def solve_w(
    v: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    solver: Any = None,
) -> np.ndarray:
    """Return the donor-weight vector ``w`` on the simplex.

    Parameters
    ----------
    v : np.ndarray
        Length-``P`` V-weight vector.
    X1 : np.ndarray
        Length-``P`` treated predictor vector.
    X0 : np.ndarray
        ``(P, N)`` donor predictor matrix.
    solver : Any
        cvxpy solver. ``None`` -> OSQP, with SCS as fallback.
    """
    N = X0.shape[1]
    D = np.diag(v)
    H = X0.T @ D @ X0
    f = -X1 @ D @ X0
    w = cp.Variable(N, nonneg=True)
    objective = cp.quad_form(w, cp.psd_wrap(H)) + 2.0 * f @ w
    constraints = [cp.sum(w) == 1.0, w <= 1.0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=solver or cp.OSQP)
    except Exception:
        problem.solve(solver=cp.SCS)
    if w.value is None:
        problem.solve(solver=cp.SCS)
    if w.value is None:
        raise RuntimeError(
            f"SparseSC inner W-weight QP failed (status={problem.status})."
        )
    return np.clip(np.asarray(w.value, dtype=float), 0.0, None)
