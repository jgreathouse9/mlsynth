"""Rolling cross-validation over time for the mlSC penalty (Section 5.2).

Selects the penalty ``lambda`` by holding out the last ``cv_holdout_periods``
pre-treatment periods, refitting the penalized state-county program over a grid
of candidate penalties on the earlier pre-period, and choosing the penalty whose
held-out one-step-ahead forecast MSE is smallest.

This is a faithful port of ``get_lambda_cv`` in the author's reference
implementation (``multi_level_sc_estimator.mlSC``): the same train/test split
(``t_cv = T0 - cv_holdout_periods``), the same objective
``||Y - X omega||^2 + lambda * sigma_y^2 * omega^T Q omega`` on the simplex, and
the same default grid (with a ``0`` candidate that drops the penalty to recover
fully-disaggregated SC, regularized by a tiny ridge for a unique solution).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import cvxpy as cp
import numpy as np

from .structures import MLSCInputs

# Reference default grid (``mlSC_estimator``'s ``lambda_grid`` argument).
_DEFAULT_GRID = np.concatenate((
    [0.0],
    np.logspace(np.log10(1e-8), np.log10(5), 50),
    np.logspace(np.log10(10), np.log10(1000), 5),
))


def _county_sc_holdout_mse(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray, solver: Any,
) -> float:
    """``lambda = 0`` branch: fully-disaggregated SC with a tiny ridge.

    Matches the reference ``synthetic_control_counties`` (a small ``1e-8``
    L2 term keeps the simplex solution unique on a flat objective).
    """
    M = X_train.shape[1]
    w = cp.Variable(M)
    objective = cp.Minimize(
        cp.sum_squares(Y_train - X_train @ w) + 1e-8 * cp.sum_squares(w)
    )
    problem = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])
    problem.solve(solver=solver or cp.SCS)
    if w.value is None:
        return float("inf")
    return float(np.mean((Y_test - X_test @ np.asarray(w.value)) ** 2))


def select_lambda_cv(
    inputs: MLSCInputs,
    Q: np.ndarray,
    sigma_y2: float,
    *,
    lambda_grid: Optional[Sequence[float]] = None,
    cv_holdout_periods: int = 1,
    solver: Any = None,
) -> float:
    """Select the mlSC penalty by rolling cross-validation over time.

    Parameters
    ----------
    inputs : MLSCInputs
        Pre-processed two-level panel (uses the pre-period only).
    Q : np.ndarray
        Block-diagonal penalty matrix, shape ``(M, M)``.
    sigma_y2 : float
        Outcome-variance scale on the penalty (as in :func:`solve_mlsc`).
    lambda_grid : sequence of float, optional
        Candidate penalties. ``None`` -> the reference default grid.
    cv_holdout_periods : int
        Trailing pre-treatment periods held out as the forecast target.
    solver : Any
        cvxpy solver; ``None`` -> ``cp.SCS``.

    Returns
    -------
    float
        The grid penalty with the smallest held-out forecast MSE.
    """
    T0 = inputs.T0
    t_cv = T0 - cv_holdout_periods
    if t_cv < 1:
        raise ValueError(
            "cross-validation needs at least one training period: "
            f"T0={T0} with cv_holdout_periods={cv_holdout_periods} leaves none."
        )

    Y = inputs.Y_agg_treated[:T0]
    X = inputs.X_disagg[:T0, :]
    X_train, Y_train = X[:t_cv], Y[:t_cv]
    X_test, Y_test = X[t_cv:T0], Y[t_cv:T0]

    grid = np.asarray(_DEFAULT_GRID if lambda_grid is None else lambda_grid, dtype=float)

    # Build the penalized problem once and re-solve per lambda via a Parameter
    # (the reference reuses a single cvxpy Problem across the grid).
    M = X_train.shape[1]
    omega = cp.Variable(M)
    lambd = cp.Parameter(nonneg=True)
    constraints = [cp.sum(omega) == 1, omega >= 0]
    objective = cp.Minimize(
        cp.sum_squares(Y_train - X_train @ omega)
        + lambd * sigma_y2 * cp.quad_form(omega, cp.psd_wrap(Q))
    )
    problem = cp.Problem(objective, constraints)

    cv_error = np.full(grid.shape[0], np.inf)
    for i, v in enumerate(grid):
        if v == 0.0:
            cv_error[i] = _county_sc_holdout_mse(
                X_train, Y_train, X_test, Y_test, solver
            )
            continue
        lambd.value = float(v)
        try:
            problem.solve(solver=solver or cp.SCS)
        except cp.error.SolverError:  # pragma: no cover - solver hiccup on a grid point
            continue
        if omega.value is None:
            continue
        cv_error[i] = float(np.mean((Y_test - X_test @ np.asarray(omega.value)) ** 2))

    if not np.isfinite(cv_error).any():  # pragma: no cover - defensive
        raise RuntimeError("mlSC cross-validation: no grid penalty was solvable.")

    return float(grid[int(np.argmin(cv_error))])
