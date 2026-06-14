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

Each grid penalty is solved by the native active-set simplex QP after folding the
penalty into the design as a ``sqrt(lambda sigma_y^2) R`` augmentation (``R^T R ==
Q``); since every grid point is strictly convex (the ``lambda = 0`` point via the
ridge floor), the optimum is unique and matches cvxpy to solver tolerance with no
canonicalisation overhead in the loop. An explicit ``solver`` routes through
cvxpy instead.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .optimization import _RIDGE_FLOOR, _augmented_design
from .penalty import build_sqrt_factor
from .structures import MLSCInputs

# Reference default grid (``mlSC_estimator``'s ``lambda_grid`` argument).
_DEFAULT_GRID = np.concatenate((
    [0.0],
    np.logspace(np.log10(1e-8), np.log10(5), 50),
    np.logspace(np.log10(10), np.log10(1000), 5),
))


def _holdout_mse_native(
    inputs: MLSCInputs, X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray, penalty_scale: float,
) -> float:
    """One grid point's held-out forecast MSE via the active-set simplex QP."""
    from ..bilevel.active_set import solve_simplex_qp

    B, A = _augmented_design(X_train, Y_train, inputs, penalty_scale)
    omega = solve_simplex_qp(B, A)
    return float(np.mean((Y_test - X_test @ omega) ** 2))


def _county_sc_holdout_mse(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray, solver: Any,
) -> float:
    """``lambda = 0`` branch via cvxpy (escape hatch): tiny-ridge disaggregate SC."""
    import cvxpy as cp

    M = X_train.shape[1]
    w = cp.Variable(M)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(Y_train - X_train @ w) + _RIDGE_FLOOR * cp.sum_squares(w)),
        [cp.sum(w) == 1, w >= 0],
    )
    problem.solve(solver=solver or cp.SCS)
    if w.value is None:  # pragma: no cover - defensive: county-SC solve returned None
        return float("inf")
    return float(np.mean((Y_test - X_test @ np.asarray(w.value)) ** 2))


def _select_lambda_cv_cvxpy(
    inputs, Q, sigma_y2, grid, X_train, Y_train, X_test, Y_test, solver,
):
    """cvxpy escape hatch: the original Parameter-based grid sweep."""
    import cvxpy as cp

    M = X_train.shape[1]
    omega = cp.Variable(M)
    lambd = cp.Parameter(nonneg=True)
    objective = cp.Minimize(
        cp.sum_squares(Y_train - X_train @ omega)
        + lambd * sigma_y2 * cp.quad_form(omega, cp.psd_wrap(Q))
    )
    problem = cp.Problem(objective, [cp.sum(omega) == 1, omega >= 0])
    cv_error = np.full(grid.shape[0], np.inf)
    for i, v in enumerate(grid):
        if v == 0.0:
            cv_error[i] = _county_sc_holdout_mse(X_train, Y_train, X_test, Y_test, solver)
            continue
        lambd.value = float(v)
        try:
            problem.solve(solver=solver or cp.SCS)
        except cp.error.SolverError:  # pragma: no cover - solver hiccup on a grid point
            continue
        if omega.value is None:  # pragma: no cover - defensive
            continue
        cv_error[i] = float(np.mean((Y_test - X_test @ np.asarray(omega.value)) ** 2))
    return cv_error


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
        Block-diagonal penalty matrix, shape ``(M, M)`` (used only by the cvxpy
        escape hatch; the native path rebuilds the square-root factor).
    sigma_y2 : float
        Outcome-variance scale on the penalty (as in :func:`solve_mlsc`).
    lambda_grid : sequence of float, optional
        Candidate penalties. ``None`` -> the reference default grid.
    cv_holdout_periods : int
        Trailing pre-treatment periods held out as the forecast target.
    solver : Any
        cvxpy solver. ``None`` (default) uses the native active-set simplex QP;
        any explicit value routes through cvxpy with that solver.

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

    if solver is not None:
        cv_error = _select_lambda_cv_cvxpy(
            inputs, Q, sigma_y2, grid, X_train, Y_train, X_test, Y_test, solver
        )
    else:
        cv_error = np.array([
            _holdout_mse_native(inputs, X_train, Y_train, X_test, Y_test,
                                float(v) * sigma_y2)
            for v in grid
        ])

    if not np.isfinite(cv_error).any():  # pragma: no cover - defensive
        raise RuntimeError("mlSC cross-validation: no grid penalty was solvable.")

    return float(grid[int(np.argmin(cv_error))])
