"""Lambda sweep + V-weight optimization for SparseSC.

For each lambda on the grid, the outer V-weight problem is a smooth
bound-constrained nonlinear program (``v >= 0``) solved with
``scipy.optimize.minimize`` using L-BFGS-B. Each function evaluation
calls the inner W-weight QP, so the total cost is roughly

    |grid| * (outer iterations) * (one cvxpy solve per outer iter).

The selected lambda is the value that minimizes the *unpenalized*
validation-block MSE, regardless of which block the outer V-objective
uses. The outer V-objective window is controlled by
``outer_loss_window``:

* ``"validation"`` (default, paper) -- outer V minimizes validation-
  block MSE + lambda * ||V||_1. Matches Vives-i-Bastida (2023)
  Algorithm 1.
* ``"training"`` -- outer V minimizes training-block MSE + lambda *
  ||V||_1. Matches the unpublished MATLAB driver ``sparse_synth.m``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .inner import solve_w
from .objective import outer_loss, selection_mse


def default_lambda_grid(size: int = 51) -> np.ndarray:
    """Return ``[0, logspace(-4, 0, size - 1)]`` (matches MATLAB)."""
    return np.concatenate([[0.0], np.logspace(-4, 0, size - 1)])


def default_v20(X0: np.ndarray) -> np.ndarray:
    """MATLAB starting v2 = (sd_1 / sd_k)^2 for k > 1."""
    sd = X0.std(axis=1, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (sd[0] / sd[1:]) ** 2


def sweep_lambda(
    X1: np.ndarray,
    X0: np.ndarray,
    Y1: np.ndarray,
    Y0: np.ndarray,
    T0_total: int,
    T0_train: int,
    lambda_grid: Optional[np.ndarray] = None,
    solver: Any = None,
    max_outer_iter: int = 200,
    ftol: float = 1e-8,
    outer_loss_window: str = "validation",
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep lambda and return the best V-weights.

    Parameters
    ----------
    outer_loss_window : {"validation", "training"}
        Which pre-treatment block the outer V-objective evaluates the
        outcome MSE over. ``"validation"`` (default) follows the paper;
        ``"training"`` follows the MATLAB driver.

    Returns
    -------
    optv : np.ndarray
        Final V-weights, shape ``(P,)`` with ``optv[0] = 1``.
    opt_lambda : float
        Lambda value selected on the validation MSE.
    grid : np.ndarray
        Lambda grid actually used.
    outer_curve : np.ndarray
        Penalized outer objective at each grid point.
    val_curve : np.ndarray
        Unpenalized validation MSE at each grid point. This is what
        selects the optimal lambda.
    v_path : np.ndarray
        Per-grid-point V-weights, shape ``(len(grid), P)``.
    """
    if outer_loss_window not in {"validation", "training"}:
        raise ValueError(
            "outer_loss_window must be 'validation' or 'training', "
            f"got {outer_loss_window!r}."
        )

    if lambda_grid is None:
        lambda_grid = default_lambda_grid()
    lambda_grid = np.asarray(lambda_grid, dtype=float)

    Z0_train = Y0[:T0_train, :]
    Z1_train = Y1[:T0_train]
    Z0_val = Y0[T0_train:T0_total, :]
    Z1_val = Y1[T0_train:T0_total]

    if outer_loss_window == "validation":
        Z0_outer, Z1_outer = Z0_val, Z1_val
    else:
        Z0_outer, Z1_outer = Z0_train, Z1_train

    P = X0.shape[0]
    v20 = default_v20(X0)
    bounds = [(0.0, None)] * (P - 1)

    outer_curve = np.full(lambda_grid.size, np.nan)
    val_curve = np.full(lambda_grid.size, np.nan)
    v_path = np.zeros((lambda_grid.size, P))

    best_val = np.inf
    best_v = np.concatenate([[1.0], v20])
    best_lambda = float(lambda_grid[0])

    for idx, lam in enumerate(lambda_grid):
        res = minimize(
            outer_loss,
            x0=v20,
            args=(X1, X0, Z1_outer, Z0_outer, float(lam), solver),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_outer_iter), "ftol": float(ftol)},
        )
        v2_hat = np.clip(res.x, 0.0, None)
        outer_curve[idx] = float(res.fun)
        val_curve[idx] = selection_mse(v2_hat, X1, X0, Z1_val, Z0_val,
                                       solver=solver)
        v_path[idx, :] = np.concatenate([[1.0], v2_hat])
        if val_curve[idx] < best_val:
            best_val = val_curve[idx]
            best_lambda = float(lam)
            best_v = v_path[idx, :].copy()

    return best_v, best_lambda, lambda_grid, outer_curve, val_curve, v_path


def recover_w(
    v: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    solver: Any = None,
) -> np.ndarray:
    """Final donor-weight recovery at the selected V-weights."""
    return solve_w(v, X1, X0, solver=solver)
