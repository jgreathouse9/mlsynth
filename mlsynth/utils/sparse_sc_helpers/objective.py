"""Training loss and validation MSE for SparseSC.

``training_loss`` is the MATLAB ``loss_function.m``: pre-period
mean-squared outcome error on the *training block* plus an L1 penalty
on the V-weights. ``validation_mse`` is the MATLAB ``mse.m``: pre-
period MSE on the *validation block*, no penalty.

Both routines reconstruct ``v = [1, v2]`` (first weight pinned to 1)
and call the inner W-weight QP from :mod:`inner`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .inner import solve_w


def training_loss(
    v2: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1_train: np.ndarray,
    Z0_train: np.ndarray,
    lam: float,
    solver: Any = None,
) -> float:
    """Equivalent to MATLAB ``loss_function.m``."""
    v = np.concatenate([[1.0], v2])
    w = solve_w(v, X1, X0, solver=solver)
    residual = Z1_train - Z0_train @ w
    return float(np.mean(residual ** 2) + lam * np.sum(np.abs(v)))


def validation_mse(
    v2: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1_val: np.ndarray,
    Z0_val: np.ndarray,
    solver: Any = None,
) -> float:
    """Equivalent to MATLAB ``mse.m``."""
    v = np.concatenate([[1.0], v2])
    w = solve_w(v, X1, X0, solver=solver)
    residual = Z1_val - Z0_val @ w
    return float(np.mean(residual ** 2))
