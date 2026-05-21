"""Outer V-objective and unpenalized MSE for SparseSC.

Two outer-objective windows are supported, controlled at call sites
via the ``Z1, Z0`` arguments:

* "validation" (default, matches Vives-i-Bastida (2023) Algorithm 1
  page 5): the outer V-weights minimize the validation-block outcome
  MSE plus an L1 penalty on V.
* "training" (matches the unpublished MATLAB driver
  ``loss_function.m``): the outer V-weights minimize the training-
  block outcome MSE plus an L1 penalty on V.

The two choices reflect a documented disagreement between the paper
and the MATLAB code that has circulated privately; see
``mlsynth.estimators.SparseSC`` for context. ``selection_mse`` is
the unpenalized validation-block MSE used to pick lambda regardless
of which window the outer objective uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .inner import solve_w


def outer_loss(
    v2: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    lam: float,
    solver: Any = None,
) -> float:
    """Outer V-objective: ``mean((Z1 - Z0 w(v))^2) + lam * ||v||_1``.

    Pass the validation block (``Z1_val``, ``Z0_val``) to match the
    paper's Algorithm 1; pass the training block to match the MATLAB
    driver.
    """
    v = np.concatenate([[1.0], v2])
    w = solve_w(v, X1, X0, solver=solver)
    residual = Z1 - Z0 @ w
    return float(np.mean(residual ** 2) + lam * np.sum(np.abs(v)))


def selection_mse(
    v2: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1_val: np.ndarray,
    Z0_val: np.ndarray,
    solver: Any = None,
) -> float:
    """Unpenalized validation-block MSE used to pick lambda."""
    v = np.concatenate([[1.0], v2])
    w = solve_w(v, X1, X0, solver=solver)
    residual = Z1_val - Z0_val @ w
    return float(np.mean(residual ** 2))


# Backwards-compatible aliases retained so existing helper imports
# keep working. Both call sites are updated to the new names below.
training_loss = outer_loss
validation_mse = selection_mse
