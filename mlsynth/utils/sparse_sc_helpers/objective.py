"""Outer V-objective for SparseSC, with closed-form gradient.

The outer objective is

    L(v_2; lam) = (1/T_outer) * ||Z1 - Z0 w*(v)||^2  +  lam * ||v||_1

where ``v = [1; v_2]`` (the first predictor is pinned at 1 to break the
positive-scale invariance of the inner simplex QP, as argued in
Vives-i-Bastida (2023) Appendix 6.1) and ``w*(v)`` solves the inner
simplex QP

    min_w  w' H(v) w  -  2 g(v)' w     s.t.  1' w = 1, w >= 0,

with ``H(v) = X0' diag(v) X0`` and ``g(v) = X0' diag(v) X1``.

Two outer windows are supported, controlled at call sites via the
``Z1, Z0`` arguments: pass the validation block to match Algorithm 1
in the paper; pass the training block to match the MATLAB driver.

This module provides three callables:

* ``outer_loss``      -- the loss alone (kept for back-compat).
* ``selection_mse``   -- the unpenalised validation-block MSE used to
                         choose lambda.
* ``outer_loss_and_grad`` -- ``(loss, grad)`` with the closed-form
                         envelope-theorem gradient.

The closed-form gradient avoids the ``2(P-1)``-evaluation finite-
difference cost that L-BFGS-B otherwise incurs per outer step. The
derivation is in the module docstring of ``optimization.py``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .inner import solve_w

# Donors with w_i smaller than this are treated as inactive when
# building the reduced KKT system. Setting this too tight (1e-12) makes
# the active set sensitive to solver noise; too loose (1e-3) loses
# active donors that genuinely carry a small amount of weight.
ACTIVE_TOL = 1e-7


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

    Pass the validation block to match the paper's Algorithm 1; pass
    the training block to match the MATLAB driver.
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
    """Unpenalised validation-block MSE used to select lambda."""
    v = np.concatenate([[1.0], v2])
    w = solve_w(v, X1, X0, solver=solver)
    residual = Z1_val - Z0_val @ w
    return float(np.mean(residual ** 2))


def outer_loss_and_grad(
    v2: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    lam: float,
    solver: Any = None,
    active_tol: float = ACTIVE_TOL,
) -> Tuple[float, np.ndarray]:
    """Return ``(loss, grad)`` of the outer V-objective w.r.t. ``v_2``.

    Implements the envelope-theorem gradient described in this module's
    docstring. The active set is recovered from the inner solution by
    thresholding on ``active_tol``. A single ``(|A|+1) x (|A|+1)``
    KKT solve gives the adjoint, after which all ``P-1`` gradient
    components are computed in ``O(P |A|)`` work.

    The L1 part contributes ``+lam`` per coordinate (right-derivative
    under the ``v_2 >= 0`` bound L-BFGS-B already enforces).

    Falls back to lstsq if the reduced KKT matrix is numerically
    singular (which can happen when the same donor appears at multiple
    predictor rows).
    """
    v2 = np.asarray(v2, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    Z1 = np.asarray(Z1, dtype=float)
    Z0 = np.asarray(Z0, dtype=float)

    P, _ = X0.shape
    v = np.concatenate([[1.0], v2])

    w = solve_w(v, X1, X0, solver=solver)
    r_outer = Z1 - Z0 @ w
    T_outer = max(int(Z1.shape[0]), 1)
    loss = float(np.mean(r_outer ** 2) + lam * np.sum(np.abs(v)))

    active = np.flatnonzero(w > active_tol)
    if active.size == 0:
        active = np.array([int(np.argmax(w))], dtype=int)

    XA = X0[:, active]                       # (P, |A|)
    ZA = Z0[:, active]                       # (T_outer, |A|)
    nA = active.size
    w_A = w[active]

    # Reduced KKT system
    #   [ 2 H_AA   1_A ] [w_A]   [2 g_A]
    #   [ 1_A^T   0   ] [mu ] = [1   ]
    # with H_AA = XA' diag(v) XA. Build it then add a trace-scaled
    # ridge for numerical robustness.
    HAA = XA.T @ (v[:, None] * XA)
    HAA = 0.5 * (HAA + HAA.T)
    diag_mean = np.trace(HAA) / nA if nA > 0 else 1.0
    if diag_mean <= 0:
        diag_mean = 1.0
    HAA = HAA + 1e-12 * diag_mean * np.eye(nA)

    M_kkt = np.zeros((nA + 1, nA + 1))
    M_kkt[:nA, :nA] = 2.0 * HAA
    M_kkt[:nA, nA] = 1.0
    M_kkt[nA, :nA] = 1.0

    # Adjoint RHS: [Z0_A' r_outer ; 0]. Then z_aug = M_kkt^{-1} RHS
    # and dL/dv_k = -(4/T_outer) * r_k * (X0[k, A] @ z) where
    # r_k = X1[k] - X0[k, A] w_A and z = z_aug[:|A|].
    b_aug = np.concatenate([ZA.T @ r_outer, [0.0]])
    try:
        z_aug = np.linalg.solve(M_kkt, b_aug)
    except np.linalg.LinAlgError:
        z_aug = np.linalg.lstsq(M_kkt, b_aug, rcond=None)[0]
    z = z_aug[:nA]

    # Per-predictor pre-fit residual r_k = X1[k] - X0[k, :] w
    r_pred = X1 - X0 @ w                     # (P,)
    Xz = XA @ z                              # (P,)
    grad_smooth_v = -(4.0 / T_outer) * r_pred * Xz   # (P,)

    # L1 contribution: +lam per coordinate (right-derivative under v>=0)
    grad_l1_v = lam * np.ones(P)

    grad_v2 = (grad_smooth_v + grad_l1_v)[1:]        # drop anchor entry
    return loss, grad_v2


# Backwards-compatible aliases retained so existing helper imports
# keep working.
training_loss = outer_loss
validation_mse = selection_mse
