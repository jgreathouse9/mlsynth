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

from typing import Any, Optional, Tuple

import numpy as np
from scipy.linalg import null_space

from ...exceptions import MlsynthDataError
from .inner import WarmStart, solve_w

# Donors with w_i smaller than this are treated as inactive when
# building the reduced KKT system. Setting this too tight (1e-12) makes
# the active set sensitive to solver noise; too loose (1e-3) loses
# active donors that genuinely carry a small amount of weight.
ACTIVE_TOL = 1e-7


def outcome_scale(Y1: np.ndarray, Y0: np.ndarray, T0_total: int) -> float:
    """The pooled standard deviation of every unit's pre-treatment outcome.

    The outer objective trades an outcome MSE against ``lam * ||v||_1`` on a grid in
    absolute units, so the sweep divides the outcome blocks by this to make the grid
    relative to the outcome's own variation: a change of units then leaves the selection
    unchanged (#933). The level drops out with the mean; the post-period does not enter.
    An outcome with no pre-treatment variation has scale 1, since every MSE is then zero.
    """
    Y = np.column_stack([np.asarray(Y1, float).reshape(-1, 1), np.asarray(Y0, float)])[:int(T0_total)]
    if not np.all(np.isfinite(Y)):
        raise MlsynthDataError("SparseSC: the pre-treatment outcome has non-finite values, so it has "
                               "no scale to put the penalty grid on.")
    s = float(np.std(Y))
    return s if s > 0.0 else 1.0


def outer_loss(
    v2: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    lam: float,
    solver: Any = None,
    warm: Optional[WarmStart] = None,
) -> float:
    """Outer V-objective: ``mean((Z1 - Z0 w(v))^2) + lam * ||v||_1``.

    Pass the validation block to match the paper's Algorithm 1; pass
    the training block to match the MATLAB driver. ``warm`` carries the
    inner answer from one call to the next.
    """
    v = np.concatenate([[1.0], v2])
    w = solve_w(v, X1, X0, solver=solver, warm=warm)
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
    warm: Optional[WarmStart] = None,
) -> Tuple[float, np.ndarray]:
    """Return ``(loss, grad)`` of the outer V-objective w.r.t. ``v_2``.

    Implements the envelope-theorem gradient described in this module's
    docstring. The active set is recovered from the inner solution by
    thresholding on ``active_tol``. A single ``(|A|+1) x (|A|+1)``
    KKT solve gives the adjoint, after which all ``P-1`` gradient
    components are computed in ``O(P |A|)`` work.

    The L1 part contributes ``+lam`` per coordinate (right-derivative
    under the ``v_2 >= 0`` bound L-BFGS-B already enforces).

    The adjoint ``z`` solves the reduced KKT system
    ``2 H_AA z + mu 1 = b, 1' z = 0``. With ``z = N y`` for an orthonormal
    basis ``N`` of ``1' z = 0`` it is ``y = (1/2) (D N)^+ (D N)^+' N' b`` with
    ``D = V^(1/2) (X0_A - X1 1')``, computed from the SVD of ``D N``: the
    Hessian is never formed, the level of each predictor drops out (on
    ``1' z = 0``, ``X0_A z = (X0_A - X1 1') z``), and a rank-deficient active
    set gets the minimum-norm adjoint, with rank decided by
    ``max(shape) eps`` times the largest singular value (agents_solvers.md,
    rules 5 and 6). This replaced a Gram-form solve with a 1e-12 ridge (#934).
    """
    v2 = np.asarray(v2, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    Z1 = np.asarray(Z1, dtype=float)
    Z0 = np.asarray(Z0, dtype=float)

    P, _ = X0.shape
    v = np.concatenate([[1.0], v2])

    w = solve_w(v, X1, X0, solver=solver, warm=warm)
    r_outer = Z1 - Z0 @ w
    T_outer = max(int(Z1.shape[0]), 1)
    loss = float(np.mean(r_outer ** 2) + lam * np.sum(np.abs(v)))

    active = np.flatnonzero(w > active_tol)
    if active.size == 0:
        active = np.array([int(np.argmax(w))], dtype=int)

    XA = X0[:, active]                       # (P, |A|)
    ZA = Z0[:, active]                       # (T_outer, |A|)
    nA = active.size

    b = ZA.T @ r_outer                       # adjoint right-hand side
    if nA > 1:
        N = null_space(np.ones((1, nA)))     # (|A|, |A| - 1), orthonormal
        DN = (np.sqrt(v)[:, None] * (XA - X1[:, None])) @ N
        _, S, Vt = np.linalg.svd(DN, full_matrices=False)
        keep = S > max(DN.shape) * np.finfo(float).eps * (S[0] if S.size else 0.0)
        y = 0.5 * (Vt[keep].T @ ((Vt[keep] @ (N.T @ b)) / S[keep] ** 2))
        z = N @ y
    else:
        z = np.zeros(nA)                     # one active donor: w is pinned locally

    # Per-predictor pre-fit residual r_k = X1[k] - X0[k, :] w
    r_pred = X1 - X0 @ w                     # (P,)
    Xz = (XA - X1[:, None]) @ z              # (P,); equals XA @ z since 1'z = 0
    grad_smooth_v = -(4.0 / T_outer) * r_pred * Xz   # (P,)

    # L1 contribution: +lam per coordinate (right-derivative under v>=0)
    grad_l1_v = lam * np.ones(P)

    grad_v2 = (grad_smooth_v + grad_l1_v)[1:]        # drop anchor entry
    return loss, grad_v2


# Backwards-compatible aliases retained so existing helper imports
# keep working.
training_loss = outer_loss
validation_mse = selection_mse
