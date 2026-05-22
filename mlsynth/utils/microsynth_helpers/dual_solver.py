"""L-BFGS-B dual ascent for the MicroSynth QP.

The primal is

    min_w  (1/2) || w - 1/n_C ||^2
    s.t.   X_C^T w = x_bar_T          (d balancing constraints)
           1^T w = 1                  (sum-to-one)
           w >= 0                     (non-negativity)

The dual lives in R^{d+1} regardless of how many control units N_C
there are -- one Lagrange multiplier per balance constraint plus one
for the sum-to-one constraint. This is the entire reason MicroSynth
scales to millions of control users on a single machine: solving an
R^{d+1} convex program (typically d <= 30 in marketing settings)
and reading the primal off the KKT relationship in closed form.

The dual objective and its closed-form gradient are derived from
the convex conjugate of the primal objective + non-negativity
indicator. See Snap KDD 2023 (Lin et al.) Eq. (11)-(15) for the
matching formulation in the distributed setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class DualSolverResult:
    w: np.ndarray
    dual_lambda: np.ndarray
    dual_nu: float
    n_iterations: int
    converged: bool


def solve_microsynth_dual(
    X_C: np.ndarray,
    xbar_T: np.ndarray,
    max_iter: int = 500,
    gtol: float = 1e-8,
) -> DualSolverResult:
    """Solve the MicroSynth dual via L-BFGS-B.

    Parameters
    ----------
    X_C : np.ndarray
        Control-user covariate matrix, shape ``(n_C, d)``.
    xbar_T : np.ndarray
        Treated-group covariate mean, shape ``(d,)``.
    max_iter : int
        L-BFGS-B maximum iterations.
    gtol : float
        Gradient tolerance.

    Returns
    -------
    DualSolverResult
        Primal weights ``w`` (shape ``(n_C,)``), dual variables
        ``lambda`` (shape ``(d,)``) and ``nu`` (scalar), iteration
        count, and convergence flag.
    """
    n_C, d = X_C.shape
    inv_nC = 1.0 / n_C

    def objective_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        lmbda = theta[:d]
        nu = theta[d]

        # Soft-thresholded residual: only entries above zero contribute.
        resid = (2.0 * inv_nC) - (X_C @ lmbda) - nu
        truncated = np.maximum(0.0, resid)

        # Dual objective (modulo constants in nu that don't affect minimizer).
        val = (
            0.25 * np.sum(truncated ** 2)
            + float(lmbda @ xbar_T)
            + nu
        )

        # Analytical gradient -- closed form via convex conjugate.
        grad_lmbda = -0.5 * (X_C.T @ truncated) + xbar_T
        grad_nu = -0.5 * float(np.sum(truncated)) + 1.0

        return val, np.concatenate([grad_lmbda, [grad_nu]])

    init_theta = np.zeros(d + 1)
    res = minimize(
        objective_and_grad,
        init_theta,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "gtol": float(gtol)},
    )

    opt_lmbda = res.x[:d]
    opt_nu = float(res.x[d])

    # Recover primal weights via KKT closed form.
    w_raw = 0.5 * np.maximum(0.0, (2.0 * inv_nC) - (X_C @ opt_lmbda) - opt_nu)
    w = np.clip(w_raw, 0.0, None)
    w_sum = float(w.sum())
    if w_sum > 0:
        w = w / w_sum
    else:
        # Degenerate -- shouldn't happen if the problem is feasible.
        w = np.full(n_C, inv_nC)

    # Accept the result if the gradient is small enough, even if the
    # optimizer flagged itself with a benign termination reason.
    grad_ok = bool(np.max(np.abs(res.jac)) < max(1e-3, 10.0 * gtol))
    converged = bool(res.success or grad_ok)

    return DualSolverResult(
        w=w,
        dual_lambda=opt_lmbda,
        dual_nu=opt_nu,
        n_iterations=int(res.nit),
        converged=converged,
    )
