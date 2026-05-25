"""Synthetic-control weight solvers for SCMO (NumPy/cvxpy, no estutils.Opt).

The simplex solver replaces ``Opt.SCopt(scm_model_type="SIMPLEX")``: it
minimizes the squared imbalance between the treated and donor matching
vectors subject to the convex-combination constraint (weights >= 0, sum 1),
exactly the Tian-Lee-Panchenko / Abadie program.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp


def simplex_weights(Z_treated: np.ndarray, Z_donors: np.ndarray) -> np.ndarray:
    """Convex (simplex) SC weights minimizing ``||Z_treated - Z_donors' w||^2``.

    Parameters
    ----------
    Z_treated : np.ndarray
        Treated matching vector, shape ``(P,)``.
    Z_donors : np.ndarray
        Donor matching matrix, shape ``(J, P)``.

    Returns
    -------
    np.ndarray
        Donor weights, shape ``(J,)``; non-negative and summing to one.
    """
    J = Z_donors.shape[0]
    w = cp.Variable(J)
    objective = cp.Minimize(cp.sum_squares(Z_treated - Z_donors.T @ w))
    cp.Problem(objective, [w >= 0, cp.sum(w) == 1]).solve(
        solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000
    )
    w_hat = np.clip(np.asarray(w.value).ravel(), 0.0, None)
    total = w_hat.sum()
    return w_hat / total if total > 0 else w_hat            # exact simplex (sum == 1)
