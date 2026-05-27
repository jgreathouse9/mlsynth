"""Kernel smoothing and the SHC/ASHC quadratic programs.

Self-contained SHC primitives, relocated from the legacy ``estutils`` module so
the Synthetic Historical Control estimator no longer depends on it:

* :func:`smooth` -- local-linear (Gaussian-kernel) smoother for the treated
  unit's pre-period, recovering its latent trend;
* :func:`loocv_bandwidth` -- leave-one-out bandwidth selection for the smoother;
* :func:`solve_shc_qp` -- the SHC (convex-hull) / ASHC (ridge-augmented) QP;
* :func:`tune_lambda_ashc` -- holdout tuning of the ASHC ridge parameter.

This is a leaf module (numpy / cvxpy / scipy only) so it can be imported from
both the SHC orchestrator and the donor selector without import cycles.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh


def smooth(y_pre, bw):
    """Local-linear Gaussian-kernel smoother of ``y_pre`` at bandwidth ``bw``."""
    T_pre = len(y_pre)
    smoothed = np.zeros(T_pre)
    for i in range(T_pre):
        w = np.exp(-0.5 * ((np.arange(T_pre) - i) / bw) ** 2)
        w /= w.sum()
        X = np.vstack([np.ones(T_pre), np.arange(T_pre) - i]).T
        W = np.diag(w)
        beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y_pre
        smoothed[i] = beta[0]
    return smoothed


def loocv_bandwidth(y_pre, bandwidth_grid):
    """Leave-one-out CV choice of smoother bandwidth over ``bandwidth_grid``."""
    T_pre = len(y_pre)
    cv_errors = []
    for h in bandwidth_grid:
        errors = []
        for i in range(T_pre):
            y_train = np.delete(y_pre, i)
            idx = np.arange(T_pre) != i
            w = np.exp(-0.5 * ((np.where(idx)[0] - i) / h) ** 2)
            w /= w.sum()
            X = np.vstack([np.ones(T_pre - 1), np.where(idx)[0] - i]).T
            W = np.diag(w)
            beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y_train
            y_pred = beta[0]
            errors.append((y_pre[i] - y_pred) ** 2)
        cv_errors.append(np.mean(errors))
    best_h = bandwidth_grid[np.argmin(cv_errors)]
    return best_h, cv_errors


def solve_shc_qp(L, ell_eval, use_augmented=False, w_shc=None, lam=None,
                 varsigma=1e-6, tol=1e-8):
    """Solve the SHC (convex-hull) or ASHC (ridge-augmented) quadratic program.

    Parameters
    ----------
    L : np.ndarray
        Donor matrix (m x N).
    ell_eval : np.ndarray
        Evaluation (latent-trend) vector (m,).
    use_augmented : bool
        If True solve ASHC (ridge toward ``w_shc`` with strength ``lam``);
        otherwise the simplex-constrained SHC.
    w_shc : np.ndarray, optional
        SHC weight vector (required for ASHC).
    lam : float, optional
        ASHC ridge parameter (required for ASHC).
    varsigma, tol : float
        Low-variance-direction penalty weight and eigenvalue threshold.

    Returns
    -------
    (w_opt, obj_val) : tuple
        Optimal weights and objective value (``(None, None)`` if infeasible).
    """
    N = L.shape[1]
    w = cp.Variable(N)

    if use_augmented:
        if lam is None or w_shc is None:
            raise ValueError("lam and w_shc must be provided for ASHC.")
        fit_term = cp.sum_squares(ell_eval - L @ w)
        deviation = (1 / (2 * lam)) * cp.sum_squares(w - w_shc)
    else:
        fit_term = cp.sum_squares(ell_eval - L @ w)
        deviation = 0

    G = L.T @ L
    eigvals, eigvecs = eigh(G)
    C = eigvecs[:, eigvals < tol]
    penalty = varsigma * cp.sum_squares(C.T @ w) if C.size > 0 else 0

    objective = cp.Minimize(fit_term + deviation + penalty)
    constraints = [cp.sum(w) == 1]
    if not use_augmented:
        constraints.append(w >= 0)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    return (w.value, prob.value) if w.value is not None else (None, None)


def tune_lambda_ashc(L, ell_eval, w_shc, lambda_grid=None, split_ratio=0.5):
    """Holdout-validate the ASHC ridge parameter ``lambda`` over ``lambda_grid``."""
    m = len(ell_eval)
    train_size = int(split_ratio * m)
    ell_train, ell_val = ell_eval[:train_size], ell_eval[train_size:]
    L_train, L_val = L[:train_size, :], L[train_size:, :]

    if lambda_grid is None:
        lambda_grid = np.logspace(-6, 2, 50)

    lambda_errors = {}
    for lam in lambda_grid:
        w_hat, _ = solve_shc_qp(L_train, ell_train, use_augmented=True,
                                w_shc=w_shc, lam=lam)
        if w_hat is not None:
            mse = np.mean((ell_val - L_val @ w_hat) ** 2)
            lambda_errors[lam] = mse

    best_lambda = min(lambda_errors, key=lambda_errors.get)
    return best_lambda, lambda_errors


# Backwards-compatible alias for the historical private name.
_solve_SHC_QP = solve_shc_qp
