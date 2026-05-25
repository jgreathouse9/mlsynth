"""L2-relaxation estimation (Shi & Wang 2024).

Solves the L2-relaxation primal (their Eq. 3)

    min_beta  (1/2) ||beta||_2^2   s.t.   || eta_hat - Sigma_hat beta ||_inf <= tau,

with ``Sigma_hat = X'X / T1`` and ``eta_hat = X'y / T1`` over the pre-period,
intercept ``alpha_hat = mean(y) - mean(X beta)`` (zero-mean residuals), and OOS
prediction ``y_hat_t = alpha_hat + x_t' beta``. The tuning parameter ``tau`` is
chosen by sequential out-of-sample validation on the tail of the training
window.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cvxpy as cp

from ....exceptions import MlsynthEstimationError


def l2_relax(y_pre: np.ndarray, X_pre: np.ndarray, tau: float) -> Tuple[np.ndarray, float]:
    """Solve the L2-relaxation primal for coefficients and intercept."""
    T1 = X_pre.shape[0]
    Sigma = (X_pre.T @ X_pre) / T1
    eta = (X_pre.T @ y_pre) / T1
    beta = cp.Variable(X_pre.shape[1])
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(beta)),
        [cp.norm(eta - Sigma @ beta, "inf") <= tau],
    )
    for solver in ("CLARABEL", "OSQP", "ECOS"):
        try:
            prob.solve(solver=solver)
            if beta.value is not None:
                break
        except Exception:
            continue
    if beta.value is None:
        raise MlsynthEstimationError("L2-relaxation failed: all solvers diverged.")
    b = np.asarray(beta.value).ravel()
    intercept = float(np.mean(y_pre) - np.mean(X_pre @ b))
    return b, intercept


def cross_validate_tau(
    y_pre: np.ndarray, X_pre: np.ndarray, val_frac: float = 0.2,
    n_coarse: int = 40, n_fine: int = 40,
) -> float:
    """Sequential OOS validation for tau (coarse grid, then zoom)."""
    T1 = y_pre.shape[0]
    n_val = max(2, int(round(val_frac * T1)))
    yt, Xt = y_pre[:-n_val], X_pre[:-n_val]
    yv, Xv = y_pre[-n_val:], X_pre[-n_val:]
    eta = (Xt.T @ yt) / Xt.shape[0]
    tau_max = float(np.max(np.abs(eta)))

    def val_mse(tau: float) -> float:
        try:
            b, a = l2_relax(yt, Xt, tau)
        except MlsynthEstimationError:
            return np.inf
        return float(np.mean((yv - (Xv @ b + a)) ** 2))

    coarse = np.linspace(1e-4 * tau_max, tau_max, n_coarse)
    best = min(coarse, key=val_mse)
    width = tau_max / n_coarse
    fine = np.linspace(max(0.0, best - width), best + width, n_fine)
    return float(min(fine, key=val_mse))


def fit_l2(
    y: np.ndarray, X: np.ndarray, T0: int, tau: Optional[float] = None
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Fit L2-relaxation; return ``(beta, intercept, counterfactual, tau)``."""
    y_pre, X_pre = y[:T0], X[:T0]
    tau_used = cross_validate_tau(y_pre, X_pre) if tau is None else float(tau)
    beta, intercept = l2_relax(y_pre, X_pre, tau_used)
    counterfactual = X @ beta + intercept
    return beta, intercept, counterfactual, tau_used
