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


def _shc_objective(L, ell_eval, w, C, use_augmented, w_shc, lam, varsigma):
    """The SHC/ASHC objective value at ``w`` (matches the cvxpy formulation)."""
    obj = float(np.sum((ell_eval - L @ w) ** 2))
    if use_augmented:
        obj += float((1.0 / (2.0 * lam)) * np.sum((w - w_shc) ** 2))
    if C.size > 0:
        obj += float(varsigma * np.sum((C.T @ w) ** 2))
    return obj


def _solve_shc_qp_osqp(P, q, N, nonneg):
    """Solve ``min ½ wᵀP w + qᵀw`` s.t. ``1ᵀw = 1`` (and ``w ≥ 0`` if ``nonneg``)
    with a direct OSQP factorization. Returns ``w`` or ``None`` on failure.

    OSQP avoids cvxpy's per-call canonicalization overhead -- the SHC donor
    selector solves this micro-QP ~2000 times per fit, where that overhead, not
    the arithmetic, dominates.
    """
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError:  # pragma: no cover - osqp/scipy are hard deps
        return None
    rows = [np.ones((1, N))]
    lo, hi = [1.0], [1.0]
    if nonneg:
        rows.append(np.eye(N))
        lo += [0.0] * N
        hi += [np.inf] * N
    A = sp.csc_matrix(np.vstack(rows))
    P_csc = sp.triu(sp.csc_matrix(P), format="csc")
    prob = osqp.OSQP()
    try:
        prob.setup(P=P_csc, q=np.asarray(q, dtype=float), A=A,
                   l=np.asarray(lo, dtype=float), u=np.asarray(hi, dtype=float),
                   verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                   max_iter=20000, polish=True)
        res = prob.solve()
    except Exception:  # pragma: no cover - defensive
        return None
    status = getattr(res.info, "status_val", None)
    if status in (1, 2) and res.x is not None and np.all(np.isfinite(res.x)):
        return np.asarray(res.x, dtype=float)
    return None


def _solve_shc_qp_cvxpy(L, ell_eval, C, use_augmented, w_shc, lam, varsigma):
    """Reference cvxpy solve -- the robustness fallback for the OSQP path."""
    N = L.shape[1]
    w = cp.Variable(N)
    fit_term = cp.sum_squares(ell_eval - L @ w)
    deviation = (1 / (2 * lam)) * cp.sum_squares(w - w_shc) if use_augmented else 0
    penalty = varsigma * cp.sum_squares(C.T @ w) if C.size > 0 else 0
    constraints = [cp.sum(w) == 1]
    if not use_augmented:
        constraints.append(w >= 0)
    prob = cp.Problem(cp.Minimize(fit_term + deviation + penalty), constraints)
    prob.solve(solver=cp.CLARABEL)
    return (w.value, prob.value) if w.value is not None else (None, None)


def solve_shc_qp(L, ell_eval, use_augmented=False, w_shc=None, lam=None,
                 varsigma=1e-6, tol=1e-8):
    """Solve the SHC (convex-hull) or ASHC (ridge-augmented) quadratic program.

    .. math::

       \\min_w\\ \\lVert \\boldsymbol{\\ell} - L w\\rVert_2^2
         + \\mathbb{1}_{\\text{ASHC}}\\tfrac{1}{2\\lambda}\\lVert w - w_{\\text{shc}}\\rVert_2^2
         + \\varsigma\\lVert C^\\top w\\rVert_2^2
       \\quad\\text{s.t.}\\quad \\mathbf{1}^\\top w = 1,\\ \\
       w \\ge 0 \\ (\\text{SHC only}),

    where ``C`` spans the (near-)null directions of ``LᵀL``. Solved by a direct
    OSQP factorization (the SHC donor selector calls this thousands of times per
    fit, so cvxpy's per-call canonicalization overhead dominates); falls back to
    cvxpy if OSQP is unavailable or does not converge, so the result is
    unchanged.

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
    if use_augmented and (lam is None or w_shc is None):
        raise ValueError("lam and w_shc must be provided for ASHC.")

    N = L.shape[1]
    G = L.T @ L
    eigvals, eigvecs = eigh(G)
    C = eigvecs[:, eigvals < tol]

    # 0.5 wᵀP w + qᵀw form of the objective above.
    P = 2.0 * G
    q = -2.0 * (L.T @ ell_eval)
    if C.size > 0:
        P = P + (2.0 * varsigma) * (C @ C.T)
    if use_augmented:
        P = P + (1.0 / lam) * np.eye(N)
        q = q - (1.0 / lam) * np.asarray(w_shc, dtype=float)

    w_opt = _solve_shc_qp_osqp(P, q, N, nonneg=not use_augmented)
    if w_opt is None:                       # robustness fallback (no result drift)
        return _solve_shc_qp_cvxpy(L, ell_eval, C, use_augmented, w_shc, lam, varsigma)
    obj = _shc_objective(L, ell_eval, w_opt, C, use_augmented, w_shc, lam, varsigma)
    return w_opt, obj



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
