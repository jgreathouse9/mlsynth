"""Per-period weight solver for Dynamic Synthetic Control.

Implements equations (2.7) and (2.17-2.18) of Zheng & Chen (2024):

1. **QP step** (the convex-hull feasibility check). Solve

   .. math::

      \\min_w  (Z_1 - Z_0 w)' V (Z_1 - Z_0 w) + \\big( \\sum_i w_i - 1 \\big)^2
      \\qquad
      \\text{s.t.}  \\sum w_i = 1,\\ 0 \\le w_i \\le 1.

   This always has a solution and gives a feasible starting point.

2. **EL refinement** (the empirical-likelihood step). When the QP
   solution has mean ``|Z_1 - Z_0 w| <= eps``, refine by

   .. math::

      \\max_w  \\prod_i w_i
      \\qquad
      \\text{s.t.}  Z_0 w = Z_1,\\ \\sum w_i = 1,\\ 0 \\le w_i \\le 1.

   This is the empirical-likelihood maximisation that gives DSC its
   asymptotic-theory guarantees (Theorem 1 of the paper). When the EL
   step diverges, fail open: use the QP solution.

The R reference (``Dynamic_Synthetic_Control_new.R``) uses
``LowRankQP`` for step 1 and ``NlcOptim::solnl`` for step 2; we
mirror those with cvxpy and ``scipy.optimize.minimize`` respectively.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from ...exceptions import MlsynthEstimationError


def _qp_step(
    Z1: np.ndarray, Z0: np.ndarray, V_diag: np.ndarray,
) -> np.ndarray:
    """Convex-hull QP for the initial feasible point.

    Parameters
    ----------
    Z1 : np.ndarray
        Length-``k`` treated unit's match target (covariates + lagged
        outcome stacked).
    Z0 : np.ndarray
        Shape ``(k, N_co)`` donor matching targets.
    V_diag : np.ndarray
        Length-``k`` per-variable importance weights (the diagonal of
        the V matrix). Replaces the SC method's cross-validated V with
        per-period OLS-coefficient magnitudes, per the paper.

    Returns
    -------
    np.ndarray
        Length-``N_co`` simplex weight vector.
    """
    k, N_co = Z0.shape
    w = cp.Variable(N_co, nonneg=True)
    # Diagonal-weighted quadratic loss with a small sum-to-one penalty
    # (matching the R reference's `P = 1 1'` term).
    diff = Z1 - Z0 @ w
    loss = cp.sum_squares(cp.multiply(np.sqrt(V_diag), diff)) + cp.square(cp.sum(w) - 1.0)
    prob = cp.Problem(cp.Minimize(loss), [cp.sum(w) == 1])
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:
        raise MlsynthEstimationError(
            f"DSC: per-period QP solver failed ({exc})."
        ) from exc
    if w.value is None:
        raise MlsynthEstimationError(
            f"DSC: per-period QP returned no solution (status={prob.status!r})."
        )
    out = np.clip(np.asarray(w.value).flatten(), 0.0, None)
    s = out.sum()
    return out / s if s > 0 else out


def _el_refine(
    w_init: np.ndarray, Z1: np.ndarray, Z0: np.ndarray,
    *, max_iter: int = 200, tol: float = 1e-9,
) -> Optional[np.ndarray]:
    """Empirical-likelihood refinement (equation 2.7).

    Maximises :math:`\\prod_i w_i` (equivalently, minimises
    :math:`-\\sum_i \\log w_i`) subject to ``Z0 w = Z1`` and
    ``sum(w) = 1``, with ``0 <= w <= 1``.

    Returns
    -------
    np.ndarray | None
        Length-``N_co`` refined weights, or ``None`` if the solver
        fails / never enters the interior.
    """
    N_co = w_init.size

    # Small positive floor so the log objective stays finite.
    w_start = np.maximum(w_init, 1e-8)
    w_start = w_start / w_start.sum()

    def neg_log_prod(w: np.ndarray) -> float:
        # -sum log(w); clip below to keep autograd-free objective stable.
        return -float(np.sum(np.log(np.maximum(w, 1e-12))))

    def grad(w: np.ndarray) -> np.ndarray:
        return -1.0 / np.maximum(w, 1e-12)

    constraints = [
        {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
    ]
    for k in range(Z0.shape[0]):
        Z0_row = Z0[k]
        Z1_k = float(Z1[k])
        constraints.append(
            {"type": "eq", "fun": (lambda w, r=Z0_row, t=Z1_k: float(r @ w - t))}
        )
    bounds = [(0.0, 1.0)] * N_co

    try:
        res = minimize(
            neg_log_prod, w_start, jac=grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": max_iter, "ftol": tol},
        )
    except Exception:
        return None
    if not res.success or res.x is None:
        return None
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    if s <= 0:
        return None
    return w / s


def solve_dsc_weights(
    Z1: np.ndarray,
    Z0: np.ndarray,
    V_diag: np.ndarray,
    *,
    el_tolerance: float = 1e-2,
) -> Tuple[np.ndarray, bool]:
    """One-period DSC weight solve: QP feasibility + EL refinement.

    Parameters
    ----------
    Z1 : np.ndarray
        Length-``k`` treated targets (per-period covariates + lagged
        outcome).
    Z0 : np.ndarray
        Shape ``(k, N_co)`` donor targets.
    V_diag : np.ndarray
        Length-``k`` variable-importance diagonal for the QP.
    el_tolerance : float
        Maximum mean absolute mismatch ``|Z_1 - Z_0 w|`` at which the
        EL refinement step is attempted. (R reference uses ``0.01``.)
        When the QP residual exceeds this, the EL step is skipped and
        the QP weights are returned.

    Returns
    -------
    weights : np.ndarray
        Length-``N_co`` simplex weight vector.
    used_el : bool
        ``True`` if EL refinement succeeded and was used; ``False``
        otherwise (QP fallback).
    """
    if V_diag.size != Z1.size or Z0.shape[0] != Z1.size:
        raise ValueError(
            f"DSC: Z1/Z0/V shape mismatch: Z1 {Z1.shape}, Z0 {Z0.shape}, "
            f"V {V_diag.shape}."
        )
    w_qp = _qp_step(Z1, Z0, V_diag)
    mismatch = float(np.mean(np.abs(Z1 - Z0 @ w_qp)))
    if mismatch > el_tolerance:
        return w_qp, False
    w_el = _el_refine(w_qp, Z1, Z0)
    if w_el is None:
        return w_qp, False
    return w_el, True


def variable_importance(
    Y: np.ndarray, X: np.ndarray, Y_lag1: np.ndarray, T0: int,
) -> np.ndarray:
    """Per-period OLS-coefficient magnitudes used as the V diagonal.

    For each time ``t`` in ``1..T``, regress ``Y[:, t]`` on
    ``[Y_lag1[:, t], X[:, t, :]]`` over the panel and return
    ``|coefficients|`` (ignoring the intercept). Mirrors the R
    reference's ``paramt`` matrix.

    Parameters
    ----------
    Y, X, Y_lag1 : np.ndarray
        Outcome, covariate cube, and lagged outcome (see :class:`DSCARInputs`).
    T0 : int
        Pre-period length. For ``t <= T0`` the full panel is used;
        for ``t > T0`` only the donor rows are used (the treated rows
        carry the post-treatment outcomes, which would contaminate the
        coefficient estimate).

    Returns
    -------
    np.ndarray
        Shape ``(T, 1 + p)`` per-period absolute coefficient matrix.
        Column ``0`` is ``|rho_t|`` (lagged outcome); columns
        ``1..p`` are ``|beta_t|`` (exogenous covariates).
    """
    N, T = Y.shape
    p = X.shape[2]
    V = np.zeros((T, 1 + p))
    for t in range(T):
        if t == 0:
            # No lag at t = 0 -> use only X[:, 0, :] and intercept.
            x_cols = [np.ones(N), *(X[:, 0, k] for k in range(p))]
            design = np.column_stack(x_cols)
            target = Y[:, 0]
            mask = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
            if mask.sum() < design.shape[1] + 1:
                continue
            try:
                beta, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
            except np.linalg.LinAlgError:
                continue
            # V_diag column 0 stays 0 (no lag at t=0); cols 1..p are |beta_k|.
            V[t, 1:] = np.abs(beta[1:])
            continue
        cols = [np.ones(N), Y_lag1[:, t]]
        for k in range(p):
            cols.append(X[:, t, k])
        design = np.column_stack(cols)
        target = Y[:, t]
        mask = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
        if mask.sum() < design.shape[1] + 1:
            continue
        try:
            beta, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
        except np.linalg.LinAlgError:
            continue
        # beta = (intercept, rho_t, beta_1, ..., beta_p); strip intercept.
        V[t, :] = np.abs(beta[1:])
    return V
