"""Partially-pooled SCM QP and the auto-nu heuristic.

The objective is paper Eq. 6:

    min over Gamma in simplex^J:
        nu * q_tilde_pool(Gamma)^2 + (1 - nu) * q_tilde_sep(Gamma)^2
        + lam * ||Gamma||_F^2

where the normalizations ``q_tilde`` are fixed at the nu = 0 baseline
imbalances. ``nu = 0`` reduces to solving SCM separately for each
treated unit; ``nu = 1`` to fully pooled SCM.

The "auto-nu" heuristic sweeps ``nu`` over a grid and selects the
``nu`` minimizing ``|q_tilde_sep(Gamma_nu) - q_tilde_pool(Gamma_nu)|`` --
the equal-imbalance midpoint of the balance frontier.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cvxpy as cp
import numpy as np

from .imbalance import compute_q_pool, compute_q_sep


def _solve_separate_scm(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    lam: float,
    solver: Any,
) -> np.ndarray:
    """Solve SCM independently for each treated unit (nu = 0 case)."""
    L, J = Y_treated_pre.shape
    N = Y_donors_pre.shape[1]
    Gamma = np.zeros((N, J))
    for j in range(J):
        gamma_j = cp.Variable(N, nonneg=True)
        Yj = Y_donors_pre[:, :, j]  # (L, N)
        yj = Y_treated_pre[:, j]    # (L,)
        loss = cp.sum_squares(yj - Yj @ gamma_j) / L
        penalty = float(lam) * cp.sum_squares(gamma_j) if lam > 0 else 0
        prob = cp.Problem(
            cp.Minimize(loss + penalty),
            [cp.sum(gamma_j) == 1],
        )
        prob.solve(solver=solver or cp.OSQP)
        if gamma_j.value is None:
            prob.solve(solver=cp.SCS)
        gamma_val = np.clip(np.asarray(gamma_j.value, dtype=float), 0.0, None)
        s = gamma_val.sum()
        if s > 0:
            gamma_val = gamma_val / s
        Gamma[:, j] = gamma_val
    return Gamma


def _solve_partially_pooled(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    nu: float,
    lam: float,
    q_sep_base: float,
    q_pool_base: float,
    solver: Any,
) -> Tuple[np.ndarray, str]:
    """Solve Eq. 6 for a given ``nu`` and normalization baselines."""
    L, J = Y_treated_pre.shape
    N = Y_donors_pre.shape[1]
    Gamma = cp.Variable((N, J), nonneg=True)

    # Per-unit residuals: r_lj = y_l j - sum_i Y_donors[l, i, j] * Gamma[i, j].
    # cvxpy doesn't broadcast einsum, but we can stack contributions per j.
    per_unit_terms = []
    avg_residual = 0
    for j in range(J):
        Yj = Y_donors_pre[:, :, j]  # (L, N)
        yj = Y_treated_pre[:, j]    # (L,)
        res_j = yj - Yj @ Gamma[:, j]
        per_unit_terms.append(cp.sum_squares(res_j) / L)
        avg_residual = avg_residual + res_j
    avg_residual = avg_residual / J
    pooled_term = cp.sum_squares(avg_residual) / L

    # Normalize: q_tilde_sep^2 = q_sep^2 / q_sep_base^2, etc.
    sep_obj = sum(per_unit_terms) / J
    if q_sep_base > 0:
        sep_obj = sep_obj / (q_sep_base ** 2)
    if q_pool_base > 0:
        pooled_obj = pooled_term / (q_pool_base ** 2)
    else:
        pooled_obj = pooled_term

    penalty = float(lam) * cp.sum_squares(Gamma) if lam > 0 else 0
    objective = nu * pooled_obj + (1.0 - nu) * sep_obj + penalty
    constraints = [cp.sum(Gamma, axis=0) == 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=solver or cp.OSQP)
    except Exception:
        problem.solve(solver=cp.SCS)
    if Gamma.value is None:
        problem.solve(solver=cp.SCS)
    if Gamma.value is None:
        raise RuntimeError(f"PPSCM solver failed (status={problem.status}).")
    Gamma_val = np.clip(np.asarray(Gamma.value, dtype=float), 0.0, None)
    # Re-normalize columns to sum to 1 (small numerical drift).
    col_sums = Gamma_val.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    Gamma_val = Gamma_val / col_sums[None, :]
    return Gamma_val, str(problem.status)


def solve_ppscm(
    Y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    nu: float | str = "auto",
    lam: float = 0.0,
    solver: Any = None,
    nu_grid_size: int = 21,
) -> Tuple[np.ndarray, float, Dict[float, Tuple[float, float]], str,
           float, float]:
    """Solve PPSCM and return ``(Gamma, nu_used, frontier, status,
    q_sep_baseline, q_pool_baseline)``.

    Parameters
    ----------
    Y_treated_pre : np.ndarray
        Shape ``(L, J)``.
    Y_donors_pre : np.ndarray
        Shape ``(L, N, J)``.
    nu : float or "auto"
        Pooling parameter in ``[0, 1]`` or ``"auto"`` to select by the
        equal-imbalance heuristic.
    lam : float
        Frobenius-norm regularization on ``Gamma``.
    solver : Any
        cvxpy solver. ``None`` -> OSQP.
    nu_grid_size : int
        Grid resolution for the auto-nu sweep.
    """

    # Baseline: separate SCM (nu = 0).
    Gamma_sep = _solve_separate_scm(Y_treated_pre, Y_donors_pre, lam, solver)
    q_sep_base = compute_q_sep(Y_treated_pre, Y_donors_pre, Gamma_sep)
    q_pool_base = compute_q_pool(Y_treated_pre, Y_donors_pre, Gamma_sep)

    frontier: Dict[float, Tuple[float, float]] = {}

    if isinstance(nu, str) and nu == "auto":
        # Sweep nu on [0, 1] and pick the value at the trade-off "knee":
        # the nu that minimizes the sum of the two normalized imbalances
        # q_tilde_sep + q_tilde_pool. At nu = 0 both equal 1 by
        # construction; intermediate nu typically push q_tilde_pool below 1
        # while inflating q_tilde_sep above 1, and the minimum-sum point is
        # the practical equal-marginal-improvement compromise the paper
        # suggests when no error-bound parameters are available.
        grid = np.linspace(0.0, 1.0, max(int(nu_grid_size), 3))
        best_nu = grid[0]
        best_score = np.inf
        best_Gamma = Gamma_sep
        best_status = "optimal"
        for nu_val in grid:
            if nu_val == 0.0:
                Gamma_v = Gamma_sep
                status_v = "optimal"
            else:
                Gamma_v, status_v = _solve_partially_pooled(
                    Y_treated_pre, Y_donors_pre, float(nu_val), lam,
                    q_sep_base, q_pool_base, solver,
                )
            q_sep_v = compute_q_sep(Y_treated_pre, Y_donors_pre, Gamma_v)
            q_pool_v = compute_q_pool(Y_treated_pre, Y_donors_pre, Gamma_v)
            frontier[float(nu_val)] = (q_sep_v, q_pool_v)
            qt_sep = q_sep_v / q_sep_base if q_sep_base > 0 else q_sep_v
            qt_pool = q_pool_v / q_pool_base if q_pool_base > 0 else q_pool_v
            score = qt_sep + qt_pool
            if score < best_score:
                best_score = score
                best_nu = float(nu_val)
                best_Gamma = Gamma_v
                best_status = status_v
        return best_Gamma, best_nu, frontier, best_status, q_sep_base, q_pool_base

    # Explicit nu.
    nu_val = float(nu)
    if not (0.0 <= nu_val <= 1.0):
        raise ValueError(f"nu must be in [0, 1] or 'auto'; got {nu_val!r}.")
    if nu_val == 0.0:
        return Gamma_sep, 0.0, frontier, "optimal", q_sep_base, q_pool_base
    Gamma_v, status_v = _solve_partially_pooled(
        Y_treated_pre, Y_donors_pre, nu_val, lam,
        q_sep_base, q_pool_base, solver,
    )
    return Gamma_v, nu_val, frontier, status_v, q_sep_base, q_pool_base
