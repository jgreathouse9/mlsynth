import cvxpy as cp
import numpy as np
from typing import List, Optional

def _build_objective(
    X_E: np.ndarray,
    v: cp.Variable,
    treated_vec: np.ndarray,
    lambda_penalty: float
):
    """
    Construct the quadratic objective for synthetic control weights.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_est, N)
        Standardized feature matrix for estimation period.
    v : cp.Variable, shape (N,)
        CVXPY variable representing control weights to optimize.
    treated_vec : np.ndarray, shape (T_est,)
        Treated unit time series over estimation period.
    lambda_penalty : float
        Weight for regularization term penalizing control units that poorly match treated series.

    Returns
    -------
    objective : cp.Expression
        CVXPY expression representing the objective function:
        ||X_E @ v - treated_vec||^2 + lambda_penalty * sum_j v_j ||X_E[:, j] - treated_vec||^2

    Notes
    -----
    - `match_term` encourages the synthetic control to track the treated series.
    - `penalty_term` discourages high weights on poorly matching control units.
    """
    match_term = cp.sum_squares(X_E @ v - treated_vec)

    penalty_term = lambda_penalty * cp.sum([
        v[j] * cp.sum_squares(X_E[:, j] - treated_vec)
        for j in range(X_E.shape[1])
    ])

    return match_term + penalty_term


def _build_constraints(
    v: cp.Variable,
    treated_idx: List[int]
):
    """
    Construct constraints for synthetic control QP.

    Parameters
    ----------
    v : cp.Variable, shape (N,)
        CVXPY variable representing control weights.
    treated_idx : list of int
        Indices of treated units that should have zero weight in control.

    Returns
    -------
    constraints : list of cp.Constraint
        - Nonnegativity: v >= 0
        - Sum-to-one: sum(v) == 1
        - Exclusion: v[treated_idx] == 0
    """
    constraints = [
        v >= 0,
        cp.sum(v) == 1
    ]

    for j in treated_idx:
        constraints.append(v[j] == 0)

    return constraints


def _solve_qp_problem(objective, constraints) -> Optional[np.ndarray]:
    """
    Solve a CVXPY quadratic program and return optimal control weights.

    Parameters
    ----------
    objective : cp.Expression
        Quadratic objective function for synthetic control.
    constraints : list of cp.Constraint
        List of CVXPY constraints (nonnegativity, sum-to-one, exclusion of treated units).

    Returns
    -------
    solution : np.ndarray, shape (N,) or None
        Optimal weights vector for control units. Returns None if solver fails.

    Notes
    -----
    - Uses OSQP solver with tight absolute and relative tolerances (eps_abs=eps_rel=1e-6).
    - The returned vector corresponds to the variable `v` in the original problem.
    """
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6)

    if prob.variables()[0].value is None:
        return None

    return np.asarray(prob.variables()[0].value).flatten()

def solve_control_qp(
    X_E: np.ndarray,
    treated_vec: np.ndarray,
    treated_idx: List[int],
    lambda_penalty: float = 0.1
) -> Optional[np.ndarray]:
    """
    Solve quadratic program to compute synthetic control weights for non-treated units.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_est, N)
        Standardized feature matrix over the estimation period.
    treated_vec : np.ndarray, shape (T_est,)
        Vector of treated unit values over estimation period.
    treated_idx : list of int
        Indices of treated units to exclude from control.
    lambda_penalty : float, optional
        Regularization weight penalizing control units that poorly match treated series (default 0.1).

    Returns
    -------
    v : np.ndarray, shape (N,)
        Optimal synthetic control weights for all units. Zeros at treated indices.
        Returns None if solver fails.

    Notes
    -----
    - The optimization solves:
        min_v ||X_E @ v - treated_vec||^2 + lambda_penalty * sum_j v_j ||X_E[:, j] - treated_vec||^2
      subject to:
        v >= 0, sum(v) = 1, v[treated_idx] = 0
    - Uses OSQP solver via CVXPY.
    """
    _, N = X_E.shape
    v = cp.Variable(N)

    objective = _build_objective(X_E, v, treated_vec, lambda_penalty)
    constraints = _build_constraints(v, treated_idx)

    solution = _solve_qp_problem(objective, constraints)

    return solution



def compute_nmse(X: np.ndarray, w: np.ndarray, target: np.ndarray, idx: np.ndarray, treated_idx: list) -> float:
    """
    Compute normalized mean squared error (NMSE) between synthetic treated series and target series.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix over all time periods.
    w : np.ndarray, shape (len(treated_idx),)
        Weights for treated units.
    target : np.ndarray, shape (T,)
        Observed target series (treated unit mean or outcome).
    idx : np.ndarray
        Time indices to compute NMSE over (e.g., backcast/validation period).
    treated_idx : list of int
        Indices of treated units corresponding to weights w.

    Returns
    -------
    nmse : float
        Normalized mean squared error: mean((synthetic - target)^2 / var(target)).

    Notes
    -----
    - Normalization is done using per-time variance across all outcome columns.
    - Used for evaluating synthetic control fit over baseline (pre-treatment) periods.
    """
    synth = X[idx][:, treated_idx] @ w
    tgt = target[idx]
    denom = np.var(X[idx, :len(target)], axis=1) + 1e-8  # variance over Y units
    return float(np.mean(((synth - tgt) ** 2) / denom))


def compute_effect_series(X: np.ndarray, treated_idx: list, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute synthetic treatment effect time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix over all time periods.
    treated_idx : list of int
        Indices of treated units.
    w : np.ndarray, shape (len(treated_idx),)
        Weights for treated units.
    v : np.ndarray, shape (N,)
        Synthetic control weights for all units.

    Returns
    -------
    effects : np.ndarray, shape (T,)
        Time series of estimated treatment effects:
        synthetic treated series minus synthetic control series.

    Notes
    -----
    - Synthetic treated series: X[:, treated_idx] @ w
    - Synthetic control series: X @ v
    """
    return X[:, treated_idx] @ w - X @ v
