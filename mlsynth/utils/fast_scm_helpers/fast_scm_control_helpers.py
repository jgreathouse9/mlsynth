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
    Construct the synthetic control quadratic objective.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_E, N)
        Standardized feature matrix over the estimation period.
    v : cp.Variable, shape (N,)
        Optimization variable representing control weights.
    treated_vec : np.ndarray, shape (T_E,)
        Synthetic treated series over the estimation period.
    lambda_penalty : float
        Regularization strength for unit-specific mismatch penalties.

    Returns
    -------
    objective : cp.Expression
        CVXPY expression representing:
            ||X_E v - treated_vec||^2
            + lambda_penalty * sum_j v_j ||X_E[:, j] - treated_vec||^2

    Notes
    -----
    - First term enforces global fit to the treated series.
    - Second term penalizes assigning weight to units that individually
      deviate from the treated trajectory.
    - The penalty is linear in `v`, preserving convexity.
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
    Construct feasibility constraints for synthetic control weights.

    Parameters
    ----------
    v : cp.Variable, shape (N,)
        Control weight vector.
    treated_idx : list of int
        Indices of treated units to exclude from the donor pool.

    Returns
    -------
    constraints : list of cp.Constraint
        Constraints enforcing:
        - Nonnegativity: v >= 0
        - Convex combination: sum(v) == 1
        - Exclusion: v[j] = 0 for j in treated_idx

    Notes
    -----
    - Ensures the synthetic control is a convex combination of donor units.
    - Explicit exclusion prevents contamination from treated units.
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
    Solve a convex quadratic program using CVXPY.

    Parameters
    ----------
    objective : cp.Expression
        Objective function to minimize.
    constraints : list of cp.Constraint
        Feasibility constraints.

    Returns
    -------
    solution : np.ndarray, shape (N,) or None
        Optimal variable values, or None if the solver fails.

    Notes
    -----
    - Uses the OSQP solver with tight tolerances.
    - Returns None if no solution is found (caller must handle).
    - Assumes the problem is convex and well-posed.
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
    Solve for synthetic control weights given a treated trajectory.

    Parameters
    ----------
    X_E : np.ndarray, shape (T_E, N)
        Standardized feature matrix over the estimation period.
    treated_vec : np.ndarray, shape (T_E,)
        Synthetic treated series constructed from selected units.
    treated_idx : list of int
        Indices of treated units to exclude from the control.
    lambda_penalty : float, default=0.1
        Regularization strength for mismatch penalties.

    Returns
    -------
    v : np.ndarray, shape (N,) or None
        Synthetic control weights over all units. Entries at treated
        indices are zero. Returns None if optimization fails.

    Notes
    -----
    - Solves a convex QP via CVXPY.
    - The solution represents a convex combination of donor units.
    - Used downstream to construct synthetic control trajectories.
    """
    _, N = X_E.shape
    v = cp.Variable(N)

    objective = _build_objective(X_E, v, treated_vec, lambda_penalty)
    constraints = _build_constraints(v, treated_idx)

    solution = _solve_qp_problem(objective, constraints)

    return solution



def compute_nmse(X: np.ndarray, w: np.ndarray, target: np.ndarray, idx: np.ndarray, treated_idx: list) -> float:
    """
    Compute normalized mean squared error (NMSE) over a time subset.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix.
    w : np.ndarray, shape (k,)
        Weights for treated units.
    target : np.ndarray, shape (T,)
        Target series (e.g., weighted average of outcome units).
    idx : np.ndarray
        Time indices over which to evaluate the error.
    treated_idx : list of int
        Indices of treated units corresponding to `w`.

    Returns
    -------
    nmse : float
        Mean squared error normalized by per-time cross-sectional variance.

    Notes
    -----
    - Synthetic series: X[:, treated_idx] @ w
    - Normalization uses variance across outcome columns at each time step.
    - A small constant is added to avoid division by zero.
    - This normalization emphasizes relative fit rather than scale.
    """
    synth = X[idx][:, treated_idx] @ w
    tgt = target[idx]
    denom = np.var(X[idx, :len(target)], axis=1) + 1e-8  # variance over Y units
    return float(np.mean(((synth - tgt) ** 2) / denom))


def compute_effect_series(X: np.ndarray, treated_idx: list, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the estimated treatment effect time series.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Full feature matrix.
    treated_idx : list of int
        Indices of treated units.
    w : np.ndarray, shape (k,)
        Weights defining the synthetic treated unit.
    v : np.ndarray, shape (N,)
        Synthetic control weights.

    Returns
    -------
    effects : np.ndarray, shape (T,)
        Estimated treatment effect at each time:
            (synthetic treated) - (synthetic control)

    Notes
    -----
    - Synthetic treated: X[:, treated_idx] @ w
    - Synthetic control: X @ v
    - Positive values indicate treated > control.
    """
    return X[:, treated_idx] @ w - X @ v
