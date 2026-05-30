"""Solver-facing optimization utilities for SYNDES."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cvxpy as cp
import numpy as np

from ..fast_scm_helpers.structure import IndexSet
from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .formulation import build_syndes_problem_components, unpack_problem_components
from .structures import SYNDESDesign

_OPTIMAL_STATUSES = {
    "optimal",
    "optimal_inaccurate",
    # SCIP returns these when it hits gap_limit / time_limit with a valid
    # incumbent. Per Abadie & Zhao (2026, p. 13) a feasible non-optimal
    # solution is sufficient for the bias bounds, so we accept them.
    "user_limit",
    "user_limit_inaccurate",
}


def estimate_lambda(Y: np.ndarray) -> float:
    """
    Estimate SYNDES penalty parameter as average within-unit variance.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix of shape (T, N).

    Returns
    -------
    float
        Estimated lambda value.

    Raises
    ------
    MlsynthDataError
        If Y is not 2D.
    MlsynthConfigError
        If fewer than 2 time periods are provided.
    """

    if Y.ndim != 2:
        raise MlsynthDataError("Y must be a two-dimensional T x N matrix.")
    if Y.shape[0] < 2:
        raise MlsynthConfigError("At least two pre-treatment periods are required.")
    return float(np.mean(np.var(Y, axis=0, ddof=1)))


def _validate_design_inputs(Y: np.ndarray, K: Optional[int]) -> None:
    """
    Validate basic SYNDES design inputs.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape (T, N).
    K : int
        Number of treated units.

    Raises
    ------
    MlsynthConfigError
        If Y is not 2D, K is invalid, or K exceeds number of units.
    """
    if Y.ndim != 2:
        raise MlsynthConfigError("Y must be a two-dimensional T x N matrix.")
    if K is not None:
        if K <= 0:
            raise MlsynthConfigError("K must be a positive integer.")
        if K > Y.shape[1]:
            raise MlsynthConfigError("K cannot exceed the number of units.")


def _build_global_2way(Y: np.ndarray, D: cp.Variable, K: int, lam: float):
    """
    Construct global two-way SYNDES optimization components.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    D : cp.Variable
        Binary treatment assignment variable.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.

    Returns
    -------
    tuple
        (objective, constraints, variables)
    """

    components = build_syndes_problem_components(
        Y=Y, D=D, K=K, lam=lam, mode="global_2way"
    )
    return unpack_problem_components(components)


def _build_per_unit(Y: np.ndarray, D: cp.Variable, K: int, lam: float):
    """
    Construct per-unit SYNDES optimization components.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    D : cp.Variable
        Binary treatment assignment variable.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.

    Returns
    -------
    tuple
        (objective, constraints, variables)
    """

    components = build_syndes_problem_components(Y=Y, D=D, K=K, lam=lam, mode="per_unit")
    return unpack_problem_components(components)


def _extract_weights(mode, assignment, values):
    """
    Extract implied weights from solved SYNDES optimization.

    Parameters
    ----------
    mode : str
        SYNDES formulation mode.
    assignment : np.ndarray
        Binary treatment assignment vector of shape (N,).
    values : dict
        Dictionary of solved CVXPY variable values.

    Returns
    -------
    tuple of np.ndarray or (None, None, None)
        treated_weights, control_weights, contrast_weights
    """
    N = len(assignment)
    K = assignment.sum()

    w = values.get("w")
    q = values.get("q")

    # ----------------------------
    # One-way global: treated weights pinned to 1/K, control weights are the
    # free synthetic-control vector c solved by the MIP.
    # ----------------------------
    if mode == "global_equal_weights":
        treated_weights = assignment / K
        c = values.get("c")
        if c is None:
            control_weights = (1 - assignment) / (N - K)
        else:
            control_weights = np.clip(np.asarray(c, dtype=float).reshape(-1), 0.0, None)
        contrast_weights = treated_weights - control_weights
        return treated_weights, control_weights, contrast_weights

    # ----------------------------
    # Per-unit case: each treated unit has its own SC row in q.
    # The aggregate ATET-producing contrast is
    #   c_j = (1/K) (D_j - sum_i q[i, j])
    # which integrates the K per-unit estimators into one vector that
    # downstream inference can apply to a (T, N) panel.
    # ----------------------------
    if mode == "per_unit":
        if q is None:
            return None, None, None
        q_arr = np.asarray(q, dtype=float)
        contrast_weights = (assignment - q_arr.sum(axis=0)) / K
        return q_arr, None, contrast_weights

    # ----------------------------
    # Learned-weight global case (two-way global)
    # ----------------------------
    if w is not None and q is not None:
        treated_weights = q
        control_weights = w - q
        contrast_weights = 2 * q - w
        return treated_weights, control_weights, contrast_weights

    return None, None, None



def solve_synthetic_design(
    Y: np.ndarray,
    K: Optional[int],
    mode: str = "global_2way",
    lam: Optional[float] = None,
    solver: Any = "SCIP",
    verbose: bool = False,
    unit_index: Optional[IndexSet] = None,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    gap_limit: Optional[float] = None,
    time_limit: Optional[float] = None,
) -> SYNDESDesign:
    """
    Solve the SYNDES synthetic design optimization problem.

    Parameters
    ----------
    Y : np.ndarray
        Pre-treatment outcome matrix of shape (T, N).
    K : int
        Number of treated units.
    mode : {"global_2way", "global_equal_weights", "per_unit"}, optional
        SYNDES formulation type.
    lam : float, optional
        Regularization parameter. If None, estimated from Y.
    solver : Any, optional
        CVXPY-compatible solver specification.
    verbose : bool, optional
        Whether to enable solver verbosity.
    unit_index : IndexSet, optional
        Mapping from indices to unit labels.

    Returns
    -------
    SYNDESDesign
        Optimized design object.

    Raises
    ------
    MlsynthConfigError
        If inputs are invalid or lambda is negative.
    MlsynthEstimationError
        If optimization fails or is infeasible.
    """
    _validate_design_inputs(Y, K)

    lam_value = estimate_lambda(Y) if lam is None else float(lam)
    if lam_value < 0:
        raise MlsynthConfigError("lam must be nonnegative.")

    _, N = Y.shape
    D = cp.Variable(N, boolean=True)

    components = build_syndes_problem_components(
        Y=Y, D=D, K=K, lam=lam_value, mode=mode
    )

    # Budget constraint: total cost of treated units bounded above by
    # ``budget``. Paper Section 1 (line "enforce a budget constraint if
    # there is a varying cost to treat ...") motivates this. Both
    # arguments default to None (no constraint).
    if (costs is None) != (budget is None):
        raise MlsynthConfigError(
            "costs and budget must be supplied together (or both None)."
        )
    if costs is not None:
        cost_vec = np.asarray(costs, dtype=float).flatten()
        if cost_vec.shape[0] != N:
            raise MlsynthConfigError(
                f"costs must have length N={N}; got {cost_vec.shape[0]}."
            )
        if np.any(cost_vec < 0):
            raise MlsynthConfigError("costs must be non-negative.")
        if budget <= 0:
            raise MlsynthConfigError("budget must be strictly positive.")
        components = components.with_constraints([cost_vec @ D <= float(budget)])

    problem = cp.Problem(cp.Minimize(components.objective), components.constraints)

    # Plumb the user-supplied SCIP limits (Abadie & Zhao 2026 p. 13:
    # "we do not strictly require optimality of {w*, v*}, provided
    # {w*, v*} is feasible"). Only added when the solver is SCIP, since
    # ``scip_params`` is a SCIP-specific cvxpy kwarg.
    solve_kwargs: dict = {"solver": solver, "verbose": verbose}
    if str(solver).upper() == "SCIP":
        scip_params: dict = {}
        if gap_limit is not None:
            scip_params["limits/gap"] = float(gap_limit)
        if time_limit is not None:
            scip_params["limits/time"] = float(time_limit)
        if scip_params:
            solve_kwargs["scip_params"] = scip_params

    try:
        problem.solve(**solve_kwargs)
    except Exception as exc:
        raise MlsynthEstimationError(
            f"SYNDES optimization failed to solve: {exc}"
        ) from exc

    if problem.status not in _OPTIMAL_STATUSES:
        raise MlsynthEstimationError(f"SYNDES optimization failed: {problem.status}")

    if D.value is None:
        raise MlsynthEstimationError("SYNDES optimization did not return an assignment.")

    # ----------------------------
    # core outputs
    # ----------------------------
    assignment = (np.asarray(D.value).reshape(-1) >= 0.5).astype(int)
    selected_indices = np.where(assignment == 1)[0]

    # labels
    if unit_index is not None:
        selected_labels = unit_index.get_labels(selected_indices)
        all_labels = unit_index.labels
    else:
        selected_labels = selected_indices
        all_labels = np.arange(N)

    # raw solver outputs
    values = {name: var.value for name, var in components.variables.items()}

    # ----------------------------
    # NEW: unified weight extraction
    # ----------------------------
    treated_w, control_w, contrast_w = _extract_weights(mode, assignment, values)

    # Design "prediction": the pre-period treated-minus-synthetic contrast the
    # design balances, and its RMSE. Defined uniformly across modes via the
    # aggregate contrast vector.
    contrast_series = None
    pre_fit_rmse = None
    if contrast_w is not None:
        contrast_w_arr = np.asarray(contrast_w, dtype=float).reshape(-1)
        if contrast_w_arr.shape[0] == N:
            contrast_series = Y @ contrast_w_arr
            pre_fit_rmse = float(np.sqrt(np.mean(contrast_series ** 2)))

    raw_results: Dict[str, Any] = {
        "status": problem.status,
        "solver": solver,
    }

    return SYNDESDesign(
        mode=mode,
        objective_value=float(problem.value),
        lambda_value=lam_value,

        assignment=assignment,
        selected_unit_indices=selected_indices,
        selected_unit_labels=np.asarray(selected_labels),
        assignment_by_unit={
            label: int(val) for label, val in zip(all_labels, assignment)
        },

        w=values.get("w"),
        q=values.get("q"),
        z=values.get("z"),

        # ----------------------------
        # NEW FIELDS (important)
        # ----------------------------
        treated_weights=treated_w,
        control_weights=control_w,
        contrast_weights=contrast_w,
        contrast_series=contrast_series,
        pre_fit_rmse=pre_fit_rmse,

        raw_results=raw_results,
    )
