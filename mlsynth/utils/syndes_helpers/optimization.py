"""Solver-facing optimization utilities for SYNDES."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

from ..fast_scm_helpers.structure import IndexSet
from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .formulation import build_syndes_problem_components, unpack_problem_components
from .structures import SYNDESDesign, SYNDESSolveOutcome

#: cvxpy statuses that come with an incumbent. SCIP's ``timelimit`` and
#: ``gaplimit`` both arrive as ``optimal_inaccurate``: the incumbent is accepted
#: as a design -- a feasible design is a design -- and the solve's own outcome is
#: recorded on it and warned about (:class:`SYNDESSolveOutcome`).
_OPTIMAL_STATUSES = {
    "optimal",
    "optimal_inaccurate",
    "user_limit",
    "user_limit_inaccurate",
}

#: SCIP's terminal statuses that mean a limit, not a proof, ended the solve.
_SCIP_LIMIT_STATUSES = {
    "timelimit", "gaplimit", "nodelimit", "totalnodelimit", "stallnodelimit",
    "bestsollimit", "sollimit", "memlimit", "restartlimit", "userinterrupt",
}

#: How far a binary may sit from 0 or 1 and still be read as one: SCIP's
#: default feasibility tolerance (``numerics/feastol``).
_INTEGRALITY_TOL = 1e-6


class SYNDESSolveWarning(UserWarning):
    """A SYNDES mixed-integer solve ended without proving its design optimal."""


def _rounded_assignment(D_value: np.ndarray, tol: float = _INTEGRALITY_TOL) -> np.ndarray:
    """The 0/1 assignment the solver returned, refused if it is not one.

    ``D`` is dimensionless, so the tolerance is absolute: SCIP's feasibility
    tolerance on a binary.
    """
    D = np.asarray(D_value, dtype=float).reshape(-1)
    rounded = np.rint(D)
    worst = float(np.max(np.abs(D - rounded))) if D.size else 0.0
    if worst > tol or not np.isin(rounded, (0.0, 1.0)).all():
        raise MlsynthEstimationError(
            f"SYNDES optimization returned an assignment that is not integral: an "
            f"entry lies {worst:.3g} from the nearest of 0 and 1, beyond the "
            f"solver's feasibility tolerance {tol:g}.")
    return rounded.astype(int)


def _gap(objective: float, bound: float) -> float:
    """Relative optimality gap, ``(objective - bound) / max(|objective|, |bound|)``.

    Relative to the larger of the two, with no floor, so it does not depend on
    the units of the outcome. Zero when both are zero.
    """
    scale = max(abs(float(objective)), abs(float(bound)))
    if scale == 0.0:
        return 0.0
    return (float(objective) - float(bound)) / scale


def _refit(Y: np.ndarray, assignment: np.ndarray, mode: str, lam: float,
           restrictions: Optional[Any]) -> Tuple[Dict[str, Any], float]:
    """Weights and objective for a fixed treated set.

    The same program with ``D`` fixed to ``assignment``: its McCormick rows are
    then exact, so this is the convex QP of the design's weights, and its
    optimum is what the rounded assignment attains. The donor-side restrictions
    apply unchanged.
    """
    N = assignment.shape[0]
    D = cp.Variable(N)
    components = build_syndes_problem_components(
        Y=Y, D=D, K=int(assignment.sum()), lam=lam, mode=mode)
    constraints = list(components.constraints) + [D == assignment.astype(float)]
    if restrictions is not None and not restrictions.is_empty:
        from .restrictions import donor_constraints
        constraints += donor_constraints(
            mode, components.variables, D, restrictions.donor_exclusion)
    problem = cp.Problem(cp.Minimize(components.objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception as exc:
        raise MlsynthEstimationError(
            f"SYNDES refit for the rounded assignment failed: {exc}") from exc
    if problem.status != "optimal" or problem.value is None:
        raise MlsynthEstimationError(
            f"SYNDES refit for the rounded assignment failed: {problem.status}.")
    values = {name: var.value for name, var in components.variables.items()}
    return values, float(problem.value)


def _scip_limits(solver: Any, gap_limit: Optional[float],
                 time_limit: Optional[float]) -> Tuple[Dict[str, float], bool]:
    """SCIP parameters for the limits, and whether the limits reach the solver."""
    params: Dict[str, float] = {}
    if gap_limit is not None:
        params["limits/gap"] = float(gap_limit)
    if time_limit is not None:
        params["limits/time"] = float(time_limit)
    is_scip = str(solver).upper() == "SCIP"
    requested = gap_limit is not None or time_limit is not None
    return (params if is_scip else {}), (is_scip or not requested)


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


def _simplex_within_tolerance(c: Any, excluded: np.ndarray,
                              tol: float = 1e-7) -> np.ndarray:
    """A solver's simplex weights, repaired only within its tolerance.

    ``excluded`` marks the units the weights must not use (the treated units,
    ``c <= 1 - D``). Entries there, and entries below zero, are set to zero when
    they are within ``tol`` of it -- the weights are dimensionless and sum to
    one -- and the vector is rescaled to sum to one. Anything further outside is
    a failed solve and raises.
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    excluded = np.asarray(excluded, dtype=bool).reshape(-1)
    off = float(np.max(np.abs(c[excluded]))) if excluded.any() else 0.0
    if c.min() < -tol or off > tol or abs(c.sum() - 1.0) > tol:
        raise MlsynthEstimationError(
            f"SYNDES control weights left the simplex beyond the solver's "
            f"tolerance: min {c.min():.3g}, weight on treated units {off:.3g}, "
            f"sum {c.sum():.12g}.")
    c = np.where(excluded, 0.0, np.clip(c, 0.0, None))
    return c / c.sum()


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
            control_weights = _simplex_within_tolerance(c, assignment == 1)
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
    forbidden_sets: Optional[list] = None,
    restrictions: Optional[Any] = None,
    warm_start_D: Optional[np.ndarray] = None,
    objective_lower_bound: Optional[float] = None,
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

    # No-good cuts (solution pool): forbid each previously-chosen treated set
    # ``S`` via ``sum_{i in S} D_i <= |S|-1``, so re-solving returns the next-best
    # *distinct* design. Empty by default -> the single-optimum solve, unchanged.
    prob_constraints = list(components.constraints)
    for _fs in (forbidden_sets or []):
        _fs = np.asarray(_fs, dtype=int).reshape(-1)
        if _fs.size:
            prob_constraints.append(cp.sum(D[_fs]) <= int(_fs.size) - 1)

    # Design restrictions (geography / clustering / size / forcing) as linear
    # constraints on the assignment vector D, plus donor-side exclusions that
    # couple D to the mode's control weights.
    if restrictions is not None and not restrictions.is_empty:
        from .restrictions import apply_restrictions, donor_constraints
        prob_constraints += apply_restrictions(D, restrictions)
        prob_constraints += donor_constraints(
            mode, components.variables, D, restrictions.donor_exclusion,
        )

    problem = cp.Problem(cp.Minimize(components.objective), prob_constraints)

    # SCIP runs through ``miqp_accel.solve_warm_cut``, which builds the same
    # model cvxpy builds and keeps it, so SCIP's own status, dual bound and gap
    # can be read: cvxpy reports a time-limited incumbent and a proven optimum
    # alike as ``optimal_inaccurate``. With no warm start and no cut it adds
    # nothing to the model. The limits reach SCIP only; ``limits_applied``
    # records whether they reached the solver.
    scip_params, limits_applied = _scip_limits(solver, gap_limit, time_limit)
    is_scip = str(solver).upper() == "SCIP"
    accel_info = None
    use_accel = is_scip and (warm_start_D is not None
                             or objective_lower_bound is not None)
    try:
        if is_scip:
            from ..miqp_accel import solve_warm_cut
            warm_bits = (None if warm_start_D is None
                         else np.asarray(warm_start_D, dtype=float).reshape(-1))
            _, accel_info = solve_warm_cut(
                problem, D, warm_bits=warm_bits,
                objective_lower_bound=objective_lower_bound,
                gap_limit=gap_limit, time_limit=time_limit, verbose=verbose,
            )
        else:
            problem.solve(solver=solver, verbose=verbose)
    except Exception as exc:
        raise MlsynthEstimationError(
            f"SYNDES optimization failed to solve: {exc}"
        ) from exc

    if problem.status not in _OPTIMAL_STATUSES:
        msg = f"SYNDES optimization failed: {problem.status}"
        if (restrictions is not None and not restrictions.is_empty
                and "infeasible" in str(problem.status).lower()):
            # Point the user at the most likely cause rather than a bare status:
            # the active design restrictions are jointly unsatisfiable for this K.
            msg = (
                "SYNDES optimization is infeasible under the active design "
                "restrictions (spillover/adjacency conflict, stratum quotas, "
                f"forced/forbidden units) for K={K}. No size-K treated set "
                "satisfies them -- relax the restrictions or reduce K."
            )
        raise MlsynthEstimationError(msg)

    if D.value is None:
        raise MlsynthEstimationError("SYNDES optimization did not return an assignment.")

    # ----------------------------
    # core outputs
    # ----------------------------
    assignment = _rounded_assignment(D.value)
    selected_indices = np.where(assignment == 1)[0]
    solver_objective = float(problem.value)
    values, objective = _refit(Y, assignment, mode, lam_value, restrictions)
    outcome = _solve_outcome(problem, accel_info, solver_objective, objective,
                             limits_applied)
    if outcome.stopped_at_limit or outcome.status != "optimal":
        warnings.warn(_outcome_message(outcome), SYNDESSolveWarning, stacklevel=2)

    # labels
    if unit_index is not None:
        selected_labels = unit_index.get_labels(selected_indices)
        all_labels = unit_index.labels
    else:
        selected_labels = selected_indices
        all_labels = np.arange(N)

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
    if use_accel:
        raw_results["accel"] = {
            "status": accel_info.status,
            "dual_bound": accel_info.dual_bound,
            "gap": accel_info.gap,
            "solve_time": accel_info.solve_time,
            "cut_applied": accel_info.cut_applied,
            "warm_applied": accel_info.warm_applied,
            "fell_back": accel_info.fell_back,
            "objective_lower_bound": objective_lower_bound,
        }

    return SYNDESDesign(
        mode=mode,
        objective_value=objective,
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
        solve_outcome=outcome,
    )


def _solve_outcome(problem: cp.Problem, accel_info: Any, solver_objective: float,
                   objective: float, limits_applied: bool) -> SYNDESSolveOutcome:
    """The solve's outcome, with SCIP's bound moved into the objective's units.

    SCIP bounds the canonical objective ``c'x``; cvxpy adds the objective's
    constant afterwards (the one-way objective carries ``lam / K``). The
    constant is the difference between the objective at SCIP's point and SCIP's
    primal bound there, and it is added to the dual bound.
    """
    if accel_info is None:
        status = str(problem.status)
        return SYNDESSolveOutcome(
            status=status, solver_status=status,
            stopped_at_limit=status.startswith("user_limit"),
            objective=objective, solver_objective=solver_objective,
            dual_bound=None, gap=None, solve_time=None,
            limits_applied=limits_applied)
    dual_bound = None
    gap = None
    if np.isfinite(accel_info.dual_bound) and np.isfinite(accel_info.primal_bound):
        dual_bound = float(accel_info.dual_bound) + (
            solver_objective - float(accel_info.primal_bound))
        gap = _gap(objective, dual_bound)
    return SYNDESSolveOutcome(
        status=str(accel_info.status), solver_status=str(problem.status),
        stopped_at_limit=str(accel_info.status) in _SCIP_LIMIT_STATUSES,
        objective=objective, solver_objective=solver_objective,
        dual_bound=dual_bound, gap=gap, solve_time=float(accel_info.solve_time),
        limits_applied=limits_applied)


def _outcome_message(outcome: SYNDESSolveOutcome) -> str:
    limit = {"timelimit": "the time limit", "gaplimit": "the gap limit"}.get(
        outcome.status, f"status {outcome.status!r}")
    gap = "unknown" if outcome.gap is None else f"{outcome.gap:.3g}"
    return (
        f"SYNDES's mixed-integer solve stopped at {limit} without proving its "
        f"design optimal; the design is the best one found, with a relative gap "
        f"of {gap} to the solver's lower bound. Raise time_limit, lower "
        f"gap_limit, or read design.solve_outcome.")


def solve_synthetic_design_pool(
    Y: np.ndarray,
    K: int,
    top_K: int = 1,
    *,
    mode: str = "global_2way",
    lam: Optional[float] = None,
    solver: Any = "SCIP",
    verbose: bool = False,
    unit_index: Optional[IndexSet] = None,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    gap_limit: Optional[float] = None,
    time_limit: Optional[float] = None,
    restrictions: Optional[Any] = None,
) -> list:
    """Top-``top_K`` SYNDES designs, ranked by MSE, via no-good cuts.

    Re-solves the SYNDES MIP ``top_K`` times; after each solve the chosen treated
    set is forbidden (``sum_{i in S} D_i <= |S|-1``) so the next solve returns the
    next-best *distinct* design. The returned list is ranked by objective value
    (non-decreasing), and ``solve_synthetic_design_pool(..., top_K=1)`` returns
    ``[solve_synthetic_design(...)]`` -- the single optimum unchanged. Stops early
    (returning fewer than ``top_K``) once the feasible region is exhausted.

    Parameters
    ----------
    top_K : int
        Number of distinct designs to enumerate (``>= 1``).
    (all other parameters as in :func:`solve_synthetic_design`).

    Returns
    -------
    list of SYNDESDesign
        Distinct designs ordered by ascending MSE; ``[0]`` is the global optimum.
    """
    if not isinstance(top_K, (int, np.integer)) or top_K < 1:
        raise MlsynthConfigError(f"top_K must be a positive integer; got {top_K!r}.")

    designs: list = []
    forbidden: list = []
    for _ in range(int(top_K)):
        try:
            d = solve_synthetic_design(
                Y=Y, K=K, mode=mode, lam=lam, solver=solver, verbose=verbose,
                unit_index=unit_index, costs=costs, budget=budget,
                gap_limit=gap_limit, time_limit=time_limit,
                forbidden_sets=forbidden, restrictions=restrictions,
            )
        except MlsynthEstimationError as exc:
            # Only an infeasible solve means the region is exhausted. Anything
            # else -- a missing solver, a failed refit, a solver error -- is a
            # failure, and an empty or short pool would hide it.
            if "infeasible" not in str(exc).lower():
                raise
            break  # feasible region exhausted (no further distinct design)
        designs.append(d)
        forbidden.append(np.asarray(d.selected_unit_indices, dtype=int))
    return designs
