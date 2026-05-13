"""Solver-facing optimization utilities for SCDI."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cvxpy as cp
import numpy as np

from ..fast_scm_helpers.structure import IndexSet
from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .formulation import build_scdi_problem_components, unpack_problem_components
from .structures import SCDIDesign

_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def estimate_lambda(Y: np.ndarray) -> float:
    """Estimate the paper-recommended penalty as average within-unit variance."""

    if Y.ndim != 2:
        raise MlsynthDataError("Y must be a two-dimensional T x N matrix.")
    if Y.shape[0] < 2:
        raise MlsynthConfigError("At least two pre-treatment periods are required.")
    return float(np.mean(np.var(Y, axis=0, ddof=1)))


def _validate_design_inputs(Y: np.ndarray, K: int) -> None:
    if Y.ndim != 2:
        raise MlsynthConfigError("Y must be a two-dimensional T x N matrix.")
    if K <= 0:
        raise MlsynthConfigError("K must be a positive integer.")
    if K > Y.shape[1]:
        raise MlsynthConfigError("K cannot exceed the number of units.")


def _build_global_2way(Y: np.ndarray, D: cp.Variable, K: int, lam: float):
    """Build the global two-way formulation components.

    This compatibility wrapper delegates to
    :func:`mlsynth.utils.scdi_helpers.formulation.build_global_2way_components`.
    New code should use the formulation module directly when it needs to add or
    inspect constraints before solving.
    """

    components = build_scdi_problem_components(
        Y=Y, D=D, K=K, lam=lam, mode="global_2way"
    )
    return unpack_problem_components(components)


def _build_per_unit(Y: np.ndarray, D: cp.Variable, K: int, lam: float):
    """Build the per-unit formulation components.

    This compatibility wrapper delegates to
    :func:`mlsynth.utils.scdi_helpers.formulation.build_per_unit_components`.
    New code should use the formulation module directly when it needs to add or
    inspect constraints before solving.
    """

    components = build_scdi_problem_components(Y=Y, D=D, K=K, lam=lam, mode="per_unit")
    return unpack_problem_components(components)


def _extract_weights(mode, assignment, values):
    N = len(assignment)
    K = assignment.sum()

    w = values.get("w")
    q = values.get("q")

    # ----------------------------
    # Equal weights case
    # ----------------------------
    if mode == "global_equal_weights":
        treated_weights = assignment / K
        control_weights = (1 - assignment) / (N - K)
        contrast_weights = treated_weights - control_weights
        return treated_weights, control_weights, contrast_weights

    # ----------------------------
    # Learned-weight cases
    # ----------------------------
    if w is not None and q is not None:
        treated_weights = q
        control_weights = w - q
        contrast_weights = 2 * q - w
        return treated_weights, control_weights, contrast_weights

    return None, None, None



def solve_synthetic_design(
    Y: np.ndarray,
    K: int,
    mode: str = "global_2way",
    lam: Optional[float] = None,
    solver: Any = "SCIP",
    verbose: bool = False,
    unit_index: Optional[IndexSet] = None,
) -> SCDIDesign:

    _validate_design_inputs(Y, K)

    lam_value = estimate_lambda(Y) if lam is None else float(lam)
    if lam_value < 0:
        raise MlsynthConfigError("lam must be nonnegative.")

    _, N = Y.shape
    D = cp.Variable(N, boolean=True)

    components = build_scdi_problem_components(
        Y=Y, D=D, K=K, lam=lam_value, mode=mode
    )

    problem = cp.Problem(cp.Minimize(components.objective), components.constraints)

    try:
        problem.solve(solver=solver, verbose=verbose)
    except Exception as exc:
        raise MlsynthEstimationError(
            f"SCDI optimization failed to solve: {exc}"
        ) from exc

    if problem.status not in _OPTIMAL_STATUSES:
        raise MlsynthEstimationError(f"SCDI optimization failed: {problem.status}")

    if D.value is None:
        raise MlsynthEstimationError("SCDI optimization did not return an assignment.")

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

    raw_results: Dict[str, Any] = {
        "status": problem.status,
        "solver": solver,
    }

    return SCDIDesign(
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

        raw_results=raw_results,
    )
