"""Objective and constraint builders for SCDI optimization problems."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from .structures import SCDIProblemComponents


def build_global_2way_variables(T: int, N: int) -> Dict[str, cp.Variable]:
    """Create optimization variables for the global two-way SCDI formulation."""

    return {
        "w": cp.Variable(N, nonneg=True),
        "q": cp.Variable(N, nonneg=True),
        "z": cp.Variable(T),
    }


def build_global_2way_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """Build constraints for the global two-way SCDI formulation."""

    T, _ = Y.shape
    w = variables["w"]
    q = variables["q"]
    z = variables["z"]

    constraints: List[cp.Constraint] = [
        cp.sum(D) == K,
        cp.sum(q) == 1,
        cp.sum(w) == 2,
        q <= D,
        q <= w,
        q >= w - (1 - D),
    ]

    residual_constraints = [
        z[t] == cp.sum(cp.multiply(2 * q - w, Y[t, :])) for t in range(T)
    ]
    constraints.extend(residual_constraints)
    return constraints


def build_global_2way_objective(
    Y: np.ndarray,
    lam: float,
    variables: Dict[str, cp.Variable],
) -> cp.Expression:
    """Build the objective expression for the global two-way SCDI formulation."""

    T, _ = Y.shape
    w = variables["w"]
    z = variables["z"]
    residual_loss = cp.sum_squares(z) / T
    weight_penalty = lam * cp.sum_squares(w)
    return residual_loss + weight_penalty


def build_global_2way_components(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    lam: float,
) -> SCDIProblemComponents:
    """Build objective, constraints, and variables for global two-way SCDI."""

    T, N = Y.shape
    variables = build_global_2way_variables(T, N)
    constraints = build_global_2way_constraints(Y, D, K, variables)
    objective = build_global_2way_objective(Y, lam, variables)
    return SCDIProblemComponents(
        mode="global_2way",
        objective=objective,
        constraints=constraints,
        variables=variables,
        assignment_variable=D,
    )


def build_per_unit_variables(T: int, N: int) -> Dict[str, cp.Variable]:
    """Create optimization variables for the per-unit SCDI formulation."""

    return {
        "w": cp.Variable((N, N), nonneg=True),
        "q": cp.Variable((N, N), nonneg=True),
        "z": cp.Variable((N, T)),
    }


def build_per_unit_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """Build constraints for the per-unit SCDI formulation."""

    T, N = Y.shape
    w = variables["w"]
    q = variables["q"]
    z = variables["z"]

    constraints: List[cp.Constraint] = [cp.sum(D) == K]

    for i in range(N):
        constraints.append(cp.sum(q[i, :]) == D[i])
        for j in range(N):
            constraints.extend(
                [
                    q[i, j] <= 1 - D[j],
                    q[i, j] <= w[i, j],
                    q[i, j] >= w[i, j] - D[j],
                    w[i, j] <= D[i],
                ]
            )
        residual_constraints = [
            z[i, t] == Y[t, i] * D[i] - q[i, :] @ Y[t, :] for t in range(T)
        ]
        constraints.extend(residual_constraints)

    return constraints


def build_per_unit_objective(
    Y: np.ndarray,
    K: int,
    lam: float,
    variables: Dict[str, cp.Variable],
) -> cp.Expression:
    """Build the objective expression for the per-unit SCDI formulation."""

    T, _ = Y.shape
    w = variables["w"]
    z = variables["z"]
    residual_loss = cp.sum_squares(z) / (K * T)
    weight_penalty = (lam / K) * cp.sum_squares(w)
    return residual_loss + weight_penalty


def build_per_unit_components(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    lam: float,
) -> SCDIProblemComponents:
    """Build objective, constraints, and variables for per-unit SCDI."""

    T, N = Y.shape
    variables = build_per_unit_variables(T, N)
    constraints = build_per_unit_constraints(Y, D, K, variables)
    objective = build_per_unit_objective(Y, K, lam, variables)
    return SCDIProblemComponents(
        mode="per_unit",
        objective=objective,
        constraints=constraints,
        variables=variables,
        assignment_variable=D,
    )


def build_scdi_problem_components(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    lam: float,
    mode: str,
) -> SCDIProblemComponents:
    """Dispatch to the formulation builder for an SCDI optimization mode."""

    if mode == "global_2way":
        return build_global_2way_components(Y, D, K, lam)
    if mode == "per_unit":
        return build_per_unit_components(Y, D, K, lam)
    raise ValueError("Unknown SCDI mode. Expected 'global_2way' or 'per_unit'.")


def unpack_problem_components(
    components: SCDIProblemComponents,
) -> Tuple[cp.Expression, List[cp.Constraint], Dict[str, cp.Variable]]:
    """Return objective, constraints, and variables from problem components."""

    return components.objective, components.constraints, components.variables
