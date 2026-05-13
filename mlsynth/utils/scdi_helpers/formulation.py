"""Objective and constraint builders for SCDI optimization problems."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from .structures import SCDIProblemComponents


# ============================================================
# GLOBAL TWO-WAY
# ============================================================


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

        # normalization
        cp.sum(q) == 1,
        cp.sum(w) == 2,

        # linearization q_i = w_i * D_i
        q <= D,
        q <= w,
        q >= w - (1 - D),
    ]

    residual_constraints = [
        z[t] == cp.sum(cp.multiply(2 * q - w, Y[t, :]))
        for t in range(T)
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

    constraints = build_global_2way_constraints(
        Y=Y,
        D=D,
        K=K,
        variables=variables,
    )

    objective = build_global_2way_objective(
        Y=Y,
        lam=lam,
        variables=variables,
    )

    return SCDIProblemComponents(
        mode="global_2way",
        objective=objective,
        constraints=constraints,
        variables=variables,
        assignment_variable=D,
    )


# ============================================================
# GLOBAL EQUAL WEIGHTS
# ============================================================


def build_global_equal_weights_variables(
    T: int,
    N: int,
) -> Dict[str, cp.Variable]:
    """Create optimization variables for the equal-weight global formulation.

    This special case fixes:
        treated weights  = 1 / K
        control weights  = 1 / (N - K)

    The only optimization decision is the binary assignment vector D.
    """

    return {
        "z": cp.Variable(T),
    }


def build_global_equal_weights_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """Build constraints for the equal-weight global SCDI formulation."""

    T, N = Y.shape

    if K >= N:
        raise ValueError(
            "global_equal_weights requires K to be less "
            "than the number of units."
        )

    z = variables["z"]

    treated_weight = 1.0 / K
    control_weight = 1.0 / (N - K)

    contrast = (
        cp.multiply(D, treated_weight)
        - cp.multiply(1 - D, control_weight)
    )

    constraints: List[cp.Constraint] = [
        cp.sum(D) == K,
    ]

    residual_constraints = [
        z[t] == Y[t, :] @ contrast
        for t in range(T)
    ]

    constraints.extend(residual_constraints)

    return constraints


def build_global_equal_weights_objective(
    Y: np.ndarray,
    K: int,
    lam: float,
    variables: Dict[str, cp.Variable],
) -> cp.Expression:
    """Build objective for equal-weight global SCDI formulation."""

    T, N = Y.shape

    if K >= N:
        raise ValueError(
            "global_equal_weights requires K to be less "
            "than the number of units."
        )

    z = variables["z"]

    residual_loss = cp.sum_squares(z) / T

    # Since weights are fixed:
    #
    # treated contribution:
    #   K * (1/K)^2 = 1/K
    #
    # control contribution:
    #   (N-K) * (1/(N-K))^2 = 1/(N-K)

    constant_weight_penalty = (
        lam * (
            (1.0 / K)
            + (1.0 / (N - K))
        )
    )

    return residual_loss + constant_weight_penalty


def build_global_equal_weights_components(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    lam: float,
) -> SCDIProblemComponents:
    """Build objective, constraints, and variables for equal-weight global SCDI."""

    T, N = Y.shape

    variables = build_global_equal_weights_variables(T, N)

    constraints = build_global_equal_weights_constraints(
        Y=Y,
        D=D,
        K=K,
        variables=variables,
    )

    objective = build_global_equal_weights_objective(
        Y=Y,
        K=K,
        lam=lam,
        variables=variables,
    )

    return SCDIProblemComponents(
        mode="global_equal_weights",
        objective=objective,
        constraints=constraints,
        variables=variables,
        assignment_variable=D,
    )


# ============================================================
# PER-UNIT
# ============================================================


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

    constraints: List[cp.Constraint] = [
        cp.sum(D) == K,
    ]

    for i in range(N):

        # treated units receive donor simplex
        constraints.append(
            cp.sum(q[i, :]) == D[i]
        )

        for j in range(N):

            constraints.extend(
                [
                    # q_ij = w_ij * (1 - D_j)

                    q[i, j] <= 1 - D[j],
                    q[i, j] <= w[i, j],
                    q[i, j] >= w[i, j] - D[j],

                    # only treated units may carry weights
                    w[i, j] <= D[i],
                ]
            )

        residual_constraints = [
            z[i, t] == (
                Y[t, i] * D[i]
                - q[i, :] @ Y[t, :]
            )
            for t in range(T)
        ]

        constraints.extend(residual_constraints)

    return constraints


def build_per_unit_objective(
    Y: np.ndarray,
    K: int,
    lam: float,
    variables: Dict[str, cp.Variable],
) -> cp.Expression:
    """Build objective expression for the per-unit SCDI formulation."""

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

    constraints = build_per_unit_constraints(
        Y=Y,
        D=D,
        K=K,
        variables=variables,
    )

    objective = build_per_unit_objective(
        Y=Y,
        K=K,
        lam=lam,
        variables=variables,
    )

    return SCDIProblemComponents(
        mode="per_unit",
        objective=objective,
        constraints=constraints,
        variables=variables,
        assignment_variable=D,
    )


# ============================================================
# DISPATCHER
# ============================================================


def build_scdi_problem_components(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    lam: float,
    mode: str,
) -> SCDIProblemComponents:
    """Dispatch to the formulation builder for an SCDI optimization mode."""

    if mode == "global_2way":
        return build_global_2way_components(
            Y=Y,
            D=D,
            K=K,
            lam=lam,
        )

    if mode == "global_equal_weights":
        return build_global_equal_weights_components(
            Y=Y,
            D=D,
            K=K,
            lam=lam,
        )

    if mode == "per_unit":
        return build_per_unit_components(
            Y=Y,
            D=D,
            K=K,
            lam=lam,
        )

    raise ValueError(
        "Unknown SCDI mode. Expected one of: "
        "'global_2way', 'global_equal_weights', or 'per_unit'."
    )


# ============================================================
# HELPERS
# ============================================================


def unpack_problem_components(
    components: SCDIProblemComponents,
) -> Tuple[
    cp.Expression,
    List[cp.Constraint],
    Dict[str, cp.Variable],
]:
    """Return objective, constraints, and variables from problem components."""

    return (
        components.objective,
        components.constraints,
        components.variables,
    )
