"""Objective and constraint builders for SYNDES optimization problems."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np

from .structures import SYNDESProblemComponents


# ============================================================
# GLOBAL TWO-WAY
# ============================================================


def build_global_2way_variables(T: int, N: int) -> Dict[str, cp.Variable]:
    """
    Create CVXPY decision variables for the global two-way SYNDES formulation.

    This formulation jointly optimizes treatment assignment and global
    synthetic contrast weights.

    Parameters
    ----------
    T : int
        Number of pre-treatment time periods.
    N : int
        Number of units.

    Returns
    -------
    dict
        Dictionary containing:
        - "w": unit weights (N,)
        - "q": treated-weight interaction variables (N,)
        - "z": residual vector over time (T,)
    """

    return {
        "w": cp.Variable(N, nonneg=True),
        "q": cp.Variable(N, nonneg=True),
        "z": cp.Variable(T),
    }


def build_global_2way_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: Optional[int],
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """
    Build constraints for the global two-way SYNDES formulation.

    The formulation enforces:
    - exactly K treated units
    - linearized interaction terms q_i = w_i * D_i
    - normalization constraints on weights
    - residual construction for pre-treatment fit

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape (T, N).
    D : cp.Variable
        Binary treatment assignment vector.
    K : int
        Number of treated units.
    variables : dict
        CVXPY variables {"w", "q", "z"}.

    Returns
    -------
    list of cp.Constraint
        Constraints defining the feasible region.
    """

    T, _ = Y.shape

    w = variables["w"]
    q = variables["q"]
    z = variables["z"]

    constraints: List[cp.Constraint] = [
        # normalization
        cp.sum(q) == 1,
        cp.sum(w) == 2,

        # linearization q_i = w_i * D_i
        q <= D,
        q <= w,
        q >= w - (1 - D),
    ]
    # Paper allows K=None for global modes (see Doudchenko et al. 2021,
    # paragraph after eq. 9); pin K only when explicitly supplied.
    if K is not None:
        constraints.append(cp.sum(D) == K)
    else:
        # Without a K-pin the trivial solution D=0 is admissible. Force
        # at least one treated and one control so the contrast vector
        # is well-defined.
        constraints.append(cp.sum(D) >= 1)
        constraints.append(cp.sum(D) <= Y.shape[1] - 1)

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
    """
    Construct objective for global two-way SYNDES.

    Objective corresponds to:

        (1/T) * sum_t z_t^2 + lam * ||w||_2^2

    where z_t is the pre-treatment residual of the treated-control contrast.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    lam : float
        Regularization strength on weights.
    variables : dict
        CVXPY variables {"w", "z"}.

    Returns
    -------
    cp.Expression
        Convex objective expression.
    """

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
) -> SYNDESProblemComponents:
    """
    Construct full SYNDES problem (global two-way formulation).

    Returns a complete CVXPY optimization specification consisting of:
    - objective
    - constraints
    - variables
    - assignment variable

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    D : cp.Variable
        Binary assignment vector.
    K : int
        Number of treated units.
    lam : float
        Weight regularization parameter.

    Returns
    -------
    SYNDESProblemComponents
        Fully specified optimization problem.
    """

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

    return SYNDESProblemComponents(
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
    """
    Create CVXPY variables for the equal-weight global formulation.

    This is a restricted version of global two-way SYNDES where:
        - treated weights are fixed to 1/K
        - control weights are fixed to 1/(N-K)

    Only the assignment vector D is optimized.

    Parameters
    ----------
    T : int
        Number of time periods.
    N : int
        Number of units.

    Returns
    -------
    dict
        Contains only:
        - "z": residual vector over time (T,)
    """

    return {
        "z": cp.Variable(T),
    }


def build_global_equal_weights_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: Optional[int],
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """
    Build constraints for the equal-weight global SYNDES formulation.

    The model enforces a fixed weighting scheme:
        treated units: 1/K
        control units: 1/(N-K)

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    D : cp.Variable
        Binary assignment vector.
    K : int
        Number of treated units.
    variables : dict
        Contains residual variable "z".

    Returns
    -------
    list of cp.Constraint
        Feasibility constraints for assignment and residual definition.
    """

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
    """
    Construct objective for equal-weight global SYNDES.

    The objective is:

        (1/T) * sum_t z_t^2 + lam * (1/K + 1/(N-K))

    The second term is constant with respect to D and only reflects
    the fixed weighting scheme.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.
    variables : dict
        Contains residual variable "z".

    Returns
    -------
    cp.Expression
        Objective function.
    """

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
) -> SYNDESProblemComponents:
    """
    Build full equal-weight global SYNDES problem.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix.
    D : cp.Variable
        Assignment variable.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.

    Returns
    -------
    SYNDESProblemComponents
        Complete optimization specification.
    """

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

    return SYNDESProblemComponents(
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
    """
    Create CVXPY variables for the per-unit SYNDES formulation.

    This formulation constructs a separate synthetic control for each
    treated unit i.

    Parameters
    ----------
    T : int
        Number of pre-treatment periods.
    N : int
        Number of units.

    Returns
    -------
    dict
        Contains:
        - "w": (N, N) unit-specific weights
        - "q": (N, N) interaction terms q_ij = w_ij (1 - D_j)
        - "z": (N, T) residuals per unit and time
    """

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
    """
    Build constraints for per-unit SYNDES formulation.

    Each treated unit constructs its own synthetic control using control
    units only.

    Structure:
    - D selects treated units
    - each treated unit i has weights over donor pool j
    - q_ij enforces interaction q_ij = w_ij * (1 - D_j)

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix (T, N).
    D : cp.Variable
        Binary treatment assignment.
    K : int
        Number of treated units.
    variables : dict
        Contains w, q, z.

    Returns
    -------
    list of cp.Constraint
        Constraints defining per-unit synthetic control system.
    """

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
    """
    Construct objective for per-unit SYNDES formulation.

    Objective corresponds to:

        (1 / (K T)) * sum_i sum_t z_it^2
        + (lam / K) * ||w||_F^2

    where z_it is the synthetic control residual for treated unit i.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.
    variables : dict
        Contains "w" and "z".

    Returns
    -------
    cp.Expression
        Objective function.
    """

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
) -> SYNDESProblemComponents:
    """
    Construct full per-unit SYNDES optimization problem.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix.
    D : cp.Variable
        Assignment vector.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.

    Returns
    -------
    SYNDESProblemComponents
        Full per-unit optimization specification.
    """

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

    return SYNDESProblemComponents(
        mode="per_unit",
        objective=objective,
        constraints=constraints,
        variables=variables,
        assignment_variable=D,
    )


# ============================================================
# DISPATCHER
# ============================================================


def build_syndes_problem_components(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    lam: float,
    mode: str,
) -> SYNDESProblemComponents:
    """
    Dispatch SYNDES formulation builder based on mode.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix.
    D : cp.Variable
        Binary treatment assignment.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.
    mode : {"global_2way", "global_equal_weights", "per_unit"}
        SYNDES formulation selector.

    Returns
    -------
    SYNDESProblemComponents
        Fully specified optimization problem.

    Raises
    ------
    ValueError
        If mode is not recognized.
    """

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
        "Unknown SYNDES mode. Expected one of: "
        "'global_2way', 'global_equal_weights', or 'per_unit'."
    )


# ============================================================
# HELPERS
# ============================================================


def unpack_problem_components(
    components: SYNDESProblemComponents,
) -> Tuple[
    cp.Expression,
    List[cp.Constraint],
    Dict[str, cp.Variable],
]:
    """
    Unpack SYNDES problem components.

    Parameters
    ----------
    components : SYNDESProblemComponents
        Structured optimization container.

    Returns
    -------
    tuple
        (objective, constraints, variables)
    """

    return (
        components.objective,
        components.constraints,
        components.variables,
    )
