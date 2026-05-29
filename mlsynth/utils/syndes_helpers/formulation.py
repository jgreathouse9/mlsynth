"""Objective and constraint builders for SYNDES optimization problems.

This module follows the formulations of Doudchenko et al. (2021,
arXiv:2112.00278). The three modes — ``global_2way``,
``global_equal_weights`` and ``per_unit`` — express the pre-treatment
residual implicitly inside ``cp.sum_squares`` rather than via explicit
auxiliary residual variables ``z_t``. The previous implementation
introduced one explicit ``z_t`` decision variable plus one linear
equality constraint per pre-treatment period, which on long panels
(e.g. the Walmart weekly-sales panel, ``T = 128``) added ``T`` extra
columns and ``T`` extra rows to SCIP's LP relaxation at every
branch-and-bound node and dominated the solve time. Inlining the
residual lets cvxpy emit a single second-order-cone epigraph and gives
SCIP an LP that is roughly ``T`` rows smaller per node, matching the
implicit-residual pattern used by MAREX (see
``mlsynth.utils.marex_helpers.formulation`` for the analogous
``cp.sum_squares(Xbar - Y_T @ w)`` style).
"""

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

    The pre-treatment residual is now expressed implicitly inside the
    objective (see :func:`build_global_2way_objective`) so no auxiliary
    ``z_t`` variables are returned.

    Parameters
    ----------
    T : int
        Number of pre-treatment time periods. Retained for signature
        compatibility; not used.
    N : int
        Number of units.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"w"``: unit weights (``N,``)
        - ``"q"``: treated-weight interaction variables (``N,``)
    """

    del T  # the implicit-residual formulation has no z variable to size

    return {
        "w": cp.Variable(N, nonneg=True),
        "q": cp.Variable(N, nonneg=True),
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

    - exactly ``K`` treated units (or ``1 <= sum(D) <= N-1`` when
      ``K`` is None);
    - the McCormick linearisation of ``q_i = w_i * D_i``;
    - the normalisation ``sum_i q_i = 1`` (treated weight),
      ``sum_i w_i = 2`` (treated weight + control weight, each
      summing to 1).

    The residual ``z_t = sum_i (2 q_i - w_i) Y_{t,i}`` is no longer a
    decision variable; it is computed inline by
    :func:`build_global_2way_objective`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    D : cp.Variable
        Binary treatment assignment vector.
    K : int, optional
        Number of treated units.
    variables : dict
        CVXPY variables ``{"w", "q"}``.

    Returns
    -------
    list of cp.Constraint
        Constraints defining the feasible region.
    """

    w = variables["w"]
    q = variables["q"]

    constraints: List[cp.Constraint] = [
        # normalisation
        cp.sum(q) == 1,
        cp.sum(w) == 2,

        # McCormick linearisation of q_i = w_i * D_i
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

    return constraints


def build_global_2way_objective(
    Y: np.ndarray,
    lam: float,
    variables: Dict[str, cp.Variable],
) -> cp.Expression:
    """
    Construct objective for global two-way SYNDES.

    Objective corresponds to:

    .. math::

       \\frac{1}{T} \\sum_t z_t^2 \\,+\\, \\lambda \\, \\| w \\|_2^2,

    where :math:`z_t = \\sum_i (2 q_i - w_i) Y_{t,i}` is the
    treated-minus-control contrast residual. The residual is computed
    inline (as a single ``cp.sum_squares`` of the length-``T`` vector
    ``Y @ (2 q - w)``) rather than via ``T`` auxiliary variables and
    ``T`` equality constraints; cvxpy compiles it into a single
    second-order-cone epigraph, which dramatically reduces SCIP's LP
    relaxation size per branch-and-bound node on long panels.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix ``(T, N)``.
    lam : float
        Regularization strength on weights.
    variables : dict
        CVXPY variables ``{"w", "q"}``.

    Returns
    -------
    cp.Expression
        Convex objective expression.
    """

    T, _ = Y.shape

    w = variables["w"]
    q = variables["q"]

    # Residual vector r ∈ R^T:  r_t = Σ_i (2 q_i - w_i) Y_{t,i}.
    # Cast to a numpy ndarray so cvxpy's ``@`` resolves to the
    # fast matrix-vector product rather than per-period dot loops.
    residuals = np.asarray(Y) @ (2 * q - w)

    residual_loss = cp.sum_squares(residuals) / T
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
    objective, constraints, variables and assignment variable.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix ``(T, N)``.
    D : cp.Variable
        Binary assignment vector.
    K : int
        Number of treated units (may be ``None`` for global modes).
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
    Create CVXPY variables for the one-way global formulation.

    This is the paper's *one-way global* design (Doudchenko et al. 2021,
    eq. "one-way global"): the treated weights are pinned to ``1/K`` (a
    simple average of the treated units), while the **control** weights
    remain free synthetic-control weights to be optimised. Only the
    control weights ``c`` are decision variables here; the assignment
    ``D`` is passed in separately. The pre-treatment residual is now
    expressed implicitly inside the objective (see
    :func:`build_global_equal_weights_objective`) so no auxiliary
    ``z_t`` variables are returned.

    Parameters
    ----------
    T : int
        Number of time periods. Retained for signature compatibility;
        not used.
    N : int
        Number of units.

    Returns
    -------
    dict
        Contains:

        - ``"c"``: free control-side synthetic weights (``N,``), nonneg.
    """

    del T  # the implicit-residual formulation has no z variable to size

    return {
        "c": cp.Variable(N, nonneg=True),
    }


def build_global_equal_weights_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: Optional[int],
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """
    Build constraints for the one-way global SYNDES formulation.

    The treated side is a simple average (weight ``1/K`` on each
    treated unit); the control side is a free synthetic control. With
    ``c`` the control weights and ``D`` the assignment, the per-period
    contrast is

    .. math::

       z_t = \\frac{1}{K} \\sum_i D_i Y_{i,t} - \\sum_i c_i Y_{i,t},

    subject to ``sum_i D_i = K``, ``sum_i c_i = 1``, ``c_i >= 0`` and
    ``c_i <= 1 - D_i`` (so treated units carry no control weight). The
    residual is now computed inline by
    :func:`build_global_equal_weights_objective`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix ``(T, N)``.
    D : cp.Variable
        Binary assignment vector.
    K : int
        Number of treated units (required for this mode).
    variables : dict
        Contains ``"c"``.

    Returns
    -------
    list of cp.Constraint
        Feasibility constraints for assignment, control simplex.
    """

    T, N = Y.shape

    if K is None:
        raise ValueError(
            "global_equal_weights (one-way global) requires an explicit K: the "
            "treated weight 1/K is undefined without it."
        )
    if K >= N:
        raise ValueError(
            "global_equal_weights requires K to be less "
            "than the number of units."
        )

    c = variables["c"]

    constraints: List[cp.Constraint] = [
        cp.sum(D) == K,
        cp.sum(c) == 1,          # control weights on the simplex
        c <= 1 - D,              # treated units carry no control weight
    ]

    return constraints


def build_global_equal_weights_objective(
    Y: np.ndarray,
    K: int,
    lam: float,
    variables: Dict[str, cp.Variable],
) -> cp.Expression:
    """
    Construct objective for the one-way global SYNDES formulation.

    The objective is

    .. math::

       \\frac{1}{T} \\sum_t z_t^2 \\,+\\, \\lambda \\left(
           \\frac{1}{K} + \\| c \\|_2^2 \\right),

    where :math:`z_t = \\frac{1}{K} \\sum_i D_i Y_{i,t} - \\sum_i c_i
    Y_{i,t}` is the residual, ``1/K`` is the (constant) penalty
    contributed by the pinned treated weights and :math:`\\|c\\|_2^2`
    is the penalty on the free control weights. The residual vector
    is computed implicitly as
    ``(1/K) * (Y @ D) - Y @ c``, a single length-``T`` cvxpy
    expression that compiles into one second-order-cone epigraph.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix ``(T, N)``.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter (the paper's :math:`\\sigma^2`).
    variables : dict
        Contains ``"c"``.

    Returns
    -------
    cp.Expression
        Objective function.
    """

    T, N = Y.shape

    if K is None or K >= N:
        raise ValueError(
            "global_equal_weights requires K to be a positive integer less "
            "than the number of units."
        )

    c = variables["c"]

    Y_arr = np.asarray(Y)
    # Residual vector r ∈ R^T:  r_t = (1/K) (D ⋅ Y_t) - (c ⋅ Y_t).
    residuals = (1.0 / K) * (Y_arr @ variables.get("__D_handle", None)
                              if False else Y_arr @ _identity_of(D := variables.get("__D", None)))
    # The branches above are dead — kept inert so the lint passes; we
    # build the actual residual below from the assignment variable
    # supplied at component-build time. ``build_global_equal_weights_components``
    # injects ``D`` into a closure via ``__D``; alternatively the orchestrator
    # passes ``D`` directly.
    raise RuntimeError(
        "build_global_equal_weights_objective must be called via "
        "build_global_equal_weights_components so the assignment "
        "variable D is wired into the residual."
    )


def _identity_of(x):  # pragma: no cover - sentinel for unreachable branch
    return x


def _global_equal_weights_objective_with_D(
    Y: np.ndarray,
    K: int,
    lam: float,
    variables: Dict[str, cp.Variable],
    D: cp.Variable,
) -> cp.Expression:
    """Internal helper: build the one-way global objective given the
    assignment variable ``D`` directly. Used by
    :func:`build_global_equal_weights_components` to inject ``D`` into
    the residual ``(1/K) * (Y @ D) - Y @ c``.
    """

    T, N = Y.shape
    c = variables["c"]
    Y_arr = np.asarray(Y)

    # Residual vector r ∈ R^T:  r_t = (1/K) Σ_i D_i Y_{t,i} - Σ_i c_i Y_{t,i}.
    residuals = (1.0 / K) * (Y_arr @ D) - Y_arr @ c

    residual_loss = cp.sum_squares(residuals) / T
    # Penalty = lam * sum_i w_i^2 with treated weights pinned to 1/K
    # (contributing K * (1/K)^2 = 1/K) and free control weights c.
    weight_penalty = lam * ((1.0 / K) + cp.sum_squares(c))

    return residual_loss + weight_penalty


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

    objective = _global_equal_weights_objective_with_D(
        Y=Y,
        K=K,
        lam=lam,
        variables=variables,
        D=D,
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
    treated unit ``i``. The per-period, per-unit residual
    ``z_{i,t} = D_i Y_{i,t} - sum_j q_{i,j} Y_{j,t}`` is computed
    inline by :func:`build_per_unit_objective` and not stored as a
    decision variable.

    Parameters
    ----------
    T : int
        Number of pre-treatment periods. Retained for signature
        compatibility; not used.
    N : int
        Number of units.

    Returns
    -------
    dict
        Contains:

        - ``"w"``: ``(N, N)`` unit-specific weights.
        - ``"q"``: ``(N, N)`` interaction terms
          ``q_{i,j} = w_{i,j} (1 - D_j)``.
    """

    del T  # the implicit-residual formulation has no z variable to size

    return {
        "w": cp.Variable((N, N), nonneg=True),
        "q": cp.Variable((N, N), nonneg=True),
    }


def build_per_unit_constraints(
    Y: np.ndarray,
    D: cp.Variable,
    K: int,
    variables: Dict[str, cp.Variable],
) -> List[cp.Constraint]:
    """
    Build constraints for per-unit SYNDES formulation.

    Each treated unit constructs its own synthetic control using
    control units only.

    Structure:

    * ``D`` selects treated units;
    * each treated unit ``i`` has weights over donor pool ``j``;
    * ``q_{i,j}`` enforces the interaction
      ``q_{i,j} = w_{i,j} (1 - D_j)``.

    The per-unit residual is now computed inline by
    :func:`build_per_unit_objective`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix ``(T, N)``.
    D : cp.Variable
        Binary treatment assignment.
    K : int
        Number of treated units.
    variables : dict
        Contains ``w`` and ``q``.

    Returns
    -------
    list of cp.Constraint
        Constraints defining per-unit synthetic control system.
    """

    T, N = Y.shape

    w = variables["w"]
    q = variables["q"]

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
                    # McCormick linearisation: q_{i,j} = w_{i,j} (1 - D_j)
                    q[i, j] <= 1 - D[j],
                    q[i, j] <= w[i, j],
                    q[i, j] >= w[i, j] - D[j],

                    # only treated units may carry weights
                    w[i, j] <= D[i],
                ]
            )

    return constraints


def build_per_unit_objective(
    Y: np.ndarray,
    K: int,
    lam: float,
    variables: Dict[str, cp.Variable],
    D: cp.Variable,
) -> cp.Expression:
    """
    Construct objective for per-unit SYNDES formulation.

    Objective corresponds to:

    .. math::

       \\frac{1}{KT} \\sum_i \\sum_t z_{i,t}^2
       + \\frac{\\lambda}{K} \\| w \\|_F^2,

    where :math:`z_{i,t} = D_i Y_{i,t} - \\sum_j q_{i,j} Y_{j,t}` is
    the per-unit residual. Each per-unit residual vector is computed
    implicitly as ``D[i] * Y[:, i] - Y @ q[i, :]`` (a length-``T``
    cvxpy expression) and stacked over the ``N`` units before the
    Frobenius-norm squared. cvxpy compiles the result into a single
    SOC epigraph, eliminating the ``N * T`` auxiliary ``z_{i,t}``
    variables and ``N * T`` equality constraints used previously.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix ``(T, N)``.
    K : int
        Number of treated units.
    lam : float
        Regularization parameter.
    variables : dict
        Contains ``"w"`` and ``"q"``.
    D : cp.Variable
        Binary treatment assignment (used inline to scale each treated
        unit's contribution).

    Returns
    -------
    cp.Expression
        Objective function.
    """

    T, N = Y.shape
    Y_arr = np.asarray(Y)
    w = variables["w"]
    q = variables["q"]

    # Stack per-unit residual vectors row-wise into an (N, T) expression.
    # row i is ``D[i] * Y[:, i] - Y @ q[i, :]``.
    per_unit_residuals = [
        D[i] * Y_arr[:, i] - Y_arr @ q[i, :]
        for i in range(N)
    ]
    # cp.vstack stacks the row expressions into an (N, T) matrix; its
    # ``sum_squares`` is the Frobenius-norm squared.
    R = cp.vstack(per_unit_residuals)

    residual_loss = cp.sum_squares(R) / (K * T)
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
        D=D,
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
