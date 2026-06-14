"""MlSC QP solver.

Solves Equation 5.2 of Bottmer (2025)::

    min over omega in R^M
        || Y_agg_treated_pre  -  X_disagg_pre @ omega ||^2
        + lambda * sigma_y^2 * omega^T Q omega
    s.t.  1^T omega = 1,   omega >= 0.

The penalty ``Q = R^T R`` factors in closed form (``R_s = I - v_s 1^T`` per
block; see :func:`mlsynth.utils.mlsc_helpers.penalty.build_sqrt_factor`), so the
penalized objective is an ordinary simplex least squares once the penalty is
folded into the design as the augmentation ``[X; sqrt(lambda sigma_y^2) R]`` with
a zero target. For any ``lambda > 0`` the program is strictly convex whenever the
aggregate control matrix has full column rank (the penalty fills exactly the part
of X's null space the aggregate directions do not pin down), so the optimum is
unique. We solve it with the library's active-set simplex QP
(:func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`) -- exact and free of
cvxpy's per-call canonicalisation overhead in the cross-validation loop.

The ``lambda = 0`` case is regularised by a tiny ridge (``_RIDGE_FLOOR``), the
same device the reference uses to keep the fully-disaggregated fit unique on a
flat objective. An explicit ``solver`` argument routes through cvxpy instead, an
escape hatch that preserves the original behaviour for callers who request a
particular cvxpy solver.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .penalty import build_sqrt_factor
from .structures import MLSCInputs

# Tiny ridge that keeps the unpenalised (lambda = 0) simplex fit unique on the
# flat, under-determined objective (M disaggregate columns >> T0 periods). Mirrors
# the reference ``synthetic_control_counties`` 1e-8 L2 term.
_RIDGE_FLOOR = 1e-8


def _augmented_design(
    X: np.ndarray, Y: np.ndarray, inputs: MLSCInputs, penalty_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the penalty-augmented design ``([X; sqrt(p) R], [Y; 0])``.

    For ``penalty_scale > 0`` the augmentation uses the block square-root factor
    ``R`` (``R^T R == Q``); for ``penalty_scale == 0`` it uses the uniqueness
    ridge ``sqrt(_RIDGE_FLOOR) I``.
    """
    M = X.shape[1]
    if penalty_scale > 0:
        R = build_sqrt_factor(inputs.v_population, inputs.disagg_to_agg)
        aug = np.sqrt(penalty_scale) * R
    else:
        aug = np.sqrt(_RIDGE_FLOOR) * np.eye(M)
    B = np.vstack([X, aug])
    A = np.concatenate([Y, np.zeros(M)])
    return B, A


def _aggregate(omega: np.ndarray, disagg_to_agg: np.ndarray) -> np.ndarray:
    """Implied aggregate weights ``w_s = sum_c omega_sc``."""
    S = int(disagg_to_agg.max() + 1)
    return np.array([omega[disagg_to_agg == s].sum() for s in range(S)])


def _classical_sc_warm_start(
    Y_pre: np.ndarray,
    X_disagg_pre: np.ndarray,
    v_population: np.ndarray,
    disagg_to_agg: np.ndarray,
    solver: Any,
) -> np.ndarray:
    """Aggregate-level classical SC, expanded to a disaggregate warm start.

    Used only by the cvxpy escape hatch (:func:`_solve_mlsc_cvxpy`) to seed the
    solver; the native active-set path needs no feasible warm start.
    """
    import cvxpy as cp

    S = int(disagg_to_agg.max() + 1)
    T0 = Y_pre.shape[0]
    X_agg_pre = np.zeros((T0, S))
    for s in range(S):
        mask = disagg_to_agg == s
        X_agg_pre[:, s] = X_disagg_pre[:, mask] @ v_population[mask]

    w = cp.Variable(S)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(Y_pre - X_agg_pre @ w)),
        [cp.sum(w) == 1, w >= 0],
    )
    problem.solve(solver=solver or cp.SCS)
    if w.value is None:  # pragma: no cover - defensive: warm-start solve returned None
        return v_population / max(S, 1)

    w_val = np.asarray(w.value, dtype=float)
    omega0 = np.zeros_like(v_population)
    for s in range(S):
        mask = disagg_to_agg == s
        omega0[mask] = v_population[mask] * w_val[s]
    omega0 = np.clip(omega0, 0.0, None)
    total = omega0.sum()
    return omega0 / total if total > 0 else omega0


def _solve_mlsc_cvxpy(
    inputs: MLSCInputs, Q: np.ndarray, penalty_scale: float, solver: Any,
) -> Tuple[np.ndarray, str]:
    """cvxpy escape hatch: the original SCS/CLARABEL solve (warm-started)."""
    import cvxpy as cp

    T0 = inputs.T0
    Y_pre = inputs.Y_agg_treated[:T0]
    X_pre = inputs.X_disagg[:T0, :]
    omega = cp.Variable(inputs.M)
    try:
        omega.value = _classical_sc_warm_start(
            Y_pre=Y_pre, X_disagg_pre=X_pre, v_population=inputs.v_population,
            disagg_to_agg=inputs.disagg_to_agg, solver=solver,
        )
    except Exception:  # pragma: no cover - warm start is a numerical hint only
        pass
    objective_terms = [cp.sum_squares(Y_pre - X_pre @ omega)]
    if penalty_scale > 0:
        objective_terms.append(penalty_scale * cp.quad_form(omega, cp.psd_wrap(Q)))
    problem = cp.Problem(cp.Minimize(sum(objective_terms)),
                         [cp.sum(omega) == 1, omega >= 0])
    problem.solve(solver=solver or cp.SCS, warm_start=True)
    if omega.value is None:  # pragma: no cover - defensive: cvxpy returned no solution
        raise RuntimeError(
            f"cvxpy failed to solve the mlSC QP (status={problem.status})."
        )
    return np.asarray(omega.value, dtype=float), str(problem.status)


def solve_mlsc(
    inputs: MLSCInputs,
    Q: np.ndarray,
    lambda_val: float,
    sigma_y2: float,
    solver: Any = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Solve the mlSC QP.

    Parameters
    ----------
    inputs : MLSCInputs
        Pre-processed two-level panel.
    Q : np.ndarray
        Block-diagonal penalty matrix of shape ``(M, M)`` (used only by the
        cvxpy escape hatch; the native path rebuilds the square-root factor).
    lambda_val : float
        Penalty value (already chosen via heuristic or fixed).
    sigma_y2 : float
        Outcome variance, the penalty's scale factor so that ``lambda`` is
        scale-invariant as in the paper.
    solver : Any
        cvxpy solver. ``None`` (default) uses the native active-set simplex QP;
        any explicit value routes through cvxpy with that solver.

    Returns
    -------
    omega : np.ndarray
        Optimal disaggregate weights, length ``M``.
    aggregate_weights : np.ndarray
        Implied ``w_s = sum_c omega_sc``, length ``S``.
    status : str
        Solver status string.
    """
    penalty_scale = float(lambda_val) * float(sigma_y2)

    if solver is not None:
        omega_val, status = _solve_mlsc_cvxpy(inputs, Q, penalty_scale, solver)
    else:
        T0 = inputs.T0
        Y_pre = inputs.Y_agg_treated[:T0]
        X_pre = inputs.X_disagg[:T0, :]
        B, A = _augmented_design(X_pre, Y_pre, inputs, penalty_scale)
        from ..bilevel.active_set import solve_simplex_qp

        omega_val, info = solve_simplex_qp(B, A, return_info=True)
        if info["converged"]:
            status = "optimal"
        else:  # pragma: no cover - degenerate fallback to cvxpy
            omega_val, status = _solve_mlsc_cvxpy(inputs, Q, penalty_scale, None)

    omega_val = np.clip(np.asarray(omega_val, dtype=float), 0.0, None)
    total = omega_val.sum()
    if total > 0:
        omega_val = omega_val / total

    return omega_val, _aggregate(omega_val, inputs.disagg_to_agg), status
