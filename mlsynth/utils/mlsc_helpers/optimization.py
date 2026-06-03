"""MlSC QP solver.

Solves Equation 5.2 of Bottmer (2025)::

    min over omega in R^M
        || Y_agg_treated_pre  -  X_disagg_pre @ omega ||^2
        + lambda * sigma_y^2 * omega^T Q omega
    s.t.  1^T omega = 1,   omega >= 0.

The objective is a convex QP. We seed cvxpy with the warm-start trick the
upstream package uses (Bottmer's reference implementation): solve the
classical aggregate-level SC first, then expand each aggregate weight
``w_s`` proportionally to the population shares ``v_s`` to produce a feasible
disaggregate starting point. This typically takes the SCS solver a handful
of iterations to reach optimality even on noisy panels.
"""

from __future__ import annotations

from typing import Any, Tuple

import cvxpy as cp
import numpy as np

from .structures import MLSCInputs


def _classical_sc_warm_start(
    Y_pre: np.ndarray,
    X_disagg_pre: np.ndarray,
    v_population: np.ndarray,
    disagg_to_agg: np.ndarray,
    solver: Any,
) -> np.ndarray:
    """Solve the aggregate-level classical SC and expand to disaggregate weights.

    ``Y_pre`` is the pre-treatment aggregate treated series; ``X_disagg_pre``
    is the pre-treatment disaggregate control matrix. We first build the
    implied aggregate control matrix ``X_agg_pre`` by population-weighted
    aggregation within each block, solve classical SC for state weights
    ``w_s``, then return ``omega_sc = v_sc * w_s``.
    """

    S = int(disagg_to_agg.max() + 1)
    T0 = Y_pre.shape[0]
    X_agg_pre = np.zeros((T0, S))
    for s in range(S):
        mask = disagg_to_agg == s
        X_agg_pre[:, s] = X_disagg_pre[:, mask] @ v_population[mask]

    w = cp.Variable(S)
    objective = cp.Minimize(cp.sum_squares(Y_pre - X_agg_pre @ w))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver or cp.SCS)
    if w.value is None:
        # Fallback: uniform aggregate weights.
        return v_population / max(S, 1)

    w_val = np.asarray(w.value, dtype=float)
    # omega_sc = v_sc * w_s, in disaggregate order.
    omega0 = np.zeros_like(v_population)
    for s in range(S):
        mask = disagg_to_agg == s
        omega0[mask] = v_population[mask] * w_val[s]
    # Numerical clean-up.
    omega0 = np.clip(omega0, 0.0, None)
    total = omega0.sum()
    if total > 0:
        omega0 = omega0 / total
    return omega0


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
        Block-diagonal penalty matrix of shape ``(M, M)``.
    lambda_val : float
        Penalty value (already chosen via heuristic or fixed).
    sigma_y2 : float
        Outcome variance, used as the penalty's scale factor so that
        ``lambda`` is scale-invariant as in the paper.
    solver : Any
        cvxpy solver. ``None`` -> ``cp.SCS``.

    Returns
    -------
    omega : np.ndarray
        Optimal disaggregate weights, length ``M``.
    aggregate_weights : np.ndarray
        Implied ``w_s = sum_c omega_sc``, length ``S``.
    status : str
        cvxpy solver status string.
    """

    T0 = inputs.T0
    Y_pre = inputs.Y_agg_treated[:T0]
    X_pre = inputs.X_disagg[:T0, :]

    M = inputs.M
    omega = cp.Variable(M)

    # Warm start (numerical only — cvxpy doesn't require feasibility here).
    try:
        omega.value = _classical_sc_warm_start(
            Y_pre=Y_pre,
            X_disagg_pre=X_pre,
            v_population=inputs.v_population,
            disagg_to_agg=inputs.disagg_to_agg,
            solver=solver,
        )
    except Exception:
        pass

    penalty_scale = float(lambda_val) * float(sigma_y2)
    objective_terms = [cp.sum_squares(Y_pre - X_pre @ omega)]
    if penalty_scale > 0:
        objective_terms.append(penalty_scale * cp.quad_form(omega, cp.psd_wrap(Q)))

    objective = cp.Minimize(sum(objective_terms))
    constraints = [cp.sum(omega) == 1, omega >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver or cp.SCS, warm_start=True)

    if omega.value is None:
        raise RuntimeError(
            f"cvxpy failed to solve the mlSC QP (status={problem.status})."
        )

    omega_val = np.asarray(omega.value, dtype=float)
    # Numerical clean-up: kill small negatives, renormalize.
    omega_val = np.clip(omega_val, 0.0, None)
    total = omega_val.sum()
    if total > 0:
        omega_val = omega_val / total

    S = int(inputs.disagg_to_agg.max() + 1)
    aggregate_weights = np.zeros(S)
    for s in range(S):
        aggregate_weights[s] = omega_val[inputs.disagg_to_agg == s].sum()

    return omega_val, aggregate_weights, str(problem.status)
