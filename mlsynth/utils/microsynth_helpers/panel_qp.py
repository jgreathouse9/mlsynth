"""Panel-method weight QP for MicroSynth (Robbins et al. ``microsynth``).

Faithful port of ``microsynth::my.qp`` (the ``LowRankQP`` solve in the R
package's ``weights.r``). When the user supplies ``match.out`` (lagged
outcomes), ``microsynth`` chooses control weights by a **non-negative
quadratic program**:

.. math::

   \\min_{w \\ge 0}\\ \\tfrac12 \\bigl\\| L_C^\\top w - \\ell_T \\bigr\\|^2
   \\quad\\text{s.t.}\\quad H_C^\\top w = h_T,

where

* :math:`H_C` (``hard_C``) stacks an intercept and the time-invariant
  covariates of the controls; :math:`h_T` (``hard_targets``) are the treated
  group's column **totals** (the intercept target is the treated count, so the
  weights sum to it) -- these are matched **exactly** (the equality block);
* :math:`L_C` (``soft_C``) holds each control's pre-intervention outcome
  values and :math:`\\ell_T` (``soft_targets``) the treated totals -- these are
  fit by **least squares** (the QP objective).

Unlike the exact-balance covariate constraints, this objective is rank
``\\le`` (number of lagged outcomes), so over a large control pool the optimum
is a high-dimensional face rather than a point -- the counterfactual is
**not identified** by the constraints alone (``LowRankQP`` merely returns
whichever interior-point iterate it lands on). We therefore add a strictly
convex ridge :math:`\\tfrac{\\rho}{2}\\|w\\|^2`, which selects the unique
**minimum-norm / maximum-ESS** point on that face -- the most diffuse
synthetic control consistent with exact covariate balance and the best
lagged-outcome fit -- giving a reproducible, well-defined estimand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class PanelQPSolution:
    """Result of the MicroSynth panel weight QP."""

    w: np.ndarray
    hard_residual: float          # max |H_C^T w - h_T| (should be ~0)
    soft_residual: float          # max |L_C^T w - l_T| (NaN if no soft block)
    objective: float
    converged: bool
    status: str


def solve_panel_qp(
    hard_C: np.ndarray,
    hard_targets: np.ndarray,
    soft_C: Optional[np.ndarray] = None,
    soft_targets: Optional[np.ndarray] = None,
    *,
    ridge: float = 1e-6,
    solver: Optional[str] = None,
) -> PanelQPSolution:
    """Non-negative panel weights: exact hard balance + LS soft fit + ridge.

    Parameters
    ----------
    hard_C : np.ndarray
        Control hard-constraint matrix, shape ``(n_C, k)`` -- typically an
        intercept column followed by the time-invariant covariates.
    hard_targets : np.ndarray
        Treated-group column totals of the same columns, shape ``(k,)``. The
        intercept target equals the treated count, so the weights sum to it.
    soft_C : np.ndarray, optional
        Control lagged-outcome matrix, shape ``(n_C, m)``. If ``None``, the
        objective is the ridge alone (pure minimum-norm covariate balancing).
    soft_targets : np.ndarray, optional
        Treated lagged-outcome totals, shape ``(m,)``. Required when ``soft_C``
        is given.
    ridge : float
        Strictly-convex regularizer weight :math:`\\rho > 0` selecting the
        minimum-norm / maximum-ESS optimum (uniqueness).
    solver : str, optional
        cvxpy solver name; defaults to CLARABEL.

    Returns
    -------
    PanelQPSolution

    Raises
    ------
    MlsynthEstimationError
        If the QP is infeasible (treated totals unreachable by any non-negative
        count-constrained weighting) or the solver fails to converge.
    """
    hard_C = np.asarray(hard_C, dtype=float)
    hard_targets = np.asarray(hard_targets, dtype=float)
    n_C = hard_C.shape[0]
    if ridge <= 0:
        raise MlsynthEstimationError(
            "Panel QP ridge must be strictly positive for a unique solution."
        )

    w = cp.Variable(n_C, nonneg=True)
    obj = 0.5 * ridge * cp.sum_squares(w)
    if soft_C is not None:
        soft_C = np.asarray(soft_C, dtype=float)
        soft_targets = np.asarray(soft_targets, dtype=float)
        obj = obj + 0.5 * cp.sum_squares(soft_C.T @ w - soft_targets)
    constraints = [hard_C.T @ w == hard_targets]
    problem = cp.Problem(cp.Minimize(obj), constraints)

    try:
        problem.solve(solver=solver or cp.CLARABEL)
    except cp.error.SolverError as exc:                       # pragma: no cover
        raise MlsynthEstimationError(
            f"MicroSynth panel QP solver failed: {exc}"
        ) from exc

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise MlsynthEstimationError(
            "MicroSynth panel QP did not solve "
            f"(status={problem.status}). The treated group's covariate totals "
            "may be unreachable by any non-negative weighting of the controls "
            "(treated area outside the control convex cone)."
        )

    w_val = np.maximum(np.asarray(w.value, dtype=float), 0.0)
    hard_res = float(np.max(np.abs(hard_C.T @ w_val - hard_targets)))
    if soft_C is not None:
        soft_res = float(np.max(np.abs(soft_C.T @ w_val - soft_targets)))
    else:
        soft_res = float("nan")
    return PanelQPSolution(
        w=w_val,
        hard_residual=hard_res,
        soft_residual=soft_res,
        objective=float(problem.value),
        converged=problem.status == "optimal",
        status=str(problem.status),
    )
