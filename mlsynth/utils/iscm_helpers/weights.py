"""All-units synthetic-control weights for ISCM.

For every unit :math:`i`, ISCM constructs a synthetic control from the
*other* units by the usual constrained least-squares pre-period fit
(paper eq. 5):

.. math::

   \\widehat w_i = \\arg\\min_{w}
       \\sum_{t \\le T_0} \\Bigl( Y_{it} - \\sum_{j \\ne i} w_j Y_{jt} \\Bigr)^2,
   \\quad w_j \\ge 0,\\ \\sum_{j \\ne i} w_j = 1.

These per-unit weights are the starting point of ISCM (Powell's applied
procedure initialises with the traditional SCM); the fit metric and the
treatment-effect regression are built on top of them.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


def _one_unit_weights(donor_pre: np.ndarray, target_pre: np.ndarray) -> np.ndarray:
    """Simplex SC weights fitting ``target_pre`` from ``donor_pre``.

    ``donor_pre`` is ``(T0, N-1)``; ``target_pre`` is ``(T0,)``.
    """
    n_donors = donor_pre.shape[1]
    w = cp.Variable(n_donors, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(donor_pre @ w - target_pre))
    problem = cp.Problem(objective, [cp.sum(w) == 1])
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:
        raise MlsynthEstimationError(
            f"ISCM unit-weight solver failed: {exc}"
        ) from exc
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise MlsynthEstimationError(
            f"ISCM unit-weight QP did not converge (status={problem.status})."
        )
    return np.asarray(w.value, dtype=float)


def all_units_weights(Y: np.ndarray, T0: int) -> np.ndarray:
    """Return the ``(N, N)`` all-units SC weight matrix.

    Row ``i`` holds unit ``i``'s synthetic-control weights over the other
    units (``W[i, i] = 0``, each row non-negative and summing to one).

    Parameters
    ----------
    Y : np.ndarray
        Outcomes, shape ``(N, T)``.
    T0 : int
        Number of pre-treatment periods.
    """
    N = Y.shape[0]
    pre = Y[:, :T0]                       # (N, T0)
    W = np.zeros((N, N))
    for i in range(N):
        others = [j for j in range(N) if j != i]
        donor_pre = pre[others].T          # (T0, N-1)
        w = _one_unit_weights(donor_pre, pre[i])
        W[i, others] = w
    return W
