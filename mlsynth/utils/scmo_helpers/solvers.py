"""Synthetic-control weight solvers for SCMO (NumPy/cvxpy, no estutils.Opt).

The simplex solver replaces ``Opt.SCopt(scm_model_type="SIMPLEX")``: it
minimizes the squared imbalance between the treated and donor matching
vectors subject to the convex-combination constraint (weights >= 0, sum 1),
exactly the Tian-Lee-Panchenko / Abadie program.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..bilevel.active_set import solve_simplex_qp


def simplex_weights(Z_treated: np.ndarray, Z_donors: np.ndarray) -> np.ndarray:
    """Convex (simplex) SC weights minimizing ``||Z_treated - Z_donors' w||^2``.

    Parameters
    ----------
    Z_treated : np.ndarray
        Treated matching vector, shape ``(P,)``.
    Z_donors : np.ndarray
        Donor matching matrix, shape ``(J, P)``.

    Returns
    -------
    np.ndarray
        Donor weights, shape ``(J,)``; non-negative and summing to one.
    """
    try:
        w_hat = np.clip(solve_simplex_qp(Z_donors.T, Z_treated), 0.0, None)
    except Exception as exc:
        raise MlsynthEstimationError(
            "SCMO simplex weight solve failed: the donor matching matrix may "
            f"be degenerate or ill-conditioned ({exc})."
        ) from exc
    total = w_hat.sum()
    return w_hat / total if total > 0 else w_hat            # exact simplex (sum == 1)
