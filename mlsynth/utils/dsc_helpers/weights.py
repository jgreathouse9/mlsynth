"""Simplex-constrained weight solver for Distributional Synthetic Controls.

For each pre-period :math:`t \\in \\mathcal T_0`, DSC solves

.. math::

   \\widehat w_t = \\arg\\min_{w \\in \\mathcal H}
                    \\bigl\\| \\widetilde Y_t\\, w - \\widehat Y_{1t} \\bigr\\|_2^2,
   \\qquad
   \\mathcal H = \\bigl\\{ w \\in [0, 1]^J : \\mathbf 1^\\top w = 1 \\bigr\\},

where :math:`\\widetilde Y_t` is the :math:`M \\times J` donor
pseudo-sample matrix and :math:`\\widehat Y_{1t}` is the
:math:`M \\times 1` treated pseudo-sample vector (Zhang, Zhang &
Zhang 2026 eq. 3; the loss is the squared 2-Wasserstein distance
approximated by Monte Carlo / QMC).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ...exceptions import MlsynthEstimationError


def solve_simplex_weights(
    donor_matrix: np.ndarray,
    treated_vec: np.ndarray,
) -> np.ndarray:
    """Return the simplex-constrained least-squares weight vector.

    Uses sequential least-squares programming (SLSQP) -- the same
    convex QP that the classical synthetic control of
    Abadie-Diamond-Hainmueller (2010) solves. We avoid pulling in
    cvxpy for this single helper since the problem is small and
    SLSQP is shipped with SciPy.

    Parameters
    ----------
    donor_matrix : np.ndarray
        :math:`(M, J)` design matrix.
    treated_vec : np.ndarray
        Length-``M`` target.

    Returns
    -------
    np.ndarray
        Length-``J`` weight vector with ``w >= 0`` and ``sum(w) == 1``.
    """
    if donor_matrix.ndim != 2 or treated_vec.ndim != 1:
        raise MlsynthEstimationError(
            "donor_matrix must be 2-D and treated_vec must be 1-D."
        )
    if donor_matrix.shape[0] != treated_vec.shape[0]:
        raise MlsynthEstimationError(
            "donor_matrix and treated_vec must have the same number of rows."
        )

    J = donor_matrix.shape[1]
    w0 = np.full(J, 1.0 / J)

    def loss(w: np.ndarray) -> float:
        diff = donor_matrix @ w - treated_vec
        return float(diff @ diff)

    def loss_grad(w: np.ndarray) -> np.ndarray:
        diff = donor_matrix @ w - treated_vec
        return 2.0 * donor_matrix.T @ diff

    result = minimize(
        fun=loss,
        x0=w0,
        jac=loss_grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        # SLSQP occasionally reports failure even when the solution is
        # within tolerance; check feasibility explicitly before raising.
        w = result.x
        if (
            (w < -1e-6).any()
            or abs(w.sum() - 1.0) > 1e-4
        ):
            raise MlsynthEstimationError(
                f"Simplex QP failed: {result.message}"
            )

    w = np.clip(result.x, 0.0, 1.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def wasserstein_loss_at_weights(
    donor_matrix: np.ndarray,
    treated_vec: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Squared 2-Wasserstein loss :math:`\\|\\widetilde Y_t w - \\widehat Y_{1t}\\|_2^2 / M`.

    The :math:`1/M` normalisation makes this comparable across periods
    with different grid sizes.
    """
    diff = donor_matrix @ weights - treated_vec
    return float((diff @ diff) / max(donor_matrix.shape[0], 1))
