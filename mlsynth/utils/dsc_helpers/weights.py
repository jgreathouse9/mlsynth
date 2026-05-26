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

from ...exceptions import MlsynthEstimationError


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex.

    Implements the exact :math:`O(J \\log J)` algorithm of Duchi,
    Shalev-Shwartz, Singer & Chandra (2008): sort, find the threshold
    via the cumulative sum, and soft-threshold. Returns ``w >= 0`` with
    ``sum(w) == 1``.
    """
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho_idx = np.nonzero(u - css / np.arange(1, v.size + 1) > 0)[0]
    rho = rho_idx[-1] if rho_idx.size else 0
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def solve_simplex_weights(
    donor_matrix: np.ndarray,
    treated_vec: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return the simplex-constrained least-squares weight vector.

    Solves the convex program

    .. math::

       \\widehat w = \\arg\\min_{w \\in \\mathcal H}
                     \\| \\widetilde Y_t\\, w - \\widehat Y_{1t} \\|_2^2,
       \\qquad
       \\mathcal H = \\{ w \\ge 0 : \\mathbf 1^\\top w = 1 \\},

    by **accelerated projected gradient descent** (FISTA; Beck &
    Teboulle 2009) with the exact simplex projection of Duchi et al.
    (2008). This replaces an earlier SLSQP solver that failed
    (``"Positive directional derivative for linesearch"``) once the
    donor pool grew past a few dozen units -- precisely the regime of
    Gunsilius (2023, Section 6.1), where the method is meant to use
    tens to hundreds of donors. The reference DiSCo R package solves the
    same program with a dedicated constrained least-squares routine
    (``pracma::lsqlincon``); projected gradient is its dependency-free
    analogue and returns the identical optimum (the objective is convex
    with a unique minimum value over the simplex).

    Parameters
    ----------
    donor_matrix : np.ndarray
        :math:`(M, J)` design matrix -- donor quantile functions
        evaluated on the grid.
    treated_vec : np.ndarray
        Length-``M`` target quantile function.
    max_iter : int
        Maximum FISTA iterations.
    tol : float
        Relative objective-change stopping tolerance.

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

    A = np.asarray(donor_matrix, dtype=float)
    b = np.asarray(treated_vec, dtype=float)
    J = A.shape[1]
    if J == 1:
        return np.ones(1)

    AtA = A.T @ A
    Atb = A.T @ b
    # Lipschitz constant of the gradient of ||A w - b||^2 is 2 * lambda_max(AtA).
    lip = 2.0 * float(np.linalg.norm(AtA, 2))
    step = 1.0 / max(lip, 1e-12)

    w = np.full(J, 1.0 / J)
    y = w.copy()
    t = 1.0
    prev_obj = np.inf
    for _ in range(max_iter):
        grad = 2.0 * (AtA @ y - Atb)
        w_new = _project_to_simplex(y - step * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
        diff = A @ w - b
        obj = float(diff @ diff)
        if abs(prev_obj - obj) <= tol * (1.0 + abs(obj)):
            break
        prev_obj = obj

    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(J, 1.0 / J)


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
