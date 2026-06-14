"""Batched L2-relaxation solver via a single OSQP factorization.

When several treated units share the **same** control pool (the
multiple-treated-units PDA), the L2-relaxation primal

    min_beta  (1/2) ||beta||^2   s.t.   || eta_j - Sigma beta ||_inf <= tau

has the *same* ``Sigma = X'X/T1`` for every treated unit ``j`` and every
penalty ``tau`` -- only the moment vector ``eta_j`` and the bound ``tau``
change. OSQP factorises its KKT system from ``(P = I, A = Sigma)`` **once**; we
then sweep every ``(j, tau)`` by updating only the constraint bounds
``l = eta_j - tau``, ``u = eta_j + tau`` and re-solving warm-started -- no
re-factorisation. This turns hundreds of conic solves into hundreds of cheap
ADMM updates (and matches a per-problem ``cvxpy`` solve to solver precision).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def l2_relax_batch(
    Sigma: np.ndarray, Eta: np.ndarray, taus: np.ndarray,
    eps: float = 1e-5, max_iter: int = 4000,
) -> np.ndarray:
    """Solve the L2-relaxation primal for every ``(unit, tau)``.

    Parameters
    ----------
    Sigma : np.ndarray
        Shared ``(N, N)`` second-moment matrix.
    Eta : np.ndarray
        ``(N, J)`` moment vectors, one column per treated unit.
    taus : np.ndarray
        ``(K,)`` penalty grid.
    eps, max_iter : float, int
        OSQP tolerance and iteration cap (warm-started, no polish).

    Returns
    -------
    np.ndarray
        ``(J, K, N)`` coefficients ``beta[j, k]`` for unit ``j`` at ``taus[k]``.
    """
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the batched L2-relaxation solver needs `osqp` and `scipy`."
        ) from exc

    Sigma = np.asarray(Sigma, dtype=float)
    Eta = np.asarray(Eta, dtype=float)
    taus = np.asarray(taus, dtype=float)
    N, J = Eta.shape
    K = taus.shape[0]
    order = np.argsort(-taus)                 # descending -> warm-start path

    prob = osqp.OSQP()
    prob.setup(
        P=sp.eye(N, format="csc"), q=np.zeros(N),
        A=sp.csc_matrix(Sigma), l=Eta[:, 0] - taus[0], u=Eta[:, 0] + taus[0],
        eps_abs=eps, eps_rel=eps, max_iter=max_iter,
        polish=False, warm_starting=True, verbose=False,
    )
    out = np.zeros((J, K, N))
    for j in range(J):
        for k in order:
            prob.update(l=Eta[:, j] - taus[k], u=Eta[:, j] + taus[k])
            res = prob.solve()
            if res.x is not None and np.all(np.isfinite(res.x)):
                out[j, k] = res.x
    return out


# Tight OSQP tolerance for the single-treated path: the benchmarks pin
# value-for-value results, so we solve the unique primal optimum to ~1e-8, which
# matches an interior-point (cvxpy/CLARABEL) solve. (At this tolerance OSQP's
# active-set polishing changes nothing, so it stays off -- and silent.)
_SINGLE_EPS = 1e-9
_SINGLE_MAX_ITER = 50000


def l2_relax_solve(
    Sigma: np.ndarray, eta: np.ndarray, tau: float,
) -> np.ndarray:
    """One L2-relaxation primal solve (standardised scale) via tight OSQP.

    Returns the ``(N,)`` coefficient vector solving
    ``min ||beta||^2 / 2  s.t.  ||eta - Sigma beta||_inf <= tau``.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    return l2_relax_grid(Sigma, eta, np.asarray([float(tau)]))[0]


def l2_relax_grid(
    Sigma: np.ndarray, eta: np.ndarray, taus: np.ndarray,
) -> np.ndarray:
    """L2-relaxation primal for one unit across a ``tau`` grid (shared Sigma).

    Returns ``(K, N)`` coefficients, one row per ``taus[k]``. A single KKT
    factorization is reused across the grid (the cross-validation hot loop).
    """
    eta = np.asarray(eta, dtype=float).ravel()
    return l2_relax_batch(
        Sigma, eta[:, None], np.asarray(taus, dtype=float),
        eps=_SINGLE_EPS, max_iter=_SINGLE_MAX_ITER,
    )[0]
