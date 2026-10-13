"""The profiled HSC metric and the donor-weight quadratic program.

Implements the profiled representation (Proposition 1 of Liu & Xu, 2026).
For a fixed allocation ``rho in (0, 1)`` and smoothness order ``q in {1, 2}``:

    lambda_rho = rho / (1 - rho)
    K_q        = D_q' D_q                       (roughness matrix)
    S_{rho,q}  = (I + lambda_rho K_q)^{-1}       (smoother)
    W_{rho,q}  = (1 / rho) (I - S_{rho,q})       (donor-matching metric)

with the boundary cases

    rho = 0:  S = I,    W = K_q                  (SC on q-th differences)
    rho = 1:  S = P_0,  W = I - P_0              (SC on levels + intercept/trend)

where ``P_0`` projects onto ``Null(K_q)`` (constants for q=1, linear trends
for q=2). Donor weights then solve a ridge-regularized simplex QP under the
metric ``W_{rho,q}``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError
from ..solvers.active_set import solve_simplex_qp


def difference_operator(T: int, q: int) -> np.ndarray:
    """``q``-th order difference operator ``D_q`` of shape ``(T - q, T)``."""
    if q not in (1, 2):
        raise ValueError(f"HSC smoothness order q must be 1 or 2; got {q}.")
    if T <= q:
        raise ValueError(f"Need T > q to form D_q; got T={T}, q={q}.")
    return np.diff(np.eye(T), n=q, axis=0)


def roughness_matrix(T: int, q: int) -> np.ndarray:
    """Roughness matrix ``K_q = D_q' D_q`` of shape ``(T, T)``."""
    D = difference_operator(T, q)
    return D.T @ D


def smoother_and_metric(T: int, q: int, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(S_{rho,q}, W_{rho,q})`` for the profiled HSC problem.

    Parameters
    ----------
    T : int
        Length of the (pre-treatment) series.
    q : int
        Smoothness order (1 or 2).
    rho : float
        Allocation parameter in ``[0, 1]``.

    Returns
    -------
    S : np.ndarray
        Smoother that extracts the smooth residual ``E`` from the discrepancy.
    W : np.ndarray
        Symmetric PSD donor-matching metric (a seminorm).
    """

    if rho <= 0.0:
        return np.eye(T), roughness_matrix(T, q)
    if rho >= 1.0:
        cols = [np.ones(T)]
        if q == 2:
            cols.append(np.arange(T, dtype=float))
        B = np.column_stack(cols)
        P0 = B @ np.linalg.solve(B.T @ B, B.T)
        return P0, np.eye(T) - P0

    lam = rho / (1.0 - rho)
    S = np.linalg.solve(np.eye(T) + lam * roughness_matrix(T, q), np.eye(T))
    W = (1.0 / rho) * (np.eye(T) - S)
    return S, W


def sdid_ridge_coefficient(X_pre: np.ndarray, n_post: int) -> float:
    """SDID-style absolute ridge coefficient on ``||omega||^2``.

    Following Liu & Xu (2026) §7 and Arkhangelsky et al. (2021), the donor
    penalty is ``zeta^2 * T0`` with ``zeta = T_post^{1/4} * sigma_dX``, where
    ``sigma_dX`` is the standard deviation of the first differences of the
    donor outcomes over the pre-period. This scales the ridge to the data and
    yields diversified (non-corner) donor weights.

    Parameters
    ----------
    X_pre : np.ndarray
        Donor pre-treatment matrix, shape ``(T0, N)``.
    n_post : int
        Number of post-treatment periods.

    Returns
    -------
    float
        Absolute coefficient on ``||omega||^2`` (``zeta^2 * T0``).
    """

    T0 = X_pre.shape[0]
    sigma_dX = float(np.std(np.diff(X_pre, axis=0)))
    zeta = (max(int(n_post), 1) ** 0.25) * sigma_dX
    return float((zeta ** 2) * T0)


def fit_donor_weights(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    ridge: float = 1e-6,
    ridge_abs: Optional[float] = None,
    solver: Optional[object] = None,
) -> np.ndarray:
    """Solve the simplex-constrained, metric-weighted donor QP.

    ``argmin_{omega in simplex} (Y - X omega)' W (Y - X omega) + c ||omega||^2``,
    where the ridge coefficient ``c`` is either ``ridge_abs`` (absolute, e.g.
    the SDID-style ``zeta^2 T0`` from :func:`sdid_ridge_coefficient`) when
    supplied, or the relative ``ridge * trace(X'WX) / N`` otherwise.

    Parameters
    ----------
    X : np.ndarray
        Donor matrix, shape ``(T, N)``.
    Y : np.ndarray
        Treated outcome, shape ``(T,)``.
    W : np.ndarray
        Donor-matching metric, shape ``(T, T)``.
    ridge : float
        Relative ridge coefficient (used only when ``ridge_abs`` is ``None``).
    ridge_abs : float, optional
        Absolute ridge coefficient on ``||omega||^2``. Overrides ``ridge``.
    solver : optional
        CVXPY solver; defaults to Clarabel.

    Returns
    -------
    np.ndarray
        Non-negative donor weights summing to one, shape ``(N,)``.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=float)

    if X.ndim != 2:
        raise MlsynthEstimationError("HSC: X must be 2D (periods, donors).")
    N = X.shape[1]
    if N == 0:
        raise MlsynthEstimationError(
            "HSC: no donors, so sum(omega) == 1 has no solution."
        )
    T = X.shape[0]
    if Y.shape[0] != T or W.shape != (T, T):
        raise MlsynthEstimationError(
            f"HSC: shapes disagree: X has {T} rows, Y has {Y.shape[0]} and W is "
            f"{W.shape}, expected ({T}, {T})."
        )

    # The metric is a seminorm, so factor it as W = L'L and carry the ridge as
    # design rows with no target (Zou and Hastie 2005, Lemma 1):
    #
    #     B = [L X ; sqrt(c) I],  A = [L Y ; 0]
    #
    # gives B'B = X'WX + cI and B'A = X'WY, which is this program written as a
    # residual. Going through W and not the Gram X'WX is what keeps it accurate:
    # the Gram squares the design's condition number, while a symmetric
    # eigendecomposition of W is backward stable. Negative eigenvalues are
    # rounding on a PSD matrix, so they clamp to zero.
    W_sym = 0.5 * (W + W.T)
    evals, evecs = np.linalg.eigh(W_sym)
    L = np.sqrt(np.maximum(evals, 0.0))[:, None] * evecs.T
    LX = L @ X

    # trace(X'WX) = ||LX||_F^2, so the relative ridge needs no Gram either.
    coef = (
        float(ridge_abs)
        if ridge_abs is not None
        else ridge * (float(np.sum(LX * LX)) / max(N, 1))
    )
    if coef < 0.0:
        raise MlsynthEstimationError(
            f"HSC: the ridge coefficient is negative ({coef}); it penalises "
            "||omega||^2 and cannot be."
        )

    B = np.vstack([LX, np.sqrt(coef) * np.eye(N)])
    A = np.concatenate([L @ Y, np.zeros(N)])
    try:
        return solve_simplex_qp(B, A)
    except ValueError as exc:  # pragma: no cover - the guards above precede it
        raise MlsynthEstimationError(
            f"HSC donor-weight QP failed to solve: {exc}"
        ) from exc
