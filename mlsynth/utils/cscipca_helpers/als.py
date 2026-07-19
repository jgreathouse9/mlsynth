"""Alternating-least-squares core for CSC-IPCA (Wang 2024, Sec. 3).

The structural model for the untreated potential outcome is

.. math::

   Y_{it} = (X_{it}\\,\\Gamma)\\,F_t' + \\epsilon_{it},

with an ``L x K`` mapping matrix ``\\Gamma`` and ``K`` latent factors
``F_t``. Because the loadings are ``\\Lambda_{it} = X_{it}\\Gamma`` rather than
a free ``\\Lambda_i``, eigendecomposition does not apply and the objective

.. math::

   \\min_{\\Gamma, F} \\sum_{i,t} \\big(Y_{it} - (X_{it}\\Gamma) F_t'\\big)^2

is minimized by alternating least squares: with ``F`` fixed the ``\\Gamma``
subproblem is a single linear solve of the ``LK`` normal equations; with
``\\Gamma`` fixed each ``F_t`` is a ``K``-vector least-squares solve. This
module implements those two steps in vectorized form (the reference
``CongWang141/JMP`` drives them with an explicit ``N x T`` Kronecker loop; the
einsum forms here are algebraically identical and cross-validated in the
tests).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _svd_init(Y: np.ndarray, K: int) -> np.ndarray:
    """Top-``K`` SVD initialization of the factors, ``F0`` shape ``(K, T)``.

    Mirrors the reference initialization ``F0 = diag(S) @ V'`` from the top-K
    singular triplets of the ``(N, T)`` outcome matrix, in descending order.
    """
    _, S, Vt = np.linalg.svd(np.asarray(Y, dtype=float), full_matrices=False)
    k = min(K, S.shape[0])
    F0 = (S[:k, None] * Vt[:k])           # (k, T)
    if k < K:  # pragma: no cover - guarded upstream (K <= min(N, T))
        F0 = np.vstack([F0, np.zeros((K - k, F0.shape[1]))])
    return F0


def solve_gamma(Y: np.ndarray, X: np.ndarray, F: np.ndarray, K: int) -> np.ndarray:
    """Solve the ``Gamma`` subproblem for fixed factors ``F``.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix, shape ``(N, T)``.
    X : np.ndarray
        Covariate cube, shape ``(N, T, L)``.
    F : np.ndarray
        Fixed factors, shape ``(K, T)``.
    K : int
        Number of factors.

    Returns
    -------
    np.ndarray
        Estimated mapping matrix, shape ``(L, K)``.
    """
    N, T, L = X.shape
    # numer[l, k] = sum_{i,t} Y_it X_itl F_kt
    numer = np.einsum("it,itl,kt->lk", Y, X, F, optimize=True).reshape(L * K)
    # denom[(l,k),(m,j)] = sum_t (sum_i X_itl X_itm) F_kt F_jt
    G = np.einsum("itl,itm->tlm", X, X, optimize=True)          # (T, L, L)
    denom = np.einsum("tlm,kt,jt->lkmj", G, F, F, optimize=True).reshape(L * K, L * K)
    # Least squares, not a plain solve: collinear covariates (e.g. log GDP,
    # log GDP-per-capita and log population) make ``denom`` rank-deficient, so a
    # direct inverse is singular. The counterfactual (X Gamma) F is invariant to
    # which Gamma is picked among the equivalent ones, so the minimum-norm
    # lstsq solution gives the same fit robustly (matching the reference's
    # ``_mldivide``).
    gamma, *_ = np.linalg.lstsq(denom, numer, rcond=None)
    return gamma.reshape(L, K)


def solve_factors(Y: np.ndarray, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Solve each ``F_t`` for a fixed mapping matrix ``Gamma``.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix, shape ``(N, T)``.
    X : np.ndarray
        Covariate cube, shape ``(N, T, L)``.
    gamma : np.ndarray
        Fixed mapping matrix, shape ``(L, K)``.

    Returns
    -------
    np.ndarray
        Estimated factors, shape ``(K, T)``.
    """
    T = X.shape[1]
    XG = np.einsum("itl,lk->itk", X, gamma, optimize=True)      # (N, T, K)
    denom = np.einsum("itk,itj->tkj", XG, XG, optimize=True)    # (T, K, K)
    numer = np.einsum("itk,it->tk", XG, Y, optimize=True)       # (T, K)
    # Per-period least squares (matching the reference): a rank-deficient
    # per-period system -- possible under collinear covariates -- would make a
    # direct solve singular, so use the minimum-norm lstsq solution.
    F = np.empty((T, XG.shape[2]))
    for t in range(T):
        F[t], *_ = np.linalg.lstsq(denom[t], numer[t], rcond=None)
    return F.T                                                   # (K, T)


def als_estimate(
    Y: np.ndarray, X: np.ndarray, K: int, max_iter: int = 100, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Alternating least squares for the factors and mapping matrix.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix, shape ``(N, T)``.
    X : np.ndarray
        Covariate cube, shape ``(N, T, L)``.
    K : int
        Number of latent factors.
    max_iter : int
        Maximum ALS iterations.
    tol : float
        Convergence tolerance on ``max(|Delta Gamma|, |Delta F|)``.

    Returns
    -------
    tuple
        ``(F, gamma, n_iter, converged)`` -- factors ``(K, T)``, mapping
        ``(L, K)``, iteration count, and whether ``tol`` was met.
    """
    _, _, L = X.shape
    F0 = _svd_init(Y, K)
    gamma0 = np.zeros((L, K))
    # Convergence is measured on the fitted values (X Gamma) F, not on the raw
    # (Gamma, F): the bilinear objective is invariant to the Gamma -> Gamma R,
    # F -> R^{-1} F rotation, so the parameters can drift within that subspace
    # forever while the fit -- the thing the counterfactual depends on -- has
    # already converged.
    fit0 = counterfactual(X, gamma0, F0)
    n_iter, converged = 0, False
    for n_iter in range(1, max_iter + 1):
        gamma1 = solve_gamma(Y, X, F0, K)
        F1 = solve_factors(Y, X, gamma1)
        fit1 = counterfactual(X, gamma1, F1)
        delta = np.abs(fit1 - fit0).max()
        F0, gamma0, fit0 = F1, gamma1, fit1
        if delta <= tol:
            converged = True
            break
    return F0, gamma0, n_iter, converged


def normalize(gamma: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate ``(Gamma, F)`` to the identifiable normalization.

    The bilinear objective is invariant to ``Gamma -> Gamma R``,
    ``F -> R^{-1} F`` for any invertible ``R`` (Sec. 3, Step 3). Following
    Connor-Korajczyk (1993) / Bai-Ng (2002), this fixes ``Gamma'Gamma = I_K``
    and ``FF'/T`` diagonal so the estimates are comparable across fits. The
    counterfactual ``(X Gamma) F`` is unchanged by the rotation.
    """
    import scipy.linalg as sla

    R1 = sla.cholesky(gamma.T @ gamma)
    R2, _, _ = sla.svd(R1 @ F @ F.T @ R1.T)
    gamma_norm = sla.lstsq(R1.T, gamma.T)[0].T @ R2   # (gamma / R1) @ R2
    F_norm = sla.lstsq(R2, R1 @ F)[0]                 # R2 \ (R1 @ F)
    return gamma_norm, F_norm


def counterfactual(X: np.ndarray, gamma: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Imputed outcome ``hat Y_it = (X_it Gamma) F_t``.

    Parameters
    ----------
    X : np.ndarray
        Covariate cube, shape ``(N, T, L)``.
    gamma : np.ndarray
        Mapping matrix, shape ``(L, K)``.
    F : np.ndarray
        Factors, shape ``(K, T)``.

    Returns
    -------
    np.ndarray
        Imputed outcome matrix, shape ``(N, T)``.
    """
    return np.einsum("itl,lk,kt->it", X, gamma, F, optimize=True)
