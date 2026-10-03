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

from typing import Dict, List, Tuple

import numpy as np

# Contraction order per (subscripts, operand shapes). ``optimize=True`` re-runs
# numpy's greedy path search on every call, which on these operands costs more
# than the contraction: planning was 27 percent of a conformal run's time. The
# path is a function of the subscripts and the shapes alone, and ``True`` is
# ``"greedy"``, so caching it changes the planning and not the arithmetic -- the
# contraction order, and therefore the bits, are the ones ``optimize=True``
# picked. Shapes are part of the key because the greedy path is chosen for them;
# a key without them would reuse one panel's order on another's operands.
_EINSUM_PATHS: Dict[Tuple, List] = {}


def _einsum_path(subscripts: str, *operands: np.ndarray) -> List:
    """Cached greedy contraction path for ``subscripts`` on these shapes."""
    key = (subscripts,) + tuple(op.shape for op in operands)
    path = _EINSUM_PATHS.get(key)
    if path is None:
        path = np.einsum_path(subscripts, *operands, optimize="greedy")[0]
        _EINSUM_PATHS[key] = path
    return path


def _contract(subscripts: str, *operands: np.ndarray) -> np.ndarray:
    """``np.einsum(..., optimize=True)`` with the path search done once."""
    return np.einsum(subscripts, *operands,
                     optimize=_einsum_path(subscripts, *operands))


# Largest per-period condition number the batched LU in :func:`solve_factors` is
# trusted with. An LU solve carries a relative error of about ``cond * eps``, so
# ``1e4`` holds its disagreement with the per-period minimum-norm solution near
# ``1e-12``; measured over 3000 generated panels (including exactly singular
# periods and collinear covariates) the worst was 6.8e-13. Raising it to 1e12
# admits only 3.6 percent more panels to the fast path and costs five orders of
# magnitude of agreement, so the threshold sits at the knee and not at the
# rank-decision boundary ``lstsq`` itself uses.
_MAX_COND = 1.0e4


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
    numer = _contract("it,itl,kt->lk", Y, X, F).reshape(L * K)
    # denom[(l,k),(m,j)] = sum_t (sum_i X_itl X_itm) F_kt F_jt
    G = _contract("itl,itm->tlm", X, X)          # (T, L, L)
    denom = _contract("tlm,kt,jt->lkmj", G, F, F).reshape(L * K, L * K)
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
    XG = _contract("itl,lk->itk", X, gamma)      # (N, T, K)
    denom = _contract("itk,itj->tkj", XG, XG)    # (T, K, K)
    numer = _contract("itk,it->tk", XG, Y)       # (T, K)
    # One batched LU for all T periods when every period can carry it. The
    # systems are K x K -- two by two on the Brexit panel -- so the loop of
    # ``np.linalg.lstsq`` calls below spent most of its time in numpy's per-call
    # wrapper and not in LAPACK: 349 us for T = 22 against 14 us batched.
    #
    # The gate is not ``try: solve except LinAlgError``. ``solve`` raises only on
    # an exactly singular matrix, and the case collinear covariates actually
    # produce is a *nearly* singular one, where it returns an amplified solution
    # and reports nothing: on one generated panel it answered 7.2e227 where the
    # minimum-norm solution is 0.33. ``lstsq`` truncates the offending singular
    # value instead, which is the behaviour this function promises, so the LU is
    # used only where the two provably agree.
    # ``eigvalsh`` reads only the lower triangle, so it is the same matrix
    # ``solve`` factorizes only if the contraction is exactly symmetric. It is:
    # entries (k, j) and (j, k) are the same products -- float multiplication is
    # commutative to the bit -- reduced over the same i in the same order.
    # Averaging with the transpose first was a no-op on 4000 generated panels
    # across eight orders of magnitude of column scaling, so it is not done.
    eigvals = np.linalg.eigvalsh(denom)          # ascending; denom is symmetric PSD
    if (eigvals.shape[1]
            and np.all(eigvals[:, 0] > 0.0)
            and np.all(eigvals[:, 0] * _MAX_COND > eigvals[:, -1])):
        return np.linalg.solve(denom, numer[..., None])[..., 0].T
    # A rank-deficient or ill-conditioned period takes per-period least squares
    # (matching the reference), whose minimum-norm solution is defined there.
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
    return _contract("itl,lk,kt->it", X, gamma, F)
