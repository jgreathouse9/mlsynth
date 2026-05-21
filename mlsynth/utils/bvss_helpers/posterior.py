"""Math primitives for the BVS-SS Metropolis-within-Gibbs sampler.

Implements Eqs. (4) and (5) of Xu & Zhou (2025), "Bayesian Synthetic
Control with a Soft Simplex Constraint" (arXiv:2503.06454), plus the
log-determinant of the model-complexity factor :math:`A_\\alpha(\\gamma, \\tau)`
when :math:`\\alpha = 1` (Lemma S2).

All functions are pure numpy and side-effect free.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import det, solve
from scipy.special import factorial


def VM(gamma_vec: np.ndarray, tau: float, Gram: np.ndarray) -> np.ndarray:
    """Posterior covariance matrix for the selected predictors.

    Implements Eq. (5):

        V_{\\gamma, \\tau} = X_\\gamma^T X_\\gamma + \\tau^{-1} I.

    Parameters
    ----------
    gamma_vec : np.ndarray
        Length-``N`` binary inclusion indicator.
    tau : float
        Prior precision parameter for the included coefficients.
    Gram : np.ndarray
        Pre-computed Gram matrix ``X^T X``.

    Returns
    -------
    np.ndarray
        Either ``V_{\\gamma, \\tau}`` of shape ``(|gamma|, |gamma|)`` or the
        scalar fallback ``[[1/tau]]`` when no predictors are selected.
    """

    idx = np.where(gamma_vec == 1)[0]
    if len(idx) == 0:
        return np.array([[1 / tau]])
    return Gram[np.ix_(idx, idx)] + np.eye(len(idx)) / tau


def RSS(
    gamma_vec: np.ndarray,
    tau: float,
    z: np.ndarray,
    X: np.ndarray,
    Gram: np.ndarray,
) -> float:
    """Quadratic form ``z^T \\Sigma_{\\gamma, \\tau} z``.

    Uses

        \\Sigma_{\\gamma, \\tau} = I - X_\\gamma V_{\\gamma, \\tau}^{-1} X_\\gamma^T,

    so that

        z^T \\Sigma_{\\gamma, \\tau} z = z^T z - (X_\\gamma^T z)^T V^{-1} (X_\\gamma^T z).
    """

    idx = np.where(gamma_vec == 1)[0]
    if len(idx) == 0:
        return float(z.T @ z)
    Xg = X[:, idx]
    Xz = Xg.T @ z
    V = VM(gamma_vec, tau, Gram)
    return float(z.T @ z - Xz.T @ solve(V, Xz))


def RSS2(
    gamma_vec: np.ndarray,
    tau: float,
    z1: np.ndarray,
    z2: np.ndarray,
    X: np.ndarray,
    Gram: np.ndarray,
) -> float:
    """Bilinear form ``z_1^T \\Sigma_{\\gamma, \\tau} z_2``.

    The symmetric counterpart of :func:`RSS`, used inside the pair Gibbs
    step to evaluate the cross term that determines :math:`\\beta_{i,j}`
    (Lemma S1 / Eq. S3 of the paper).
    """

    idx = np.where(gamma_vec == 1)[0]
    if len(idx) == 0:
        return float(z1.T @ z2)
    Xg = X[:, idx]
    Xz1 = Xg.T @ z1
    Xz2 = Xg.T @ z2
    V = VM(gamma_vec, tau, Gram)
    return float(z1.T @ z2 - Xz1.T @ solve(V, Xz2))


def AM(
    gamma_vec: np.ndarray,
    tau: float,
    theta: float,
    Gram: np.ndarray,
    N: int,
) -> float:
    """Log of the model-complexity factor for a given inclusion vector.

    Computes the part of :math:`\\log p(\\gamma, \\mu_\\gamma)` that depends
    on the structure of ``gamma`` for the :math:`\\alpha = 1` case
    (Lemma S2 + Eq. (2) of the paper):

        \\log A_1(\\gamma, \\tau)
            = |\\gamma| \\log \\theta + (N - |\\gamma|) \\log (1 - \\theta)
              + \\log (|\\gamma| - 1)!
              - \\tfrac{1}{2} \\log \\det V_{\\gamma, \\tau}.

    The ``-\\tfrac{|\\gamma|}{2} \\log \\tau`` term is handled separately
    inside :func:`loglike`.
    """

    sum_gamma = np.sum(gamma_vec)
    p = sum_gamma * np.log(theta) + (N - sum_gamma) * np.log(1 - theta)
    V = VM(gamma_vec, tau, Gram)
    det_log = np.log(det(V)) if det(V) > 0 else 0.0
    return p + np.log(factorial(max(sum_gamma - 1, 1))) - 0.5 * det_log


def loglike(
    gamma_vec: np.ndarray,
    tau: float,
    mu: np.ndarray,
    phi: float,
    Y: np.ndarray,
    X: np.ndarray,
    Gram: np.ndarray,
) -> float:
    """Log of the marginal likelihood ``p(y | \\gamma, \\mu_\\gamma, \\tau, \\phi)``.

    Implements Eq. (4) of the paper:

        \\log p(y \\mid \\gamma, \\mu_\\gamma, \\tau, \\phi)
            = \\tfrac{M}{2} \\log \\phi
              - \\tfrac{|\\gamma|}{2} \\log \\tau
              - \\tfrac{1}{2} \\log \\det V_{\\gamma, \\tau}
              - \\tfrac{\\phi}{2} (y - X_\\gamma \\mu_\\gamma)^T
                                  \\Sigma_{\\gamma, \\tau}
                                  (y - X_\\gamma \\mu_\\gamma).
    """

    z = Y - X @ mu
    V = VM(gamma_vec, tau, Gram)
    det_log = np.log(det(V)) if det(V) > 0 else 0.0
    return (
        0.5 * len(Y) * np.log(phi)
        - 0.5 * np.sum(gamma_vec) * np.log(tau)
        - 0.5 * det_log
        - 0.5 * phi * RSS(gamma_vec, tau, z, X, Gram)
    )
