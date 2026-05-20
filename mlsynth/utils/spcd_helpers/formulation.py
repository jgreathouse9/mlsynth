"""Iteration-matrix construction for SPCD.

Implements Eq. (2) of the paper,

    M = Y Y^T + alpha I + lambda 1 1^T,

and supplies spectral-based auto-defaults for the three hyperparameters
``alpha``, ``lambda``, ``beta`` when the user has not specified them.

Reference
---------
Lu, Y., Li, J., Ying, L., & Blanchet, J. (2022).
"Synthetic Principal Component Design: Fast Covariate Balancing with
Synthetic Controls." arXiv:2211.15241v1.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthDataError


def validate_spcd_inputs(Y_pre: np.ndarray) -> None:
    """Check basic shape and feasibility of the pre-treatment matrix.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(T_pre, N)``.

    Raises
    ------
    MlsynthDataError
        If ``Y_pre`` is not 2D, has fewer than 2 periods, or fewer than
        2 units.
    """

    if Y_pre.ndim != 2:
        raise MlsynthDataError("Y_pre must be a two-dimensional T x N matrix.")
    if Y_pre.shape[0] < 2:
        raise MlsynthDataError("SPCD requires at least two pre-treatment periods.")
    if Y_pre.shape[1] < 2:
        raise MlsynthDataError("SPCD requires at least two units.")


def build_iteration_matrix(
    Y_pre: np.ndarray,
    alpha: Optional[float] = None,
    lam: Optional[float] = None,
    beta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Build the SPCD iteration matrix ``M`` and its inverse.

    Implements Eq. (2) of the paper:

        M = Y_pre.T @ Y_pre + alpha I + lambda 1 1^T

    Note that ``Y_pre`` here is ``(T_pre, N)`` (mlsynth convention),
    which is the transpose of the paper's ``Y in R^{N x T}``. The product
    ``Y_pre.T @ Y_pre`` is exactly the paper's ``Y Y^T``.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment matrix of shape ``(T_pre, N)``.
    alpha : float, optional
        Ridge term ``alpha`` in Eq. (2). If ``None``, defaults to a small
        positive multiple of the largest eigenvalue of ``Y_pre.T @ Y_pre``
        (just enough to keep ``M`` numerically positive-definite without
        dominating the spectrum).
    lam : float, optional
        Sum-zero penalty ``lambda`` in Eq. (2). If ``None``, defaults to
        the largest eigenvalue of ``Y_pre.T @ Y_pre`` so that Theorem 1's
        "large enough lambda" condition is satisfied on the scale of the
        data.
    beta : float, optional
        Iteration step parameter ``beta`` used in Eqs. (4), (5), (7),
        (8). If ``None``, defaults to ``1 / lambda_max(M)``, the smallest
        eigenvalue of ``M^{-1}``. This is the natural scale that keeps
        ``M^{-1} + beta I`` from being numerically dominated by either
        term at large ``N``.

    Returns
    -------
    M : np.ndarray
        The N x N iteration matrix from Eq. (2).
    M_inv : np.ndarray
        The inverse of ``M``, computed via ``np.linalg.solve(M, I)``.
    alpha : float
        Final value used (auto-estimated if input was ``None``).
    lam : float
        Final value used (auto-estimated if input was ``None``).
    beta : float
        Final value used (auto-estimated if input was ``None``).
    """

    validate_spcd_inputs(Y_pre)
    N = Y_pre.shape[1]

    YtY = Y_pre.T @ Y_pre

    eigvals_YtY = np.linalg.eigvalsh(YtY)
    lam_max_YtY = float(eigvals_YtY[-1])

    if alpha is None:
        alpha = max(1e-6 * lam_max_YtY, 1e-12)
    if lam is None:
        lam = lam_max_YtY if lam_max_YtY > 0 else 1.0

    ones = np.ones((N, N), dtype=float)
    M = YtY + alpha * np.eye(N) + lam * ones

    M_inv = np.linalg.solve(M, np.eye(N))

    if beta is None:
        lam_max_M = float(np.linalg.eigvalsh(M)[-1])
        beta = 1.0 / lam_max_M if lam_max_M > 0 else 1.0

    return M, M_inv, float(alpha), float(lam), float(beta)
