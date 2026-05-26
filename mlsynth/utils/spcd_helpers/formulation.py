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

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthDataError


@lru_cache(maxsize=256)
def _marchenko_pastur_median(beta: float) -> float:
    """Median of the Marchenko-Pastur law with aspect ratio ``beta``.

    For an ``m x n`` noise matrix with i.i.d. unit-variance entries, the
    squared singular values follow a Marchenko-Pastur law with ratio
    ``beta = min(m, n) / max(m, n)`` on the support
    ``[(1 - sqrt(beta))^2, (1 + sqrt(beta))^2]``. The median has no closed
    form, so it is computed once (and cached) by integrating the density on
    a fine grid. This is the constant used by the Gavish-Donoho
    median-singular-value noise estimator.

    Parameters
    ----------
    beta : float
        Aspect ratio in ``(0, 1]``.

    Returns
    -------
    float
        Median of the Marchenko-Pastur distribution.
    """

    beta = min(max(float(beta), 1e-6), 1.0)
    lo = (1.0 - np.sqrt(beta)) ** 2
    hi = (1.0 + np.sqrt(beta)) ** 2
    xs = np.linspace(lo, hi, 8192)
    density = np.sqrt(np.clip((hi - xs) * (xs - lo), 0.0, None)) / (
        2.0 * np.pi * beta * xs
    )
    cdf = np.cumsum(density)
    cdf *= (xs[1] - xs[0])
    cdf /= cdf[-1]
    return float(xs[int(np.searchsorted(cdf, 0.5))])


def estimate_noise_variance(Y_pre: np.ndarray) -> float:
    """Estimate the idiosyncratic noise variance ``sigma^2`` of ``Y_pre``.

    In the paper's outcome model the ridge term ``alpha`` in Eq. (2) plays
    the role of the noise variance ``sigma = Var(e_it)`` (it is the
    coefficient of the variance term ``sigma * sum_i w_i^2`` in the MSE
    decomposition), and the appendix convergence theorems set the
    perturbation ridge on the *noise* scale ``alpha = ||Delta||`` with
    ``alpha <= ||Delta||``. So ``alpha`` must track the **noise** scale of
    the data, not its dominant (signal/level) eigenvalue.

    This uses the Gavish-Donoho (2014) median-singular-value estimator,
    which is parameter-free and robust when the signal is approximately
    low-rank: with ``Y_pre`` of shape ``(m, n)`` and
    ``beta = min(m, n) / max(m, n)``,

    .. math::

        \\hat\\sigma = \\frac{\\operatorname{median}(\\text{singular values})}
            {\\sqrt{\\max(m, n)\\, \\mu_\\beta}},

    where ``mu_beta`` is the Marchenko-Pastur median. Returns
    ``hat_sigma^2``.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(T_pre, N)``.

    Returns
    -------
    float
        Estimated noise variance ``sigma^2`` (always positive).
    """

    m, n = Y_pre.shape
    beta = min(m, n) / max(m, n)
    singular_values = np.linalg.svd(Y_pre, compute_uv=False)
    median_sv = float(np.median(singular_values))
    mu_beta = _marchenko_pastur_median(round(beta, 4))
    sigma = median_sv / np.sqrt(max(m, n) * mu_beta)
    return float(sigma ** 2)


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
        Ridge term ``alpha`` in Eq. (2), which plays the role of the
        idiosyncratic noise variance ``sigma`` (Assumption 1). If ``None``,
        it is estimated from the **noise** scale of ``Y_pre`` via the
        Gavish-Donoho median-singular-value estimator
        (:func:`estimate_noise_variance`), consistent with the paper's
        appendix condition ``alpha = ||Delta||`` on the perturbation scale.
        Note the paper itself treats ``alpha`` as a *pre-defined*
        hyperparameter and gives no formula; this default is a heuristic.
        For data generated with a known noise level (e.g. the paper's
        Section 4.1 simulations with ``sigma = 1``), pass that value
        explicitly.
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
        # alpha plays the role of the noise variance sigma (Eq. (2) /
        # Assumption 1); estimate it from the noise scale of the data
        # rather than the dominant (signal) eigenvalue.
        alpha = max(estimate_noise_variance(Y_pre), 1e-8)
    if lam is None:
        lam = lam_max_YtY if lam_max_YtY > 0 else 1.0

    ones = np.ones((N, N), dtype=float)
    M = YtY + alpha * np.eye(N) + lam * ones

    M_inv = np.linalg.solve(M, np.eye(N))

    if beta is None:
        lam_max_M = float(np.linalg.eigvalsh(M)[-1])
        beta = 1.0 / lam_max_M if lam_max_M > 0 else 1.0

    return M, M_inv, float(alpha), float(lam), float(beta)
