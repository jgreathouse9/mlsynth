"""Principal Component Pursuit (PCP) via ADMM.

Implements Step 3 of Bayani (2021), with the convex relaxation due to
Candes, Li, Ma & Wright (2011):

.. math::

   \\min_{L,S}  \\| L \\|_* + \\lambda \\| S \\|_1
   \\quad \\text{s.t.} \\quad  Y = L + S.

The augmented Lagrangian is solved by alternating direction method of
multipliers (ADMM): :math:`L` is updated via the singular value
soft-thresholding operator
:math:`\\mathcal{D}_{1/\\mu}`, :math:`S` via element-wise soft-thresholding
:math:`\\mathcal{S}_{\\lambda/\\mu}`, and the dual variable
:math:`\\Lambda` is the standard multiplier update.

Defaults follow Bayani (2021) Section 2.4:

* :math:`\\lambda = 1 / \\sqrt{\\max(N, T)}` (Candes et al. 2011 default).
* :math:`\\mu = \\frac{N \\cdot T}{4 \\cdot \\sum_{ij} |Y_{ij}|}`
  (Bayani's rule of thumb).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ....exceptions import MlsynthDataError, MlsynthEstimationError


@dataclass(frozen=True)
class PCPResult:
    """Decomposition output: ``Y = L + S``, plus solver diagnostics."""

    low_rank: np.ndarray    # (m, n) — L̂
    sparse: np.ndarray      # (m, n) — Ŝ
    iterations: int
    converged: bool
    lambda_used: float
    mu_used: float


def _soft_threshold(X: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft-thresholding :math:`\\mathcal{S}_\\tau(X)`."""
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0.0)


def _svt(X: np.ndarray, threshold: float) -> np.ndarray:
    """Singular value soft-thresholding :math:`\\mathcal{D}_\\tau(X)`."""
    try:
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MlsynthEstimationError(f"SVD failed inside PCP SVT step: {exc}") from exc
    sv_thresholded = np.maximum(sv - threshold, 0.0)
    return (U * sv_thresholded) @ Vt


def pcp_decompose(
    Y: np.ndarray,
    lam: Optional[float] = None,
    mu: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> PCPResult:
    """Solve the PCP problem and return the ``(L, S)`` decomposition.

    Parameters
    ----------
    Y : np.ndarray
        Observed matrix, shape ``(m, n)``.
    lam : float, optional
        Sparsity penalty :math:`\\lambda`. Defaults to
        :math:`1/\\sqrt{\\max(m, n)}` (Candes et al. 2011).
    mu : float, optional
        Augmented-Lagrangian penalty :math:`\\mu`. Defaults to
        :math:`m n / (4 \\sum |Y_{ij}|)` (Bayani 2021).
    max_iter : int
        Iteration cap. Default ``1000``.
    tol : float
        Frobenius-norm tolerance for ``Y - L - S``, scaled by
        ``||Y||_F``.
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        raise MlsynthDataError("Y must be a 2D NumPy array.")

    m, n = Y.shape
    if m == 0 or n == 0:
        return PCPResult(
            low_rank=np.zeros_like(Y), sparse=np.zeros_like(Y),
            iterations=0, converged=True, lambda_used=0.0, mu_used=0.0,
        )

    Y_frob = float(np.linalg.norm(Y, "fro"))
    if Y_frob == 0.0:
        return PCPResult(
            low_rank=np.zeros_like(Y), sparse=np.zeros_like(Y),
            iterations=0, converged=True, lambda_used=0.0, mu_used=0.0,
        )

    abs_sum = float(np.sum(np.abs(Y)))
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = (m * n) / (4.0 * abs_sum) if abs_sum > 0 else 1.0

    L = np.zeros_like(Y)
    S = np.zeros_like(Y)
    Lambda = np.zeros_like(Y)
    threshold = tol * Y_frob

    converged = False
    k = 0
    while k < max_iter:
        L = _svt(Y - S + Lambda / mu, 1.0 / mu)
        S = _soft_threshold(Y - L + Lambda / mu, lam / mu)
        residual = Y - L - S
        Lambda = Lambda + mu * residual
        if np.linalg.norm(residual, "fro") <= threshold:
            converged = True
            k += 1
            break
        k += 1

    return PCPResult(
        low_rank=L, sparse=S, iterations=k,
        converged=converged, lambda_used=float(lam), mu_used=float(mu),
    )
