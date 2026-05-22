"""Half-quadratic regularised Robust PCA (HQF).

Implements the non-convex robust PCA variant of Wang, Li, So & Liu
(2023), *Robust PCA via non-convex half-quadratic regularization*,
*Signal Processing* 204, 108816.

Solves a factored model :math:`Y = UV + S` where :math:`U \\in
\\mathbb{R}^{m \\times r}`, :math:`V \\in \\mathbb{R}^{r \\times n}`, and
:math:`S` collects sparse / outlier entries. Updates alternate between
regularised least squares for the factors and an adaptive median-based
threshold on the residual for the sparse component.

Default hyperparameters (Bayani 2021 Section 2.4, recommending Wang et
al.'s defaults):

* ``rank`` -- smallest :math:`r` with cumulative spectral energy
  :math:`\\geq` ``cumvar_threshold`` (Bayani uses 0.999).
* ``ip`` (noise-scale adaptation factor) -- ``1.0``.
* ``lam`` (factor regularisation) -- :math:`1 / \\sqrt{\\max(m, n)}`.
* ``max_iter`` -- ``1000``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError

_MAD_CONST = 1.4815  # 1/Phi^{-1}(0.75); makes MAD consistent with Gaussian std.


@dataclass(frozen=True)
class HQFResult:
    """Output of :func:`hqf_decompose`: low-rank + sparse parts + diagnostics."""

    low_rank: np.ndarray   # (m, n) — Û V̂
    sparse: np.ndarray     # (m, n) — Ŝ
    iterations: int
    rank_used: int
    lambda_used: float
    ip_used: float


def _select_rank(Y: np.ndarray, cumvar_threshold: float) -> int:
    """Smallest r with cumulative spectral energy >= ``cumvar_threshold``."""
    sv = np.linalg.svd(Y, full_matrices=False, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return 1
    cum = np.cumsum(energy) / total
    r = int(np.searchsorted(cum, cumvar_threshold) + 1)
    return max(1, min(r, min(Y.shape)))


def _mad_scale(residual: np.ndarray, ip: float) -> float:
    """Median-absolute-deviation scale used to threshold the sparse term."""
    flat = residual.flatten()
    if flat.size == 0:
        return 1e-6
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    scale = ip * _MAD_CONST * mad
    return scale if scale > 0.0 else 1e-6


def _sparse_from_residual(residual: np.ndarray, scale: float) -> np.ndarray:
    """Hard-threshold the centred residual at ``scale``."""
    centred = residual - np.median(residual)
    mask = np.abs(centred) >= scale
    return residual * mask


def hqf_decompose(
    Y: np.ndarray,
    rank: Optional[int] = None,
    cumvar_threshold: float = 0.999,
    lam: Optional[float] = None,
    ip: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 0,
) -> HQFResult:
    """Run the HQF robust PCA decomposition.

    Parameters
    ----------
    Y : np.ndarray
        Observed matrix, shape ``(m, n)``.
    rank : int, optional
        Explicit factorisation rank. If ``None``, picked by cumulative
        spectral energy at ``cumvar_threshold``.
    cumvar_threshold : float
        Cumulative-variance target used when ``rank`` is None. Bayani
        2021 uses ``0.999``.
    lam : float, optional
        Tikhonov factor for the alternating LS step. Defaults to
        :math:`1 / \\sqrt{\\max(m, n)}` (Bayani 2021).
    ip : float
        Noise-scale adaptation factor. Default ``1.0``.
    max_iter : int
        Iteration cap. Default ``1000``.
    random_state : int
        Seed for the initialisation of :math:`U`.
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        raise MlsynthDataError("Y must be a 2D NumPy array.")
    if Y.size == 0:
        raise MlsynthDataError("Y cannot be empty.")
    if ip <= 0.0:
        raise MlsynthConfigError("ip must be positive.")
    if max_iter <= 0:
        raise MlsynthConfigError("max_iter must be positive.")

    m, n = Y.shape
    r = int(rank) if rank is not None else _select_rank(Y, cumvar_threshold)
    if r < 1 or r > min(m, n):
        raise MlsynthConfigError(
            f"rank {r} out of range [1, min(m, n)={min(m, n)}]."
        )
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if lam < 0.0:
        raise MlsynthConfigError("lam must be non-negative.")

    rng = np.random.default_rng(random_state)
    U = rng.random((m, r))
    V = np.zeros((r, n))

    try:
        for _ in range(3):  # Power-factorisation warm start.
            V = np.linalg.pinv(U) @ Y
            U = Y @ np.linalg.pinv(V)
    except np.linalg.LinAlgError as exc:
        raise MlsynthEstimationError(
            f"Pseudo-inverse failed during HQF initialisation: {exc}"
        ) from exc

    L = U @ V
    S = _sparse_from_residual(Y - L, _mad_scale(Y - L, ip))
    U_prev, V_prev = U.copy(), V.copy()

    prev_rmse = np.inf
    stagnant = 0
    iter_k = 0

    for iter_k in range(1, max_iter + 1):
        target = Y - S
        I_r = np.eye(r)
        try:
            A = V @ V.T + lam * I_r
            if np.linalg.matrix_rank(A) < r:
                U = (target @ V.T + lam * U_prev) @ np.linalg.pinv(A)
            else:
                U = (target @ V.T + lam * U_prev) @ np.linalg.inv(A)
            B = U.T @ U + lam * I_r
            if np.linalg.matrix_rank(B) < r:
                V = np.linalg.pinv(B) @ (U.T @ target + lam * V_prev)
            else:
                V = np.linalg.inv(B) @ (U.T @ target + lam * V_prev)
        except np.linalg.LinAlgError as exc:
            raise MlsynthEstimationError(
                f"HQF factor update failed at iter {iter_k}: {exc}"
            ) from exc
        U_prev, V_prev = U.copy(), V.copy()
        L = U @ V
        residual = Y - L
        S = _sparse_from_residual(residual, _mad_scale(residual, ip))

        rmse = float(np.linalg.norm(residual, "fro")) / float(np.sqrt(m * n))
        if prev_rmse - rmse < 1e-6:
            stagnant += 1
        else:
            stagnant = 0
        if stagnant > 1:
            break
        prev_rmse = rmse

    return HQFResult(
        low_rank=L, sparse=S, iterations=iter_k,
        rank_used=r, lambda_used=float(lam), ip_used=float(ip),
    )
