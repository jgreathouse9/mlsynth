"""Initialization and input-validation helpers for the relaxed SYNDES solver."""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError


def validate_relaxed_inputs(Y: np.ndarray, K: int) -> None:
    """Validate basic shape and feasibility of the relaxed-solver inputs.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    K : int
        Number of treated units to select.

    Raises
    ------
    MlsynthDataError
        If ``Y`` is not a two-dimensional array.
    MlsynthConfigError
        If ``Y`` has fewer than two periods, or if ``K`` is non-positive or
        exceeds the number of units.
    """

    if Y.ndim != 2:
        raise MlsynthDataError("Y must be a two-dimensional T x N matrix.")
    if Y.shape[0] < 2:
        raise MlsynthConfigError("At least two time periods are required.")
    if K <= 0:
        raise MlsynthConfigError("K must be a positive integer.")
    if K > Y.shape[1]:
        raise MlsynthConfigError("K cannot exceed the number of units.")


def default_lambda(Y: np.ndarray) -> float:
    """Default ridge parameter: average cross-sectional variance of ``Y``.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.

    Returns
    -------
    float
        Mean of the per-unit variances of ``Y``.

    Notes
    -----
    Uses ``ddof=0`` to preserve numerical compatibility with the original
    inline default of the relaxed solver.
    """

    return float(np.mean(np.var(Y, axis=0)))


def _reconstruction_error(Yc: np.ndarray, cols: list[int]) -> float:
    """Frobenius reconstruction error of ``Yc`` projected onto columns ``cols``.

    Returns ``inf`` for an empty column set (defensive: nothing to project on),
    otherwise the residual norm of ``Yc`` after projecting onto the span of the
    selected columns.
    """
    if len(cols) == 0:
        return np.inf
    B = Yc[:, cols]
    proj = B @ np.linalg.pinv(B) @ Yc
    return float(np.linalg.norm(Yc - proj, ord="fro"))


def init_assignment(Y: np.ndarray, K: int) -> np.ndarray:
    """Greedy span-based initialization of the treatment assignment.

    Selects ``K`` units whose columns greedily minimize the Frobenius
    reconstruction error of the column-demeaned outcome matrix. Returns
    a binary assignment vector compatible with the rest of the relaxed
    pipeline.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    K : int
        Number of treated units to select.

    Returns
    -------
    np.ndarray
        Binary assignment vector of shape ``(N,)`` with ``K`` ones.
    """

    _, N = Y.shape
    Yc = Y - Y.mean(axis=1, keepdims=True)

    selected: list[int] = []
    remaining = list(range(N))

    for _ in range(K):
        best_i = None
        best_score = np.inf
        for i in remaining:
            score = _reconstruction_error(Yc, selected + [i])
            if score < best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)

    D_init = np.zeros(N)
    D_init[selected] = 1
    return D_init
