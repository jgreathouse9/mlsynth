"""Hard singular-value truncation and principal-component-regression weights —
the linear-algebra kernel shared by ClusterSC, SI, and SNN.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError


def hsvt(
    X: np.ndarray,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply hard rank-``rank`` truncation to ``X``.

    Computes :math:`\\widetilde{M} = \\sum_{i=1}^{r} \\sigma_i u_i v_i^\\top`
    (Algorithm 2 step 2 of Rho et al. 2025).

    Returns
    -------
    M_hat : np.ndarray
        Rank-``rank`` reconstruction, shape ``(m, n)``.
    U_r : np.ndarray
        Truncated left singular vectors, shape ``(m, rank)``.
    s_r : np.ndarray
        Truncated singular values, shape ``(rank,)``.
    Vt_r : np.ndarray
        Truncated right singular vectors (transposed), shape ``(rank, n)``.
    """
    if X.ndim != 2:
        raise MlsynthEstimationError("HSVT input must be a 2D matrix.")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = max(1, min(int(rank), s.size))
    U_r = U[:, :r]
    s_r = s[:r]
    Vt_r = Vt[:r, :]
    M_hat = (U_r * s_r) @ Vt_r
    return M_hat, U_r, s_r, Vt_r


def pcr_weights(
    design: np.ndarray,
    target: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Principal-component-regression weights over the columns of ``design``.

    Regresses ``target`` onto the top-``rank`` principal subspace of ``design``
    and returns the closed-form weight vector

    .. math:: \\widehat w = V_r \\, \\mathrm{diag}(1/s_r) \\, U_r^\\top \\, y,

    i.e. :func:`hsvt`-truncate ``design`` and apply the pseudo-inverse. This is
    SI-PCR eq. 10 with ``design = Y_{pre, donors}``; SNN's per-entry PCR is the
    same kernel with ``design`` the *transposed* anchor block (weights over
    anchor rows).

    Parameters
    ----------
    design : np.ndarray
        Design matrix, shape ``(m, p)``; weights are over its ``p`` columns.
    target : np.ndarray
        Response vector, shape ``(m,)``.
    rank : int
        Spectral truncation rank.

    Returns
    -------
    np.ndarray
        Weight vector, shape ``(p,)``.
    """
    target = np.ravel(target)
    _, U_r, s_r, Vt_r = hsvt(design, rank)
    return Vt_r.T @ ((U_r.T @ target) / s_r)
