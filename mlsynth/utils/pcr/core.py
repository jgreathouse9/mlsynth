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
    rcond: float | None = None,
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
    rcond : float, optional
        Relative cutoff for the retained singular values. Directions with
        ``s_i <= rcond * s_0`` carry no signal and are dropped before the
        inversion, so the effective rank is the smaller of ``rank`` and the
        design's numerical rank -- the same convention as
        :func:`numpy.linalg.pinv`. Defaults to ``max(design.shape) * eps``,
        which is inert on a well-conditioned design: when every retained
        direction clears the cutoff the returned weights are bit-for-bit the
        untruncated closed form.

    Returns
    -------
    np.ndarray
        Weight vector, shape ``(p,)``.

    Raises
    ------
    MlsynthEstimationError
        If no direction clears the cutoff, i.e. the design is numerically the
        zero matrix and no weight vector is identified.
    """
    target = np.ravel(target)
    _, U_r, s_r, Vt_r = hsvt(design, rank)

    # Without this, a design whose numerical rank falls below ``rank`` divides
    # by a singular value at or near zero: exactly zero gives 0/0 and NaN,
    # near zero gives weights of order 1e14 and no warning at all. Collinear
    # donors and a donor matrix collapsed by a long evaluation horizon both
    # reach it.
    if rcond is None:
        rcond = max(design.shape) * np.finfo(float).eps
    keep = s_r > rcond * s_r[0]
    if not np.any(keep):
        raise MlsynthEstimationError(
            "PCR design has numerical rank 0: every singular value is at or "
            "below the cutoff, so no weight vector is identified. Check that "
            "the donor matrix is not empty or identically zero."
        )
    U_r, s_r, Vt_r = U_r[:, keep], s_r[keep], Vt_r[keep, :]
    return Vt_r.T @ ((U_r.T @ target) / s_r)
