"""Readable NumPy oracle for Feng (2023) local principal component analysis.

Ported line-by-line from the author's replication code,
``yingjieum/Replication_NonlinearFactorModel_2023/Application/
Feng_LPCA_app_suppfuns.R`` (``knn.index`` / ``findknn`` / ``lpca`` / ``sel.r``),
which implements Algorithm 1 of arXiv:2311.07243 Section 3.

The matrix convention here follows the R script: ``A`` is units x periods
(n x p), the transpose of the paper's ``X`` (p x n).
"""

from __future__ import annotations

import numpy as np


def pseudo_max_distance(A: np.ndarray) -> np.ndarray:
    """Pairwise pseudo-max distance of Zhang, Levina & Zhu (2017).

    ``rho(x_i, x_j) = max_{l != i,j} |(1/p) (x_i - x_j)' x_l|``; paper
    Section 3, distance (2). R reference: ``findknn`` / ``knn.index``.

    Parameters
    ----------
    A : (n, p) ndarray
        Matching block: units by the periods reserved for K-NN (``R_dagger``).

    Returns
    -------
    (n, n) ndarray
        ``D[i, j]`` is the distance between units ``i`` and ``j``; ``D[i, i]``
        is 0, so unit ``i`` is its own nearest neighbour (paper eq. 3.1
        includes ``i`` in ``N_i``).
    """
    n, p = A.shape
    G = A @ A.T / p                       # R: A2 <- tcrossprod(A)/p
    D = np.empty((n, n), dtype=float)
    for i in range(n):
        # R: tmp <- abs(A2 - v[-1]) recycles column-wise, so
        # tmp[l, j] = |A2[l, j] - A2[i, l]| = |<x_j - x_i, x_l>/p|.
        tmp = np.abs(G - G[i][:, None])
        np.fill_diagonal(tmp, -1.0)       # R: diag(tmp) <- -1   (drops l == j)
        tmp[i, :] = -1.0                  # R: tmp[v[1], ] <- -1 (drops l == i)
        D[i] = tmp.max(axis=0)            # R: colMaxs(tmp)
    return D


def knn_index(A: np.ndarray, K: int) -> np.ndarray:
    """Indicator matrix of each unit's K nearest neighbours.

    R reference: ``knn.index``. Column ``i`` of the return flags the K units
    closest to ``i`` (itself included), matching ``tmp.d <= nth(tmp.d, K)``.
    """
    D = pseudo_max_distance(A)
    n = D.shape[0]
    keep = np.zeros((n, n), dtype=bool)
    for i in range(n):
        kth = np.partition(D[i], K - 1)[K - 1]   # R: nth(tmp.d, K), K-th smallest
        keep[:, i] = D[i] <= kth
    return keep


def select_rank(svals: np.ndarray, K: int) -> int:
    """Number of local principal components to keep.

    R reference: ``sel.r``. Consecutive singular values whose ratio falls
    below ``log log K`` are treated as indistinguishable from noise, so the
    rank is the position of the first such pair, floored at 1; if every ratio
    clears the threshold, keep ``len(svals) - 1``.
    """
    d = svals.size
    ratio = (svals[:-1] / svals[1:]) < np.log(np.log(K))
    if ratio.any():
        return max(int(np.argmax(ratio) + 1) - 1, 1)   # R: max(which.max(ratio)-1, 1)
    return d - 1


def lpca_row(
    A: np.ndarray, neighbours: np.ndarray, unit: int, K: int, n_components: int = 3
) -> tuple[np.ndarray, int]:
    """Local PCA reconstruction of one unit's row.

    R reference: ``lpca``. Takes the ``K``-neighbour submatrix of the PCA
    block ``R_ddagger``, truncates its SVD at the rank ``select_rank`` picks,
    and returns the reconstructed row for ``unit`` -- the estimate of
    ``eta_l(alpha_i)`` in paper eq. (3.3).

    Parameters
    ----------
    A : (n, p_ddagger) ndarray
        PCA block: units by the periods held out from matching.
    neighbours : (n,) bool ndarray
        Neighbour indicator for ``unit`` (a column of :func:`knn_index`).
    unit : int
        Row index of the unit whose row is reconstructed.
    K : int
        Neighbourhood size, passed to :func:`select_rank` for ``log log K``.
    n_components : int
        Maximum number of components extracted (R: ``nlam``).

    Returns
    -------
    (p_ddagger,) ndarray, int
        The reconstructed row and the rank actually used.
    """
    rows = np.flatnonzero(neighbours)
    block = A[rows]                                  # R: A[index, ][, -1]
    pos = int(np.flatnonzero(rows == unit)[0])       # R: which(A[,1] == i)

    U, s, Vt = np.linalg.svd(block, full_matrices=False)
    U, s, Vt = U[:, :n_components], s[:n_components], Vt[:n_components]
    r = select_rank(s, K)
    return (U[pos, :r] * s[:r]) @ Vt[:r], r
