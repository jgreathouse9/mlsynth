"""The two numerical steps of local principal component analysis.

Feng (2024), *Optimal Estimation of Large-Dimensional Nonlinear Factor Models*,
Section 3, Algorithm 1: match each unit to its ``K`` nearest neighbours on one
block of periods, then take a truncated SVD of the neighbour submatrix on the
complementary block. Ported from the author's ``knn.index`` / ``findknn`` /
``lpca`` / ``sel.r``.

The matrix convention here is units by periods, the transpose of the paper's
``X``.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthConfigError

METRICS = ("pseudo_max", "euclidean", "average")


def pseudo_max_distance(A: np.ndarray, metric: str = "pseudo_max") -> np.ndarray:
    """Pairwise distances between units on the matching block.

    Three of the paper's Section 3 choices are available. The default is the
    pseudo-max distance of Zhang, Levina & Zhu (2017),

    .. math::

       \\rho(x_i, x_j) = \\max_{l \\neq i, j}
         \\bigl| \\tfrac{1}{p} (x_i - x_j)^\\top x_l \\bigr|,

    which averages over features and so separates units by their noise-free
    factor structure even when the errors are heteroskedastic; the Euclidean
    distance does not. ``"euclidean"`` is
    :math:`p^{-1/2} \\lVert x_i - x_j \\rVert` and ``"average"`` is
    :math:`p^{-1} | \\mathbf{1}^\\top (x_i - x_j) |`.

    Parameters
    ----------
    A : np.ndarray
        Matching block, shape ``(n_units, n_match_periods)``.
    metric : {"pseudo_max", "euclidean", "average"}

    Returns
    -------
    np.ndarray
        Symmetric ``(n_units, n_units)`` distances with a zero diagonal, so
        each unit is its own nearest neighbour -- paper equation (3.1) includes
        ``i`` in its own neighbourhood.

    Raises
    ------
    MlsynthConfigError
        If ``metric`` is not one of the three supported names.
    """
    if metric not in METRICS:
        raise MlsynthConfigError(
            f"Unknown distance {metric!r}; expected one of {METRICS}."
        )
    A = np.asarray(A, dtype=float)
    n, p = A.shape
    if metric == "euclidean":
        diff = A[:, None, :] - A[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2) / p)
    if metric == "average":
        totals = A.sum(axis=1) / p
        return np.abs(totals[:, None] - totals[None, :])

    gram = A @ A.T / p
    distance = np.empty((n, n), dtype=float)
    for i in range(n):
        # R: tmp <- abs(A2 - v[-1]) recycles column-wise, giving
        # tmp[l, j] = |A2[l, j] - A2[i, l]| = |<x_j - x_i, x_l>/p|.
        tmp = np.abs(gram - gram[i][:, None])
        np.fill_diagonal(tmp, -1.0)   # R: diag(tmp) <- -1   (drops l == j)
        tmp[i, :] = -1.0              # R: tmp[v[1], ] <- -1 (drops l == i)
        distance[i] = tmp.max(axis=0)
    return distance


def neighbour_index(distance: np.ndarray, n_neighbours: int) -> np.ndarray:
    """Boolean neighbourhood indicator, one column per unit.

    The reference rule is a threshold, ``tmp.d <= nth(tmp.d, K)``, so exact
    ties are all kept and the realised neighbourhood can exceed ``K``. Binary
    or otherwise discrete panels tie routinely, which is why the estimator
    reports the realised size next to the requested one.

    Parameters
    ----------
    distance : np.ndarray
        Square distance matrix from :func:`pseudo_max_distance`.
    n_neighbours : int
        Requested neighbourhood size ``K``, counting the unit itself.

    Returns
    -------
    np.ndarray
        Boolean ``(n_units, n_units)``; column ``i`` flags unit ``i``'s
        neighbours.
    """
    n = distance.shape[0]
    keep = np.zeros((n, n), dtype=bool)
    for i in range(n):
        kth = np.partition(distance[i], n_neighbours - 1)[n_neighbours - 1]
        keep[:, i] = distance[i] <= kth
    return keep


def select_rank(singular_values: np.ndarray, n_neighbours: int) -> int:
    """Number of local components to keep (R: ``sel.r``).

    Consecutive singular values whose ratio falls below ``log log K`` count as
    indistinguishable from noise, so the rank is the position of the first such
    pair, floored at one; if every ratio clears the threshold the rank is
    ``len(singular_values) - 1``.

    The threshold has a consequence the paper does not draw out. Ratios of a
    descending spectrum are at least 1, and ``log log K < 1`` for ``K <= 15``
    (it crosses 1 at ``e^e = 15.15``), so below sixteen neighbours the
    comparison never fires and the rank is pinned at
    ``len(singular_values) - 1`` however the data behaves. Feng's Kansas
    application sets ``K = 14``. Paper Remark 4.3 leaves formal rank selection
    to future work, so the estimator surfaces the realised rank as a
    diagnostic.
    """
    singular_values = np.asarray(singular_values, dtype=float)
    d = singular_values.size
    if d <= 1:
        return 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (singular_values[:-1] / singular_values[1:]) < np.log(
            np.log(n_neighbours)
        )
    if ratio.any():
        return max(int(np.argmax(ratio)) + 1 - 1, 1)
    return d - 1


def local_pca_row(
    block: np.ndarray,
    neighbours: np.ndarray,
    unit: int,
    n_neighbours: int,
    n_components: int = 3,
) -> tuple[np.ndarray, int]:
    """Local PCA reconstruction of one unit's row on the PCA block.

    Applies a truncated SVD to the neighbour submatrix and returns the row of
    :math:`\\widehat F_{\\langle i \\rangle} \\widehat\\Lambda_{\\langle i
    \\rangle}^\\top` belonging to ``unit`` -- the estimate of
    :math:`\\eta_l(\\alpha_i)` in paper equation (3.3).

    Only one row of the reconstruction is ever read and the right singular
    vectors cancel from it, so the left factor of the ``K x K`` Gram matrix
    suffices: the row is ``(U[pos, :r] U[:, :r]') B``. That agrees with the
    ``K x p`` SVD to about 5e-14 and is several times faster.

    Parameters
    ----------
    block : np.ndarray
        PCA block, shape ``(n_units, n_pca_periods)``.
    neighbours : np.ndarray
        Boolean neighbour indicator for ``unit`` (a column of
        :func:`neighbour_index`).
    unit : int
        Row index of the unit being reconstructed.
    n_neighbours : int
        Requested ``K``, passed to :func:`select_rank` for its ``log log K``.
    n_components : int
        Maximum number of components extracted (the paper's ``nlam``).

    Returns
    -------
    tuple
        The reconstructed row, the rank actually used, and the weight each
        neighbour receives.

    The weights are the ``unit``-th column of the rank-``r`` projector
    :math:`U_r U_r^\\top`, so the reconstruction is literally a weighted
    combination of the neighbourhood's rows. They neither sum to one nor stay
    non-negative -- this is a projection, not a simplex fit. The unit's own
    entry is the interesting one: the treated row is part of its own
    neighbourhood, so that weight measures how much the reconstruction leans on
    the cells the estimator zeroed, which is the perturbation Theorem 6.1
    bounds.
    """
    rows = np.flatnonzero(neighbours)
    neighbourhood = np.asarray(block, dtype=float)[rows]
    position = int(np.flatnonzero(rows == unit)[0])

    eigenvalues, vectors = np.linalg.eigh(neighbourhood @ neighbourhood.T)
    top = np.argsort(eigenvalues)[::-1][:n_components]
    singular_values = np.sqrt(np.maximum(eigenvalues[top], 0.0))
    left = vectors[:, top]
    rank = select_rank(singular_values, n_neighbours)
    rank = min(rank, left.shape[1])
    weights = left[:, :rank] @ left[position, :rank]
    return weights @ neighbourhood, rank, weights
