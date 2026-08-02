"""Projection of the augmented estimate back onto the image of the isometry.

The plain FSC estimate is a convex combination of donor objects, so it lands in
:math:`\\Psi(\\mathcal M)` automatically -- that set is convex. The augmented
estimate need not, because eq. (9) allows negative weights, so section 3.2
projects it back:

.. math::

   \\tilde Y^{N,\\mathrm{aug}}_{1t}
     = \\arg\\min_{y \\in \\mathcal Y} \\lVert y - \\hat Y^{N,\\mathrm{aug}}_{1t}
       \\rVert_{\\mathcal H}.

Two spaces admit a closed form and are implemented here. For quantile functions
:math:`\\mathcal Y` is the set of increasing functions and the projection is the
increasing rearrangement (Remark 4). For symmetric positive semidefinite
matrices under the Frobenius metric it is the eigenvalue clip: symmetrize, set
negative eigenvalues to zero, reassemble. For the plain functional case
:math:`\\mathcal Y = \\mathcal H` and there is nothing to do.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...exceptions import MlsynthDataError


def project_increasing(grid: np.ndarray, values: np.ndarray,
                       bounds: Optional[Tuple[float, float]] = None
                       ) -> np.ndarray:
    """Increasing rearrangement of ``values``, optionally truncated first.

    On an ascending grid the L2 projection onto the cone of non-decreasing
    functions sampled at those points is the sorted vector, which is what this
    returns. Truncation to ``bounds`` is applied first and is itself a
    projection onto a convex set, so the composition stays inside
    :math:`\\mathcal Y`.
    """
    values = np.asarray(values, dtype=float)
    if bounds is not None:
        values = np.clip(values, bounds[0], bounds[1])
    order = np.argsort(np.asarray(grid, dtype=float), kind="stable")
    out = np.empty_like(values)
    out[order] = np.sort(values[order])
    return out


def triangular_dimension(n_cells: int) -> int:
    """Recover ``m`` from ``m (m + 1) / 2`` cells, or report the mismatch."""
    m = int(round((np.sqrt(8.0 * n_cells + 1.0) - 1.0) / 2.0))
    if m * (m + 1) // 2 != n_cells:
        raise MlsynthDataError(
            f"space='matrix' expects the row-major lower triangle of a "
            f"symmetric matrix, so the grid must have m(m+1)/2 points; got "
            f"{n_cells}, which is not a triangle number. The nearest valid "
            f"sizes are {m * (m + 1) // 2} and {(m + 1) * (m + 2) // 2}.")
    return m


def vech_to_matrix(vech: np.ndarray, dim: int) -> np.ndarray:
    """Rebuild the symmetric ``(dim, dim)`` matrix from its lower triangle."""
    rows, cols = np.tril_indices(dim)
    M = np.zeros((dim, dim), dtype=float)
    M[rows, cols] = np.asarray(vech, dtype=float)
    return M + M.T - np.diag(np.diag(M))


def matrix_to_vech(matrix: np.ndarray) -> np.ndarray:
    """Row-major lower triangle of a square matrix."""
    rows, cols = np.tril_indices(matrix.shape[0])
    return np.asarray(matrix, dtype=float)[rows, cols]


def project_psd(vech: np.ndarray, dim: int) -> np.ndarray:
    """Frobenius projection onto the positive semidefinite cone.

    Exact rather than iterative: for a symmetric matrix the nearest PSD matrix
    in Frobenius norm is obtained by clipping the eigenvalues at zero.
    """
    M = vech_to_matrix(vech, dim)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    if eigenvalues.min() >= 0.0:
        return np.asarray(vech, dtype=float)
    clipped = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return matrix_to_vech(clipped)


def frobenius_weights(dim: int) -> np.ndarray:
    """Coordinate weights making the half-vectorisation an isometry.

    The Frobenius metric counts each off-diagonal entry twice, so the map
    ``A -> vech(A)`` is an isometry only once the off-diagonal coordinates are
    scaled by ``sqrt(2)``. Applying these weights is what makes ``space='matrix'``
    fit in the metric Example 3 specifies rather than in the plain vech norm.
    """
    rows, cols = np.tril_indices(dim)
    return np.where(rows == cols, 1.0, np.sqrt(2.0))
