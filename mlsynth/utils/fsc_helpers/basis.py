"""Cubic B-spline basis for the FSC ridge augmentation.

Section 3.2 expands each curve in an orthonormal basis of :math:`\\mathcal H`
and truncates at :math:`K` terms; the authors' code uses cubic B-splines with
:math:`K - 2` equispaced knots spanning the grid. B-splines are not orthonormal,
so this is a departure from the theory that the implementation makes and the
paper's numbers depend on -- the ridge quadratic form in eq. (9) is invariant
only under an orthogonal change of basis.

The knot construction follows ``cubicBsplines::Bsplines``: the interior knots
are extended by three equispaced knots on each side, which makes the resulting
basis the ordinary partition-of-unity cubic B-spline design matrix on that
extended sequence. Verified against the authors' Fortran to 3e-15 over the grids
of all three applications.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from ...exceptions import MlsynthEstimationError


def cubic_bspline_basis(grid: np.ndarray, n_basis: int) -> np.ndarray:
    """Return the ``(len(grid), n_basis)`` cubic B-spline design matrix.

    Parameters
    ----------
    grid : numpy.ndarray
        Argument values, ascending. Only its span matters for the knots.
    n_basis : int
        Number of basis functions ``K``; the interior knot sequence has
        ``K - 2`` points, so ``K >= 4``.

    Returns
    -------
    numpy.ndarray
        Non-negative, rows summing to one across the grid's span.

    Raises
    ------
    MlsynthEstimationError
        The grid is degenerate (all points equal), so no knot spacing exists.
    """
    grid = np.asarray(grid, dtype=float)
    lo, hi = float(grid.min()), float(grid.max())
    if not hi > lo:
        raise MlsynthEstimationError(
            "the argument grid is a single point, so no B-spline basis can be "
            "built. FSC needs at least two distinct argument values.")
    knots = np.linspace(lo, hi, int(n_basis) - 2)
    step = knots[1] - knots[0]
    full = np.concatenate([knots[0] - step * np.arange(3, 0, -1),
                           knots,
                           knots[-1] + step * np.arange(1, 4)])
    return BSpline.design_matrix(grid, full, 3, extrapolate=True).toarray()
