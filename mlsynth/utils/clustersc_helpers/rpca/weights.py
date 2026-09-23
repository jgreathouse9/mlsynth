"""Non-negative least squares weight solver for RPCA-SC.

Implements Step 4 of Bayani (2021):

.. math::

   \\widehat{\\beta} = \\arg\\min_{\\beta \\geq 0}
                      \\| y_i^- - (L^-)^\\top \\beta \\|_2^2,

where :math:`L^-` is the HSVT-denoised pre-period donor matrix.
The paper deliberately drops the sum-to-one constraint that classical
synthetic control imposes -- the clustering step already restricts the
donor pool to units that *behave* like the treated unit, so the
non-negativity constraint is enough to keep the counterfactual
interpretable.
"""

from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import numpy as np

from ....exceptions import MlsynthEstimationError
from ...bilevel.nnls import nnls_select

# scipy's compiled nnls where it is a fixed, fast release (>= 1.15), else the
# in-house solver (scipy 1.12-1.14 regressed -- raises on the iteration cap).
nnls = nnls_select()


def solve_nnls(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
) -> np.ndarray:
    """Return the non-negative least squares weights :math:`\\widehat{\\beta}`.

    Parameters
    ----------
    denoised_donor_pre : np.ndarray
        Pre-period denoised donor matrix, shape ``(T0, J)``.
        Columns are donor units.
    target_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.

    Notes
    -----
    Wraps :func:`scipy.optimize.nnls`. The Lawson-Hanson algorithm
    is finite-step exact for the box-constrained QP arising from
    Bayani's Step 4.
    """
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError(
            "denoised_donor_pre must be 2D (T0, J)."
        )
    if denoised_donor_pre.shape[0] != target_pre.shape[0]:
        raise MlsynthEstimationError(
            f"Pre-period length mismatch: donors {denoised_donor_pre.shape[0]} "
            f"vs target {target_pre.shape[0]}."
        )
    try:
        beta, _ = nnls(denoised_donor_pre, target_pre)
    except RuntimeError as exc:  # pragma: no cover - nnls only raises on bad input
        raise MlsynthEstimationError(
            f"NNLS failed inside RPCA-SC weight step: {exc}"
        ) from exc
    return np.asarray(beta, dtype=float)


def solve_msca(
    denoised_donor_pre: np.ndarray,
    target_pre: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return MSC(a) weights and intercept against the denoised donors.

    Solves

    .. math::

       \\min_{a,\\; w \\geq 0,\\; \\mathbf{1}^\\top w = 1}
           \\| y^- - L^- w - a \\mathbf{1} \\|_2,

    the MSC(a) program of Li and Shankar (2023), Section 3.1. The donor
    weights keep the Abadie-Diamond-Hainmueller simplex restriction; the
    intercept is free in sign, which is what lets the treated unit sit
    outside the donors' hull in levels.

    The intercept is an identification change, not a better fit. Under
    ``simplex`` the treated unit must lie inside the donors' convex hull;
    under MSC(a) it must lie parallel to a point in that hull, with a
    constant level gap that holds through the post period. When the gap
    drifts, the intercept fitted on the pre-period biases every
    post-period point, and the pre-period fit will not show it.

    Parameters
    ----------
    denoised_donor_pre : np.ndarray
        Pre-period denoised donor matrix, shape ``(T0, J)``.
        Columns are donor units.
    target_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.

    Returns
    -------
    tuple of (np.ndarray, float)
        Donor weights of shape ``(J,)`` and the fitted intercept.

    Notes
    -----
    Non-negativity applies to the donor weights alone. Constraining the
    intercept to be non-negative would clamp it to zero whenever the
    treated unit sits below its donors, silently recovering the simplex
    solution on exactly the panels the intercept exists to serve.
    """
    if denoised_donor_pre.ndim != 2:
        raise MlsynthEstimationError("denoised_donor_pre must be 2D (T0, J).")
    if denoised_donor_pre.shape[0] != target_pre.shape[0]:
        raise MlsynthEstimationError(
            f"Pre-period length mismatch: donors {denoised_donor_pre.shape[0]} "
            f"vs target {target_pre.shape[0]}."
        )

    n_donors = denoised_donor_pre.shape[1]
    w = cp.Variable(n_donors)
    a = cp.Variable()
    problem = cp.Problem(
        cp.Minimize(cp.norm(target_pre - denoised_donor_pre @ w - a, 2)),
        [w >= 0, cp.sum(w) == 1],
    )
    problem.solve(solver=cp.CLARABEL)

    if w.value is None or a.value is None:
        raise MlsynthEstimationError(
            f"MSC(a) weight solve did not converge (status: {problem.status})."
        )
    return np.asarray(w.value, dtype=float), float(a.value)
