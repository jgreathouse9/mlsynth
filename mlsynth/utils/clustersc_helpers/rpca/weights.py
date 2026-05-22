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

import numpy as np
from scipy.optimize import nnls

from ....exceptions import MlsynthEstimationError


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
