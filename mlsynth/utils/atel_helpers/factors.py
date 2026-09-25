"""Diversified projection: factors as cross-sectional weighted averages.

Fan and Liao (2022) estimate the factors without an eigendecomposition. Given
weights :math:`W_{it}^{(j)}` that correlate with the loadings, the j-th factor
at time t is the donor-average

.. math:: \\widehat F_{tj} = \\frac{1}{N} \\sum_{i=1}^{N} Y_{it} W_{it}^{(j)},

which is linear in the outcomes and invariant to how the donors are ordered.
The estimate of the loading space comes from the covariates, so nothing here
depends on the donor panel's own spectrum.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError

__all__ = ["diversified_factors"]


def diversified_factors(
    donor_outcomes: np.ndarray, weights: np.ndarray, n_factors: int
) -> np.ndarray:
    """Factors ``(T, n_factors)`` from the donor panel and the weight blocks.

    The j-th factor uses weight columns ``j*T : (j+1)*T`` of ``weights``, whose
    first row belongs to the treated unit and is dropped. See
    :mod:`.sieve` for why the first ``n_factors`` blocks are the ones consumed.

    Parameters
    ----------
    donor_outcomes : np.ndarray
        Donor outcomes ``(N, T)``.
    weights : np.ndarray
        Diversified weights for every unit ``(N + 1, T * J * P)``.
    n_factors : int
        Number of factors, the ``J`` of the sieve.

    Returns
    -------
    np.ndarray
        Shape ``(T, n_factors)``.

    Raises
    ------
    MlsynthEstimationError
        If a consumed weight block is identically zero, which would return a
        factor that is identically zero and a singular local-linear design.
    """
    Y = np.asarray(donor_outcomes, dtype=float)
    W = np.asarray(weights, dtype=float)
    if Y.ndim != 2:
        raise MlsynthEstimationError("Donor outcomes must be 2-D (N, T).")
    n_donors, T = Y.shape
    if W.shape[0] != n_donors + 1:
        raise MlsynthEstimationError(
            f"The weight matrix has {W.shape[0]} rows for {n_donors} donors; "
            "it must carry the treated unit in its first row."
        )
    if W.shape[1] < n_factors * T:
        raise MlsynthEstimationError(
            f"The projection needs {n_factors} blocks of {T} columns, but the "
            f"weight matrix has only {W.shape[1]}."
        )

    F = np.empty((T, n_factors))
    donor_weights = W[1:]
    for j in range(n_factors):
        block = donor_weights[:, j * T : (j + 1) * T]
        if not np.any(block):
            raise MlsynthEstimationError(
                f"Weight block {j + 1} of {n_factors} is identically zero, so "
                "factor {0} would be zero everywhere. This happens when the "
                "sieve basis leaves trailing columns unfilled and the factor "
                "count reaches them; raise or lower the factor count, or use "
                "the bspline basis.".format(j + 1)
            )
        F[:, j] = (Y * block).mean(axis=0)
    return F
