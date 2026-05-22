"""Treated-unit projection and counterfactual construction for FMA.

Given the factor estimates ``F_hat`` from the control panel, the
treated unit's factor loading is recovered by OLS of its pre-period
outcomes on the factors (with a constant):

.. math::

   \\hat \\lambda_1 = (\\sum_{t=1}^{T_0} \\tilde F_t \\tilde F_t')^{-1}
                       \\sum_{t=1}^{T_0} \\tilde F_t \\, y_{1, t},

with :math:`\\tilde F_t = [1, F_{ht}']'`. The full-period counterfactual
is :math:`\\hat y^0_{1, t} = \\tilde F_t' \\hat\\lambda_1`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_loading_and_counterfactual(
    treated_outcome: np.ndarray,
    factors: np.ndarray,
    T0: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """OLS projection of the treated pre-period onto the factors.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated outcome series, shape ``(T,)``.
    factors : np.ndarray
        Estimated factor matrix, shape ``(T, r)``.
    T0 : int
        Number of pre-treatment periods.

    Returns
    -------
    lambda_hat : np.ndarray
        Loading vector ``(r + 1,)`` (constant + factor loadings).
    counterfactual : np.ndarray
        Predicted untreated outcome for every period, shape ``(T,)``.
    factors_with_const : np.ndarray
        Factor matrix with the constant column prepended, shape
        ``(T, r + 1)``. Returned for downstream variance plug-ins.
    residual_variance : float
        Variance of the pre-treatment residuals.
    """

    T = treated_outcome.shape[0]
    if factors.shape[0] != T:
        raise ValueError(
            f"factors has shape {factors.shape}; expected first dim = "
            f"{T} (T)."
        )

    ones = np.ones((T, 1))
    F_aug = np.concatenate([ones, factors], axis=1)         # (T, r+1)
    F_pre = F_aug[:T0]
    y_pre = treated_outcome[:T0]

    XtX = F_pre.T @ F_pre
    # Light ridge for the cases where (r+1) approaches T0.
    diag_mean = float(np.trace(XtX) / max(F_pre.shape[1], 1))
    XtX_reg = XtX + max(diag_mean, 1.0) * 1e-10 * np.eye(F_pre.shape[1])
    lambda_hat = np.linalg.solve(XtX_reg, F_pre.T @ y_pre)

    counterfactual = F_aug @ lambda_hat
    residuals_pre = y_pre - counterfactual[:T0]
    if T0 > F_aug.shape[1]:
        denom = T0 - F_aug.shape[1]
    else:
        denom = max(1, T0 - 1)
    residual_variance = float(residuals_pre @ residuals_pre) / denom

    return lambda_hat, counterfactual, F_aug, residual_variance
