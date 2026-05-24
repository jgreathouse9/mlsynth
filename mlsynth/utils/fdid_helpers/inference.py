"""Analytical inference for Forward Difference-in-Differences (Li 2023).

Li (2023) derives a closed-form variance for the difference-in-differences
ATT estimator. Writing the pre-treatment residuals of the treated unit
against its difference-in-differences fit as ``e_t``, the post-period
average treatment effect has asymptotic variance

    Var(ATT) = (omega_1 + omega_2) / T1,

where ``omega_2 = mean(e_t^2)`` is the pre-period residual variance and
``omega_1 = (T1 / T0) * omega_2`` inflates it for the post-period sample
size ``T1``. The standard error is the square root of this quantity.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def did_inference(
    att: float, pre_residuals: np.ndarray, pre_periods: int, post_periods: int
) -> Tuple[float, Tuple[float, float], float, float]:
    """Compute the Li (2023) analytical SE, 95% CI, p-value, and SATT.

    Parameters
    ----------
    att : float
        Estimated average treatment effect on the treated.
    pre_residuals : np.ndarray
        Pre-treatment residuals of the treated unit against its
        difference-in-differences fit, shape ``(T0,)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    post_periods : int
        Number of post-treatment periods ``T1``.

    Returns
    -------
    se : float
        Analytical standard error of the ATT (``nan`` if undefined).
    ci : tuple of float
        ``(lower, upper)`` 95% confidence interval.
    p_value : float
        Two-sided p-value for the ATT.
    satt : float
        Standardised ATT (``att / se * sqrt(T1)``).
    """
    if pre_periods <= 0 or post_periods <= 0:
        return np.nan, (np.nan, np.nan), np.nan, np.nan

    omega2 = float(np.mean(pre_residuals ** 2))
    omega1 = (post_periods / pre_periods) * omega2
    se = np.sqrt(omega1 + omega2) / np.sqrt(post_periods)

    if not (se > 0):
        return float(se), (np.nan, np.nan), np.nan, np.nan

    z = norm.ppf(0.975)
    ci = (att - z * se, att + z * se)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(att / se)))
    satt = att / se * np.sqrt(post_periods)
    return float(se), ci, float(p_value), float(satt)
