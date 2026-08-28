"""Analytical inference for Forward Difference-in-Differences (Li 2023).

Li (2023) derives a closed-form variance for the difference-in-differences
ATT estimator. In the paper's notation ``T1`` is the pre-period and ``T2``
the post-period. Writing the pre-treatment residuals of the treated unit
against its difference-in-differences fit as ``e_t``, and
``omega_2 = mean(e_t^2)`` for their mean square,

    omega_1 = (T2 / T1) * omega_2,
    Var(ATT) = (omega_1 + omega_2) / T2,

so the standard error is ``sqrt(omega_1 + omega_2) / sqrt(T2)``. The
``omega_1`` term prices in the error from estimating the level shift on
``T1`` pre-periods, which the post-period average inherits; it vanishes
relative to ``omega_2`` under Assumption 4(ii), where ``T2 / T1 -> 0``.

The standardised ATT is the ratio of the estimate to that standard error.
Proposition 2.1 gives it as ``sqrt(T2) * ATT / sqrt(omega_1 + omega_2)``,
which is the same quantity, and the author's own replication code computes
it that way (``FDID_Matlab.m``: ``ATT_std_FDID = sqrt(t2) * ATT_FDID /
std_Omega_hat_FDID``, annotated "it is N(0,1) under H0: ATT = 0"). It is
also what :func:`mlsynth.utils.effectutils.standardized_att` computes for
the library at large.

Note that this module's parameter names invert the paper's: ``pre_periods``
is Li's ``T1`` and ``post_periods`` her ``T2``.
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
        Standardised ATT, ``att / se`` -- standard normal under the null of
        no effect (Proposition 2.1).
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
    satt = att / se
    return float(se), ci, float(p_value), float(satt)
