"""Bias-aware intervals, breakdown values and the placebo noise scale (Spruk 2026, Section 4.3).

Pure NumPy and SciPy. An estimate of a phase-average effect is bias plus
noise. The bias is bounded by ``B(M) = M kappa RMSPE`` (equation 13), where
``M`` is the share of the pre-treatment RMSPE treated as systematic and
aligned with the post-treatment factor direction; the noise scale comes from
in-space placebos. Proposition 5 gives the shortest interval that covers at
``1 - alpha`` whenever ``|bias| <= B`` (Armstrong and Kolesar 2018, 2020), and
the breakdown value ``M*`` is the largest ``M`` at which the interval still
excludes zero.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2, norm

#: The breakdown search stops here, as in the reference implementation.
M_MAX = 5.0

#: Scales the median absolute deviation to the standard deviation of a normal.
MAD_TO_SD = 1.4826


def cv_bias_aware(b: float, alpha: float) -> float:
    """The smallest ``c`` with ``P(|Z + b| <= c) >= 1 - alpha``, ``Z ~ N(0, 1)``.

    ``|Z + b|^2`` is a noncentral chi-square with one degree of freedom and
    noncentrality ``b^2``, so ``c`` is the square root of its ``1 - alpha``
    quantile. At ``b = 0`` this is the normal quantile; for large ``b`` it
    approaches ``b + z_{1-alpha}``, which is shorter than adding the bias bound
    to a conventional interval.
    """
    b = abs(float(b))
    if b < 1e-10:
        return float(norm.ppf(1 - alpha / 2))
    return float(np.sqrt(ncx2.ppf(1 - alpha, df=1, nc=b ** 2)))


def bias_aware_halfwidth(M: float, kappa: float, rmspe: float, sd: float, alpha: float) -> float:
    """Half-length ``sd * cv(M kappa RMSPE / sd)`` of the interval of Proposition 5."""
    return cv_bias_aware(M * kappa * rmspe / sd, alpha) * sd


def breakdown(tau: float, kappa: float, rmspe: float, sd: float, alpha: float) -> float:
    """``M* = sup{M >= 0 : 0 not in CI(B(M))}``, capped at :data:`M_MAX`.

    The half-length is continuous and increasing in ``M``, so ``M*`` is zero
    when the interval at ``M = 0`` already contains zero, and otherwise the
    root of ``|tau| = halfwidth(M)``.
    """
    def excess(M: float) -> float:
        return abs(tau) - bias_aware_halfwidth(M, kappa, rmspe, sd, alpha)

    if excess(0.0) <= 0:
        return 0.0
    if excess(M_MAX) > 0:
        return M_MAX
    return float(brentq(excess, 0.0, M_MAX, xtol=1e-14, rtol=1e-12))


def placebo_noise_scale(placebo_means: np.ndarray, placebo_rmspe: np.ndarray,
                        treated_rmspe: float) -> Tuple[float, float]:
    """The noise s.d. from in-space placebos, and its robust version.

    Each placebo's phase average is divided by its own pre-treatment RMSPE, the
    standard deviation of those ratios is rescaled by the treated unit's
    RMSPE (the post/pre logic of Abadie et al. 2010), and the robust version
    replaces the standard deviation with 1.4826 times the median absolute
    deviation. Poorly fitted placebos cannot dominate either.
    """
    z = np.asarray(placebo_means, dtype=float) / np.asarray(placebo_rmspe, dtype=float)
    sd = float(z.std(ddof=1) * treated_rmspe)
    mad = float(MAD_TO_SD * np.median(np.abs(z - np.median(z))) * treated_rmspe)
    return sd, mad


def placebo_p_value(tau: float, rmspe: float, placebo_means: np.ndarray, placebo_rmspe: np.ndarray) -> float:
    """Share of the ``J + 1`` units whose ``|phase average| / RMSPE`` is at least the treated unit's."""
    stat = abs(tau) / rmspe
    ratios = np.abs(np.asarray(placebo_means, dtype=float)) / np.asarray(placebo_rmspe, dtype=float)
    return float((1 + np.sum(ratios >= stat)) / (ratios.size + 1))
