"""Covariance functions for GPITS, and the length-scale rule.

The forms follow Cho (2026), Section 3.2, and match the reference
implementation ``gpss`` (``src/kernels.cpp``) exactly. Three conventions of
that reference are preserved deliberately, because a reading of the paper
alone would get them wrong and the estimator would then disagree with the
published numbers:

* the periodic and linear components run over every column of the design,
  the one-hot indicator columns included, not the time column alone;
* the periodic component's length-scale is ``b / 2``, not ``b``;
* the linear component is the homogeneous inner product ``x . x'`` with no
  intercept or variance parameter, so the diagonal of the combined kernel is
  ``1 + 1 + x . x``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "KERNELS",
    "PERIODIC_KERNELS",
    "getb_maxvar",
    "kernel_gaussian",
    "kernel_gaussian_periodic_linear",
]

# R's optimize() default tolerance. The reference selects the length-scale
# through that call, so the tolerance is part of its definition of ``b``;
# matching it keeps cross-validation against gpss exact.
R_OPTIMIZE_TOL = float(np.finfo(float).eps ** 0.25)
MAX_SEARCH_B = 2000.0


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape ``(len(A), len(B))``."""
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def kernel_gaussian(X1: np.ndarray, X2: np.ndarray, b: float,
                    period: Optional[float] = None) -> np.ndarray:
    """Squared-exponential kernel ``exp(-||x - x'||^2 / b)`` (Eq. 14)."""
    return np.exp(-_sqdist(X1, X2) / b)


def kernel_gaussian_periodic_linear(X1: np.ndarray, X2: np.ndarray, b: float,
                                    period: float) -> np.ndarray:
    """Gaussian + periodic + linear (Eq. 15).

    The Gaussian term carries local temporal dependence, the periodic term the
    seasonal cycle at ``period``, and the linear term a trend that keeps the
    predictive variance growing with distance from the training support.
    """
    gauss = np.exp(-_sqdist(X1, X2) / b)
    diff = X1[:, None, :] - X2[None, :, :]
    per = np.exp(-(2.0 * np.sin(np.pi * np.abs(diff) / period) ** 2).sum(-1)
                 / (b / 2.0))
    return gauss + per + X1 @ X2.T


KERNELS = {
    "gaussian": kernel_gaussian,
    "gaussian_periodic_linear": kernel_gaussian_periodic_linear,
}
PERIODIC_KERNELS = frozenset({"gaussian_periodic_linear"})


def getb_maxvar(X: np.ndarray, kernel: str, period: Optional[float],
                max_search_b: float = MAX_SEARCH_B,
                tol: float = R_OPTIMIZE_TOL) -> float:
    """Length-scale maximising the variance of the off-diagonal kernel entries.

    The rule of Hartman et al. (2025), used by Cho (2026, Section 3.2) in place
    of marginal likelihood or cross-validation for ``b``: it reads only the
    covariate structure, so the design stage stays separate from the outcome.
    Values that drive every covariance toward zero or one carry no information
    about which periods resemble which, and this locates the scale where the
    pairwise covariances are most spread out.
    """
    kern = KERNELS[kernel]
    tril = np.tril_indices(X.shape[0], -1)

    def neg_var(b: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            K = kern(X, X, b, period)
        v = K[tril]
        if not np.all(np.isfinite(v)):
            return 0.0
        return -float(np.var(v, ddof=1))

    res = minimize_scalar(neg_var, bounds=(0.01, max_search_b),
                          method="bounded", options={"xatol": tol})
    return float(res.x)
