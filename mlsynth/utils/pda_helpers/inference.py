"""Shared HAC long-run-variance machinery for PDA inference.

All three PDA variants base inference on a heteroskedasticity- and
autocorrelation-consistent (HAC) long-run variance of a scalar time series of
post-period treatment effects (and, for some variants, pre-period prediction
residuals). The Bartlett/Newey-West and uniform kernels are provided; the
default truncation lag follows Newey-West.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) automatic truncation lag ``floor(4 (n/100)^(2/9))``."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 1 else 0


def hac_lrv(z: np.ndarray, lag: int | None = None, kernel: str = "bartlett") -> float:
    """HAC long-run variance of a mean-zero-able scalar series ``z``.

    ``lrv = gamma_0 + sum_{l=1}^{L} k(l) * 2 * gamma_l`` with ``gamma_l`` the
    sample autocovariance and ``k`` the Bartlett (``1 - l/(L+1)``) or uniform
    kernel. The series is de-meaned first.
    """
    z = np.asarray(z, dtype=float)
    n = z.shape[0]
    if n == 0:
        return float("nan")
    L = newey_west_lag(n) if lag is None else int(lag)
    zc = z - z.mean()
    gamma0 = float(np.mean(zc * zc))
    lrv = gamma0
    for l in range(1, min(L, n - 1) + 1):
        gl = float(np.mean(zc[l:] * zc[:-l]))
        w = (1.0 - l / (L + 1.0)) if kernel == "bartlett" else 1.0
        lrv += 2.0 * w * gl
    return max(lrv, 0.0)


def normal_test(att: float, se: float, alpha: float = 0.05):
    """Two-sided N(0,1) test: returns (p_value, (ci_lower, ci_upper))."""
    if not (se > 0) or not np.isfinite(se):
        return float("nan"), (float("nan"), float("nan"))
    z = att / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    crit = norm.ppf(1.0 - alpha / 2.0)
    return float(p), (att - crit * se, att + crit * se)
