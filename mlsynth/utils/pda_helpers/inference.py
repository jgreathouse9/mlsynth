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


def fspda_lrvar_lag(T2: int, lrvar_lag: int | None = None) -> int:
    """Bartlett-kernel truncation lag for the fsPDA long-run variance.

    Follows Shi & Huang's released ``fsPDA`` package (``est.fsPDA``): the default
    is ``floor(T2 ** (1/4))`` and a user-supplied lag must be a non-negative
    integer no larger than ``floor(sqrt(T2))``.
    """
    cap = int(np.floor(np.sqrt(T2))) if T2 > 0 else 0
    if lrvar_lag is None:
        return int(np.floor(T2 ** 0.25)) if T2 > 0 else 0
    lrvar_lag = int(lrvar_lag)
    if lrvar_lag < 0 or lrvar_lag > cap:
        raise ValueError(
            f"lrvar_lag must be a non-negative integer no larger than "
            f"floor(sqrt(T2))={cap}; got {lrvar_lag}."
        )
    return lrvar_lag


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


def lrvar_prewhite_nw(x: np.ndarray) -> float:
    """Prewhitened Newey-West long-run variance of the **mean** of ``x``.

    Andrews-Monahan (1992) VAR(1) prewhitening + a Bartlett kernel with the
    Newey-West (1994) data-driven bandwidth + the finite-sample adjustment --
    i.e. R's ``sandwich::lrvar(x, type = "Newey-West", prewhite = TRUE,
    adjust = TRUE)``, which Shi & Huang use for the post-selection t-test in
    their applications (``app1_luxury_watch/fsPDA.R``). Returns the variance of
    the sample mean (so the t-statistic is ``mean(x) / sqrt(lrvar_prewhite_nw(x))``).

    Prewhitening is what makes this differ sharply from the plain Bartlett
    estimator on strongly serially-dependent effects (e.g. monthly growth
    rates, which mean-revert): the VAR(1) step removes the bulk of the
    dependence before the kernel smooths the remainder, then the long-run
    factor ``1 / (1 - A)`` recolors it.
    """
    x = np.asarray(x, dtype=float)
    n0 = x.shape[0]
    h = x - x.mean()
    if n0 < 4:
        return float(np.dot(h, h) / max(n0 * n0, 1))     # fall back to var(mean)

    # VAR(1) prewhitening (OLS, no intercept); cap |A| < 0.97 for stability.
    h_lead, h_lag = h[1:], h[:-1]
    denom = float(h_lag @ h_lag)
    A = float(h_lag @ h_lead) / denom if denom > 0 else 0.0
    A = float(np.clip(A, -0.97, 0.97))
    e = h_lead - A * h_lag
    n = e.shape[0]

    def sig(j: int) -> float:
        return float(np.sum(e[j:] * e[:n - j]) / n)

    # Newey-West (1994) automatic bandwidth on the prewhitened residuals.
    L0 = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    s0 = sig(0) + 2.0 * sum(sig(j) for j in range(1, L0 + 1))
    s1 = 2.0 * sum(j * sig(j) for j in range(1, L0 + 1))
    if s0 > 0:
        st = 1.1447 * ((s1 / s0) ** 2) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    else:
        st = float(L0)
    L = int(min(np.floor(st), n - 1))

    s_resid = sig(0) + 2.0 * sum((1.0 - j / (st + 1.0)) * sig(j)
                                 for j in range(1, L + 1))
    s_resid = max(s_resid, 0.0)
    recolor = 1.0 / (1.0 - A)
    omega = recolor * s_resid * recolor
    omega *= n / (n - 1.0)                                # adjust = TRUE
    return float(omega / n)                              # variance of the mean


def normal_test(att: float, se: float, alpha: float = 0.05):
    """Two-sided N(0,1) test: returns (p_value, (ci_lower, ci_upper))."""
    if not (se > 0) or not np.isfinite(se):
        return float("nan"), (float("nan"), float("nan"))
    z = att / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    crit = norm.ppf(1.0 - alpha / 2.0)
    return float(p), (att - crit * se, att + crit * se)
