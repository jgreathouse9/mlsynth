"""Fixed-smoothing Series-HAC variance and the Sun (2013) bandwidth for OSC.

These are the inference primitives of the Orthogonalized Synthetic Control: an
orthonormal-series long-run-variance estimator and a CPE-optimal smoothing
parameter, feeding a t-test whose reference distribution is t with the smoothing
parameter as degrees of freedom (so size is controlled without a consistent
variance). Pure linear algebra -- solver-independent -- faithful to the
reference ``SeriesHAC.R``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, ncx2, t as _t


def orthonormal_basis(x, j: int):
    """Orthonormal Fourier basis function phi_j on [0, 1].

    Even ``j`` -> sqrt(2) sin(2 pi j x); odd ``j`` -> sqrt(2) cos(2 pi j x).
    """
    x = np.asarray(x, float)
    if j % 2 == 0:
        return np.sqrt(2.0) * np.sin(2.0 * np.pi * j * x)
    return np.sqrt(2.0) * np.cos(2.0 * np.pi * j * x)


def series_hac_variance(preg, postg, eta, h: int):
    """Orthonormal-series (fixed-smoothing) variance of the orthogonalized ATT."""
    preg = np.atleast_2d(np.asarray(preg, float))      # (Q, T0)
    postg = np.asarray(postg, float).ravel()           # (T1,)
    eta = np.asarray(eta, float).ravel()
    Qm1, T0 = preg.shape
    T1 = postg.shape[0]
    K = Qm1 + 1
    # Demean (the sample moments are ~0 by construction; harmless).
    preg = preg - preg.mean(axis=1, keepdims=True)
    postg = postg - postg.mean()

    freq = np.zeros((K, h))
    pre_t = np.arange(1, T0 + 1) / T0
    post_t = np.arange(1, T1 + 1) / T1
    for j in range(1, h + 1):
        freq[:Qm1, j - 1] = preg @ orthonormal_basis(pre_t, j) / T0
        freq[K - 1, j - 1] = postg @ orthonormal_basis(post_t, j) / T1
    Vg = freq @ freq.T / h
    V = min(T0, T1) * float(eta @ Vg @ eta) / eta[K - 1] ** 2
    return float(V)


def cpe_optimal_h(preg, p: int = 1, sig: float = 0.05) -> int:
    """CPE-optimal smoothing parameter K via Sun (2013), from the pre-residuals."""
    v = np.atleast_2d(np.asarray(preg, float))         # (d, T)
    d, T = v.shape
    delta2 = chi2.ppf(0.75, df=p)
    cva = chi2.ppf(1.0 - sig, df=1)
    tao = 1.15

    dep = v[:, 1:].T          # (T-1, d)
    indep = v[:, :-1].T       # (T-1, d)
    A = (dep.T @ indep) @ np.linalg.pinv(indep.T @ indep)   # (d, d) VAR(1)
    res = dep.T - A @ indep.T
    VA = (res @ res.T) / (T - 1)
    IA = np.linalg.pinv(np.eye(d) - A)
    omega0 = IA @ VA @ IA.T

    temp = np.linalg.pinv(np.eye(d) - A)
    t3 = temp @ temp @ temp
    inner = (A @ VA + A @ A @ VA @ A.T + A @ A @ VA - 6 * A @ VA @ A.T
             + VA @ A.T @ A + A @ VA @ A.T @ A.T + VA @ A.T)
    omegaq1 = t3 @ inner @ t3.T
    MB = -(np.pi ** 2) / 6.0 * omegaq1
    MBbar = float(np.trace(MB @ np.linalg.pinv(omega0)) / d)

    nr = 1
    if MBbar > 0:
        a1 = 4 * ncx2.pdf(cva, d, delta2) * abs(MBbar)
        a2 = delta2 * ncx2.pdf(cva, d + 2, delta2)
        Ktemp = (a2 / a1) ** (1.0 / 3.0) * T ** (2.0 / 3.0)
    else:
        a1 = chi2.pdf(cva, d) * cva * abs(MBbar)
        a2 = (tao - 1) * (1 - sig)
        Ktemp = (a2 / a1) ** 0.5 * T

    if Ktemp <= nr + 4:
        Kw = nr + 4
    elif Ktemp <= T:
        Kw = Ktemp
    else:
        Kw = T
    kstar = max(int(np.floor(Kw / 2)), 1)
    return int(2 * kstar)


def ttest_pvalue(beta_hat, V, h, n, beta0: float = 0.0):
    """Two-sided p-value: t_n = sqrt(n)(beta_hat - beta0)/sqrt(V), df = h."""
    t_n = np.sqrt(n) * (beta_hat - beta0) / np.sqrt(V)
    return float(2.0 * _t.cdf(-abs(t_n), df=h))


def ttest_ci(beta_hat, V, h, alpha: float):
    """t-interval for the ATT using the Series-HAC variance (reference scaling)."""
    cv = abs(_t.ppf(alpha, df=h))
    half = np.sqrt(V) * cv
    return (float(beta_hat - half), float(beta_hat + half))
