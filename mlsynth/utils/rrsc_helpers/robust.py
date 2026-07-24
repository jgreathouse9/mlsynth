"""Robust-regression kernels for RRSC (He, Li, Shi & Miao 2026).

* :func:`robust_factor_estimation` -- the large-N step: recover the common
  factors at a period by Huber-IRLS regression of the outcomes on the
  loadings, so sparse interference (the outlier component) is down-weighted
  (reference ``LargeN/largeN_SimuOne.R`` ``robust_factor_estimation``).
* :func:`lts_fit` -- the fixed-N step: high-breakdown Least Trimmed Squares
  (reference ``robustbase::ltsReg``, ``alpha=0.5``, ``intercept=FALSE``),
  including robustbase's finite-sample raw-scale correction (Pison, Van Aelst
  & Willems 2002) and the reweighting that yields ``ltsReg$coefficients``.
"""
from __future__ import annotations

from itertools import combinations
from math import comb, log

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthConfigError, MlsynthEstimationError

_EPS = np.finfo(float).eps


def robust_factor_estimation(Y_t, Lambda_hat, sigma_hat, delta: float = 1.345,
                             max_iter: int = 50, tol: float = 1e-6):
    """Huber-IRLS estimate of the factor scores ``f`` at one period.

    Solves ``argmin_f sum_i rho((Y_it - lambda_i' f)/sigma_i)`` with the Huber
    loss, so units with large (interference) residuals are down-weighted.
    """
    Y_t = np.asarray(Y_t, dtype=float).ravel()
    N = Y_t.size
    w = np.ones(N)
    f = np.zeros(Lambda_hat.shape[1])
    for _ in range(max_iter):
        W = w / (sigma_hat ** 2)
        A = Lambda_hat.T @ (W[:, None] * Lambda_hat)
        vals, vecs = np.linalg.eigh((A + A.T) / 2)
        A_inv = vecs @ (np.diag(1.0 / np.maximum(vals, _EPS))) @ vecs.T
        f = A_inv @ (Lambda_hat.T @ (W * Y_t))
        resid = (Y_t - Lambda_hat @ f) / sigma_hat
        w_new = np.where(np.abs(resid) <= delta, 1.0, delta / np.maximum(np.abs(resid), _EPS))
        if np.max(np.abs(w - w_new)) < tol:
            break
        w = w_new
    return f


def h_alpha_n(alpha: float, n: int, p: int) -> int:
    """robustbase ``h.alpha.n``: the LTS subset size ``h``."""
    fl = np.floor((n + p + 1) / 2)
    return int(2 * fl - n + 2 * (n - fl) * alpha)


def _lts_cnp2(p: int, n: int, alpha: float) -> float:
    """robustbase ``LTScnp2`` finite-sample raw-scale correction, ``intercept=FALSE``
    branch (Pison, Van Aelst & Willems 2002). Implemented for ``p >= 1``."""
    if p == 1:
        fp500 = 1 - np.exp(-0.0181777452315321) / n ** 0.697629772271099
        fp875 = 1 - np.exp(-0.310122738776431) / n ** 1.06241615923172
    else:
        g875 = np.array([[-0.251778730491252, -0.146660023184295],
                         [0.883966931611758, 0.86292940340761],
                         [3.0, 5.0]])
        e500 = np.array([[-0.487338281979106, -0.340762058011],
                         [0.405511279418594, 0.37972360544988],
                         [3.0, 5.0]])
        y500 = np.log(-e500[0] / p ** e500[1])
        y875 = np.log(-g875[0] / p ** g875[1])
        A500 = np.column_stack([np.ones(2), -np.log(e500[2] * p ** 2)])
        A875 = np.column_stack([np.ones(2), -np.log(g875[2] * p ** 2)])
        c500 = np.linalg.solve(A500, y500)
        c875 = np.linalg.solve(A875, y875)
        fp500 = 1 - np.exp(c500[0]) / n ** c500[1]
        fp875 = 1 - np.exp(c875[0]) / n ** c875[1]
    if alpha <= 0.875:
        fp = fp500 + (fp875 - fp500) / 0.375 * (alpha - 0.5)
    else:                                                       # pragma: no cover
        fp = fp875 + (1 - fp875) / 0.125 * (alpha - 0.875)
    return 1.0 / fp


def _raw_scale(resid: np.ndarray, h: int, n: int, p: int, alpha: float) -> float:
    """robustbase LTS raw scale = base * consistency * finite-sample correction."""
    base = np.sqrt(np.mean(np.sort(resid ** 2)[:h]))
    qa = norm.ppf((h / n + 1) / 2)
    consist = 1.0 / np.sqrt(1 - 2 * qa * norm.pdf(qa) / (h / n))
    return base * consist * _lts_cnp2(p, n, alpha)


def _best_h_subset(X, y, h, max_enum=20000):
    """Raw LTS fit: the h-subset with minimum OLS SSR. Exact enumeration for
    small problems, else FAST-LTS (elemental starts + C-steps)."""
    n, p = X.shape
    def ols(idx):
        c, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        return c
    if comb(n, h) <= max_enum:
        best = (np.inf, None)
        for idx in combinations(range(n), h):
            idx = list(idx)
            c = ols(idx)
            ssr = float(np.sum((y[idx] - X[idx] @ c) ** 2))
            if ssr < best[0]:
                best = (ssr, c)
        return best[1]
    # FAST-LTS: many elemental (size-p) starts + C-steps to convergence
    rng = np.random.default_rng(0)
    best = (np.inf, None)
    for _ in range(500):
        start = rng.choice(n, size=p, replace=False)
        try:
            c = ols(list(start))
        except np.linalg.LinAlgError:                          # pragma: no cover
            continue
        for _ in range(50):
            order = np.argsort((y - X @ c) ** 2)[:h]
            c_new = ols(order)
            if np.allclose(c_new, c):
                break
            c = c_new
        ssr = float(np.sum(np.sort((y - X @ c) ** 2)[:h]))
        if ssr < best[0]:
            best = (ssr, c)
    return best[1]


def lts_fit(X, y, alpha: float = 0.5, cutoff: float = None):
    """Least Trimmed Squares regression through the origin (``intercept=FALSE``),
    matching ``robustbase::ltsReg``.

    Returns ``(coef, raw_coef, weights)`` where ``coef`` is the reweighted
    estimate (``ltsReg$coefficients``): raw h-subset fit, robustbase raw scale,
    then OLS on units within ``cutoff`` standardized raw residuals.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if p < 1 or n <= p:
        raise MlsynthConfigError(f"lts_fit: require n > p >= 1; got n={n}, p={p}.")
    if cutoff is None:
        cutoff = norm.ppf(0.9875)                              # robustbase default
    h = h_alpha_n(alpha, n, p)
    raw_coef = _best_h_subset(X, y, h)
    if raw_coef is None:                                       # pragma: no cover
        raise MlsynthEstimationError("lts_fit: no valid h-subset found.")
    resid = y - X @ raw_coef
    s0 = _raw_scale(resid, h, n, p, alpha)
    w = np.abs(resid / s0) <= cutoff if s0 > 0 else np.ones(n, bool)
    if w.sum() <= p:                                           # pragma: no cover
        w = np.ones(n, bool)
    coef, *_ = np.linalg.lstsq(X[w], y[w], rcond=None)
    return coef, raw_coef, w
