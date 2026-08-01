"""Variance, covariance kernel, and the supremum test (Wied eq. 10-14).

Two sources of uncertainty enter the counterfactual at the same root-n rate and
both must be carried: the distribution-regression error in the post period, and
the weight-estimation error inherited from the pre-periods. Theorem 3.1(a)
splits the covariance into exactly those two pieces; ``V_w`` below is the
second, and it is why the pre-period cells are revisited during inference.

A caveat that shapes what is testable. Critical values come from simulating the
limiting Gaussian process, which requires a matrix square root of the estimated
kernel. ``numpy.linalg.eigh`` returns eigenvectors whose *signs* are
implementation-defined, so a different LAPACK build yields a different factor
``L`` and hence a different realisation from identical Gaussian draws -- even
with the seed pinned. The simulated distribution is unchanged; individual
critical values move by a few tenths of a percent. The test statistic itself
touches no eigendecomposition and is exact.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from .dr import link_functions


def _fisher_and_scores(theta_cell: np.ndarray, X: np.ndarray, y: np.ndarray,
                       grid: np.ndarray, link: str, eps: float = 1e-10):
    """Inverse Fisher information and score residuals at each grid point."""
    Lam, lam = link_functions(link)
    n, k = X.shape
    eta = theta_cell @ X.T                       # (m, n)
    L = np.clip(Lam(eta), 1e-10, 1 - 1e-10)
    d = lam(eta)
    wts = d * d / (L * (1 - L))
    Z = (y[None, :] <= grid[:, None]).astype(np.float64)
    psi = d * (Z - L) / (L * (1 - L))            # (m, n)
    Iinv = np.empty((len(grid), k, k))
    for l in range(len(grid)):
        info = np.einsum("n,nk,nj->kj", wts[l], X, X) / n + eps * np.eye(k)
        Iinv[l] = np.linalg.inv(info)
    return Iinv, psi


def weight_variance(theta, cells, unit_names, w, G, grid, n_total, pre_periods,
                    link: str, ridge: float = 0.0) -> np.ndarray:
    """Plug-in ``V_w`` (eq. 13): the asymptotic variance of the weights."""
    from .weights import projection_matrix

    J = len(w)
    m = len(grid)
    P = projection_matrix(G, ridge)
    Vw = np.zeros((J, J))
    for s in pre_periods:
        Th = np.stack([theta[(a + 1, s)] for a in range(J)], axis=1)   # (m,J,k)
        PT = np.einsum("jk,mka->mja", P, Th)
        for u in range(J + 1):
            cell = cells[(unit_names[u], s)]
            Iinv, psi = _fisher_and_scores(theta[(u, s)], cell.X, cell.y,
                                           grid, link)
            B = PT if u == 0 else -w[u - 1] * PT
            C = np.einsum("mjp,mpq->mjq", B, Iinv)
            CX = np.einsum("mjq,nq->mjn", C, cell.X)
            D = np.einsum("mjn,mn->jn", CX, psi)
            Vw += (n_total / cell.n ** 2) * (D @ D.T)
    Vw /= (len(pre_periods) ** 2 * m ** 2)
    return 0.5 * (Vw + Vw.T)


def _cell_components(theta_cell, X, y, x, grid, link, eps: float = 1e-12):
    Lam, lam = link_functions(link)
    n, k = X.shape
    eta = theta_cell @ X.T
    L = np.clip(Lam(eta), 1e-10, 1 - 1e-10)
    d = lam(eta)
    wts = d * d / (L * (1 - L))
    Z = (y[None, :] <= grid[:, None]).astype(np.float64)
    r = d * (Z - L) / (L * (1 - L))
    A = np.empty((len(grid), n))
    for l in range(len(grid)):
        Iinv = np.linalg.inv(np.einsum("n,nk,nj->kj", wts[l], X, X) / n
                             + eps * np.eye(k))
        A[l] = (X @ Iinv) @ x
    return A, r


def covariance_kernel(theta, cells, unit_names, w, th0, t, x, grid, n_total,
                      Vw, link: str) -> np.ndarray:
    """Sandwich estimator of the kernel ``K`` (eq. 11)."""
    _, lam = link_functions(link)
    J = len(w)
    m = len(grid)
    lam1 = lam(theta[(0, t)] @ x)
    lam0 = lam(th0 @ x)
    c0 = cells[(unit_names[0], t)]
    A1, r1 = _cell_components(theta[(0, t)], c0.X, c0.y, x, grid, link)
    G1 = A1 * r1
    K = (lam1[:, None] * lam1[None, :]) * ((n_total / c0.n ** 2) * (G1 @ G1.T))
    ctrl = np.zeros((m, m))
    Tt = np.zeros((m, J))
    for a in range(J):
        ca = cells[(unit_names[a + 1], t)]
        Ai, ri = _cell_components(theta[(a + 1, t)], ca.X, ca.y, x, grid, link)
        Gi = Ai * ri
        ctrl += (w[a] ** 2) * ((n_total / ca.n ** 2) * (Gi @ Gi.T))
        Tt[:, a] = theta[(a + 1, t)] @ x
    K += (lam0[:, None] * lam0[None, :]) * (ctrl + Tt @ Vw @ Tt.T)
    return 0.5 * (K + K.T)


def supremum_test(delta: np.ndarray, K: np.ndarray, n_total: int,
                  idx: Optional[np.ndarray], alpha: float, seed: int,
                  n_draws: int) -> Tuple[float, float, float]:
    """Statistic, critical value, and p-value for ``H0: Delta = 0`` (eq. 14)."""
    if idx is None:
        idx = np.arange(len(delta))
    idx = np.asarray(idx, dtype=int)
    Ks = K[np.ix_(idx, idx)]
    stat = float(np.sqrt(n_total) * np.max(np.abs(delta[idx])))
    ev, V = np.linalg.eigh(0.5 * (Ks + Ks.T))
    L = V @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))
    draws = np.random.default_rng(seed).standard_normal((len(idx), n_draws))
    sim = np.max(np.abs(L @ draws), axis=0)
    return stat, float(np.quantile(sim, 1 - alpha)), float(np.mean(sim >= stat))


def integrated_effect(delta: np.ndarray, K: np.ndarray, n_total: int,
                      idx: Optional[np.ndarray], alpha: float
                      ) -> Tuple[float, float, float]:
    """``f = mean(delta^2)`` with its standard error and one-sided bound (eq. 10).

    The bound is reported unconditionally but is only meaningful once the
    supremum test has rejected: under the null the delta-method derivative
    vanishes, the variance degenerates, and the bound collapses to zero
    (Remark 3.3).
    """
    if idx is None:
        idx = np.arange(len(delta))
    idx = np.asarray(idx, dtype=int)
    d = delta[idx]
    m = len(d)
    Ks = K[np.ix_(idx, idx)]
    f_hat = float(np.mean(d ** 2))
    var = float((4.0 / m ** 2) * (d @ Ks @ d))
    se = float(np.sqrt(max(var, 0.0)) / np.sqrt(n_total))
    return f_hat, se, float(max(0.0, f_hat - norm.ppf(1 - alpha) * se))
