"""Step 1 of CPDA: the covariate slope.

Hsiao and Zhou (2019) say only that ``beta`` comes from "Bai's (2009) or
Pesaran's (2006) method" and never say which produced which published column,
so both are here and the caller picks.

The two suit different shapes. Bai's interactive fixed effects needs both ``N``
and ``T`` large, which Remark 1 calls a luxury; Pesaran's common correlated
effects estimator is consistent as ``N`` grows with ``T`` fixed, which is the
regime most panels are in.
"""

from __future__ import annotations

import numpy as np


def beta_cce(Y: np.ndarray, X: np.ndarray, T0: int) -> np.ndarray:
    """Pesaran (2006) common correlated effects, Equation 16, on the pre-period.

    The cross-sectional averages of ``(y, x)`` stand in for the unobserved
    factors and are projected out of every unit's regression:
    ``beta = (sum_i Xi' M Xi)^-1 sum_i Xi' M yi`` with
    ``M = I - Zbar (Zbar' Zbar)^-1 Zbar'``.

    Parameters
    ----------
    Y : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    X : np.ndarray
        Donor covariates, shape ``(T, N, k)``.
    T0 : int
        Pre-period length; only the first ``T0`` rows are used.

    Returns
    -------
    np.ndarray
        The slope, shape ``(k,)``.

    Notes
    -----
    The projector is built from the raw averages, as Equation 16 prints them.
    Demeaning them first is a live convention -- some implementations do -- and
    it is not cosmetic: on Hsiao and Zhou's Table 9 panel it moves the CCE
    column from 8.62 to 15.24 against a published 9.12, so the printed form is
    the one that reproduces.
    """
    Yp = np.asarray(Y[:T0], dtype=float)
    Xp = np.asarray(X[:T0], dtype=float)
    k = Xp.shape[2]
    zbar = np.column_stack([Yp.mean(axis=1), Xp.mean(axis=1)])
    Q, _ = np.linalg.qr(zbar)
    M = np.eye(T0) - Q @ Q.T
    A = np.zeros((k, k))
    b = np.zeros(k)
    for i in range(Yp.shape[1]):
        Xi = Xp[:, i, :]
        A += Xi.T @ M @ Xi
        b += Xi.T @ M @ Yp[:, i]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def bai_objective(Y: np.ndarray, X: np.ndarray, beta: np.ndarray, r: int) -> float:
    """``min over F, Lambda of ||Y - X beta - F Lambda'||^2``.

    Bai's estimator is the argmin of this, so it is what adjudicates between
    two candidate slopes. Concentrating the factors out makes it a function of
    ``beta`` alone: the best rank-``r`` approximation error is the tail of the
    squared singular values of the residual.
    """
    R = np.asarray(Y, dtype=float) - np.einsum("tnk,k->tn", X, beta)
    s = np.linalg.svd(R, compute_uv=False)
    return float((s[r:] ** 2).sum())


def _pooled_ols(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X.reshape(-1, X.shape[2]),
                           np.asarray(Y, dtype=float).reshape(-1), rcond=None)[0]


def _iterate(Y, X, r, beta0, iters=2000, tol=1e-12):
    """Bai (2009) Equation 54, as Hsiao, Shi and Zhou (2022) write it.

    The estimated factors are projected out of the regressors as well as the
    outcome before the least-squares step. Their Equation 56 instead subtracts
    the common component and regresses on raw ``X``; their Table 1 measures it
    at a bias near 0.13 that does not shrink in ``N`` or ``T``, with empirical
    size reaching 100 percent against a 5 percent nominal, so only Equation 54
    is implemented here.
    """
    T, N, k = X.shape
    beta = np.asarray(beta0, dtype=float).copy()
    for _ in range(iters):
        R = np.asarray(Y, dtype=float) - np.einsum("tnk,k->tn", X, beta)
        U, _, _ = np.linalg.svd(R, full_matrices=False)
        F = U[:, :r]
        M = np.eye(T) - F @ F.T
        A = np.zeros((k, k))
        b = np.zeros(k)
        for i in range(N):
            Xi = X[:, i, :]
            A += Xi.T @ M @ Xi
            b += Xi.T @ M @ Y[:, i]
        new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            return new
        beta = new
    return beta


def beta_bai(Y: np.ndarray, X: np.ndarray, r: int = 2, iters: int = 2000,
             tol: float = 1e-12) -> np.ndarray:
    """Bai (2009) interactive fixed effects, as the argmin of its objective.

    Run on the donors over every period, which is Xu (2017)'s Step 1: the
    donors are never treated, so no period has to be held out.

    The objective is not convex and the iteration is start-dependent, so this
    runs from a pooled-OLS start and from zeros and keeps whichever lands
    lower. The restart is not decoration. On Hsiao and Zhou's 38-state control
    panel a zero start reaches 41,585 against pooled OLS's 35,640, and a
    version of this study that started from zeros returned a slope 18 percent
    above the best point available, which is a well-behaved counterfactual
    built on a coefficient that does not minimise anything.

    Parameters
    ----------
    Y : np.ndarray
        Donor outcomes, shape ``(T, N)``.
    X : np.ndarray
        Donor covariates, shape ``(T, N, k)``.
    r : int
        Number of factors.

    Returns
    -------
    np.ndarray
        The slope, shape ``(k,)``.
    """
    r = max(1, min(int(r), min(Y.shape) - 1)) if min(Y.shape) > 1 else 1
    best_beta, best_val = None, np.inf
    for start in (_pooled_ols(Y, X), np.zeros(X.shape[2])):
        cand = _iterate(Y, X, r, start, iters=iters, tol=tol)
        val = bai_objective(Y, X, cand, r)
        if val < best_val:
            best_beta, best_val = cand, val
    return best_beta


def estimate_beta(Y: np.ndarray, X: np.ndarray, T0: int, method: str = "cce",
                  r: int = 2) -> np.ndarray:
    """Dispatch to the requested slope estimator."""
    if method == "cce":
        return beta_cce(Y, X, T0)
    return beta_bai(Y, X, r=r)
