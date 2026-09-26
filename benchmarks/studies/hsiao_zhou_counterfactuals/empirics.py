"""Hsiao & Zhou (2019) Section 7: Tables 9 and 10, and the turnout tables.

The empirical panels carry covariates, so step 1 -- the coefficient beta --
exists here where it did not on DGP6/DGP7. The paper says it is estimated by
"Pesaran's (2006) CCE or Bai's (2009) method" and never says which produced
which column, so both are computed and both reported.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Step 1: the covariate coefficient
# ----------------------------------------------------------------------

def beta_cce(Y, X, T0):
    """Pesaran (2006) CCE, Equation 16, on the pre-period.

    ``Y`` is (T, N) and ``X`` is (T, N, k). The cross-sectional averages of
    (y, x) are projected out of every unit's regression.
    """
    Yp, Xp = Y[:T0], X[:T0]
    zbar = np.column_stack([Yp.mean(axis=1), Xp.mean(axis=1)])   # (T0, 1+k)
    Q, _ = np.linalg.qr(zbar)
    M = np.eye(T0) - Q @ Q.T
    A = np.zeros((X.shape[2], X.shape[2]))
    b = np.zeros(X.shape[2])
    for i in range(Y.shape[1]):
        Xi = Xp[:, i, :]
        A += Xi.T @ M @ Xi
        b += Xi.T @ M @ Yp[:, i]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def bai_objective(Y, X, beta, r):
    """``min over F, Lambda of ||Y - X beta - F Lambda'||^2``.

    Bai's estimator is the argmin of this, so it is what decides between two
    candidate coefficients. Concentrating out the factors makes it a function
    of ``beta`` alone: the minimum over rank-r matrices is the tail of the
    squared singular values of the residual.
    """
    R = Y - np.einsum("tnk,k->tn", X, beta)
    s = np.linalg.svd(R, compute_uv=False)
    return float((s[r:] ** 2).sum())


def _pooled_ols(Y, X):
    return np.linalg.lstsq(X.reshape(-1, X.shape[2]), Y.reshape(-1),
                           rcond=None)[0]


def _bai_iterate(Y, X, r, beta0, scheme="PCA1", iters=2000, tol=1e-12):
    """One run of Bai's alternating scheme from a given start.

    ``PCA1`` is Bai (2009) Equation 54 as Hsiao, Shi and Zhou (2022) write it:
    the estimated factors are projected out of the regressors as well as the
    outcome before the least-squares step. ``PCA2`` is their Equation 56,
    which subtracts the common component and regresses on raw ``X``.

    The two are not interchangeable. Hsiao, Shi and Zhou's Table 1 measures
    PCA2 at 1000 replications on Bai's own DGP1 with a pooled-OLS start: its
    bias holds near 0.13 whatever N and T are, and its empirical size reaches
    100 percent against a 5 percent nominal. PCA2 is kept here only so the
    tests can show the fit it produces is worse.
    """
    T, N, k = X.shape
    beta = np.asarray(beta0, dtype=float).copy()
    F = np.zeros((T, r))
    G = np.zeros((N, r))
    for _ in range(iters):
        R = Y - np.einsum("tnk,k->tn", X, beta)
        U, s, _ = np.linalg.svd(R, full_matrices=False)
        F = U[:, :r]                                    # T x r, orthonormal
        G = (F.T @ R).T                                 # N x r loadings
        A = np.zeros((k, k))
        b = np.zeros(k)
        if scheme == "PCA1":
            M = np.eye(T) - F @ F.T
            for i in range(N):
                Xi = X[:, i, :]
                A += Xi.T @ M @ Xi
                b += Xi.T @ M @ Y[:, i]
        else:
            common = F @ G.T
            for i in range(N):
                Xi = X[:, i, :]
                A += Xi.T @ Xi
                b += Xi.T @ (Y[:, i] - common[:, i])
        new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta, F, G


def beta_bai(Y, X, T0=None, r=2, beta0=None, iters=2000, tol=1e-12):
    """Bai (2009) interactive fixed effects, as the argmin of its objective.

    Controls only and the whole sample, which is what Xu (2017)'s Step 1 uses:
    the control units are never treated, so every period is usable. ``T0`` is
    accepted and unused, so the call sites that pass it positionally keep
    working.

    The objective is not convex and the iteration is start-dependent, so this
    runs PCA1 from several starts and keeps whichever lands lowest. On the
    38-state control panel of Table 9, a zero start reaches 41,585 and a
    pooled-OLS start 35,640; an earlier version of this function ran PCA2 from
    zeros and returned 41,760, which is 17 percent above the best point
    available and is what made the study's PCA column read 18.44 against the
    paper's 7.46.

    ``beta0`` forces a particular start, which the tests use to show the
    answer no longer depends on it.
    """
    starts = ([np.asarray(beta0, dtype=float)] if beta0 is not None
              else [_pooled_ols(Y, X), np.zeros(X.shape[2])])
    best = None
    for start in starts:
        cand = _bai_iterate(Y, X, r, start, scheme="PCA1", iters=iters,
                            tol=tol)
        value = bai_objective(Y, X, cand[0], r)
        if best is None or value < best[0]:
            best = (value, cand)
    # A forced start still gets the pooled-OLS restart as a floor, which is
    # the guarantee the tests assert: the answer is never worse than the
    # start it could have had.
    if beta0 is not None:
        fallback = _bai_iterate(Y, X, r, _pooled_ols(Y, X), scheme="PCA1",
                                iters=iters, tol=tol)
        if bai_objective(Y, X, fallback[0], r) < best[0]:
            best = (bai_objective(Y, X, fallback[0], r), fallback)
    return best[1]


def _beta_bai_pca2_from_zero(Y, X, r=2):
    """The scheme and start this study shipped, kept for the test that shows
    the fit it produces is worse. Not used by any estimator here."""
    return _bai_iterate(Y, X, r, np.zeros(X.shape[2]), scheme="PCA2")[0]


# ----------------------------------------------------------------------
# The seven constructions
# ----------------------------------------------------------------------

def e1_pca(y1, x1, Yco, Xco, T0, r):
    """Bai factors from the controls, then Xu Steps 2-3 (Equation 5)."""
    beta, F, _ = beta_bai(Yco, Xco, T0, r=r)
    target = y1[:T0] - x1[:T0] @ beta
    g1 = np.linalg.lstsq(F[:T0], target, rcond=None)[0]
    return x1 @ beta + F @ g1, beta


def _refit(target_pre, donors_pre, donors_all, keep):
    T0 = len(target_pre)
    if keep.size == 0:
        return np.full(len(donors_all), target_pre.mean())
    A = np.column_stack([np.ones(T0), donors_pre[:, keep]])
    c = np.linalg.lstsq(A, target_pre, rcond=None)[0]
    return np.column_stack([np.ones(len(donors_all)), donors_all[:, keep]]) @ c


def e2_cce(y1, x1, Yco, Xco, T0, beta):
    """Equation 11 with w unrestricted: every control residual, no selection."""
    v1 = y1 - x1 @ beta
    vco = Yco - np.einsum("tnk,k->tn", Xco, beta)
    return _refit(v1[:T0], vco[:T0], vco,
                  np.arange(vco.shape[1])) + x1 @ beta


def e3_cpda(y1, x1, Yco, Xco, T0, beta, seed=0):
    """Equation 15: a LASSO-selected subset of the control residuals."""
    v1 = y1 - x1 @ beta
    vco = Yco - np.einsum("tnk,k->tn", Xco, beta)
    f = LassoCV(cv=min(10, max(3, T0 // 3)), max_iter=100000,
                random_state=seed).fit(vco[:T0], v1[:T0])
    keep = np.flatnonzero(np.abs(f.coef_) > 0)
    return _refit(v1[:T0], vco[:T0], vco, keep) + x1 @ beta, keep.size


def e4_pda(y1, Yco, T0, seed=0):
    """Equation 22 on the control outcomes alone (Hsiao, Ching & Wan)."""
    f = LassoCV(cv=min(10, max(3, T0 // 3)), max_iter=100000,
                random_state=seed).fit(Yco[:T0], y1[:T0])
    keep = np.flatnonzero(np.abs(f.coef_) > 0)
    return _refit(y1[:T0], Yco[:T0], Yco, keep), keep.size


def e5_pdax(y1, x1, Yco, T0, seed=0):
    """E4 with the treated unit's own exogenous covariates in the pool."""
    pool = np.column_stack([Yco, x1])
    f = LassoCV(cv=min(10, max(3, T0 // 3)), max_iter=100000,
                random_state=seed).fit(pool[:T0], y1[:T0])
    keep = np.flatnonzero(np.abs(f.coef_) > 0)
    return _refit(y1[:T0], pool[:T0], pool, keep), keep.size


def averages(paths, y1, T0):
    bar = np.mean(np.asarray(paths), axis=0)
    a = float(np.mean(y1[:T0] - bar[:T0]))
    ma = a + bar
    A = np.column_stack([np.ones(T0), bar[:T0]])
    c = np.linalg.lstsq(A, y1[:T0], rcond=None)[0]
    return ma, c[0] + c[1] * bar


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------

def build(df, unit_col, time_col, y_col, cov_cols, treated, T0_year):
    wide_y = df.pivot(index=time_col, columns=unit_col, values=y_col).sort_index()
    units = [treated] + [u for u in wide_y.columns if u != treated]
    wide_y = wide_y[units]
    years = wide_y.index.to_numpy()
    T0 = int((years < T0_year).sum())
    X = np.stack([df.pivot(index=time_col, columns=unit_col,
                           values=c).sort_index()[units].to_numpy()
                  for c in cov_cols], axis=-1)              # (T, N, k)
    Y = wide_y.to_numpy()
    return Y, X, years, T0


def run_cell(Y, X, T0, r, beta_rule):
    y1, x1 = Y[:, 0], X[:, 0, :]
    Yco, Xco = Y[:, 1:], X[:, 1:, :]
    pca, beta_bai_hat = e1_pca(y1, x1, Yco, Xco, T0, r)
    beta = beta_bai_hat if beta_rule == "bai" else beta_cce(Yco, Xco, T0)
    cce = e2_cce(y1, x1, Yco, Xco, T0, beta)
    cpda, n3 = e3_cpda(y1, x1, Yco, Xco, T0, beta)
    pda, n4 = e4_pda(y1, Yco, T0)
    pdax, n5 = e5_pdax(y1, x1, Yco, T0)
    ma, mb = averages([pca, cce, cpda, pda, pdax], y1, T0)
    return ({"PCA": pca, "CCE": cce, "CPDA": cpda, "PDA": pda, "PDAX": pdax,
             "MA": ma, "MB": mb}, beta, (n3, n4, n5))


def mab(y1, path, T0):
    return float(np.mean(np.abs(y1[T0:] - path[T0:])))
