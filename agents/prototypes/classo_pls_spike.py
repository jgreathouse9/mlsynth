"""SPIKE CODE -- not library code. See ``agents/scope_classo.md``.

A minimal transcription of the C-Lasso penalized-least-squares estimator of

    Su, L., Shi, Z. & Phillips, P. C. B. (2016), "Identifying Latent Structures
    in Panel Data", Econometrica 84(6), 2215-2264,

built to answer one scoping question: does C-Lasso donor selection improve a
convex synthetic control? It exists so the scope's measured claims can be
re-run, and so the algorithm does not have to be re-derived when the estimator
is built. It is deliberately outside ``mlsynth/``: no Pydantic config, no
result contract, no test suite, no error translation. The library version lands
test-first, to the contract in ``CLAUDE.md``.

Cross-validated against the reference R package ``zhan-gao/classo`` on its own
``data/sample_data.rda`` (SSP's DGP 1, N=200, T=25, K=3). This file returns

    [[0.4017 1.6014] [1.0388 0.9987] [1.6197 0.3614]]

with 5 of 200 units misclassified, matching the values that package's README
publishes for its ``PLS.cvxr`` run to four decimals, on the same five units.

Transcribed from ``PLS.cvxr`` in that package and ``SSP_PLS_est.m`` in the
authors' own MATLAB repository ``zhentaoshi/C-Lasso``. Omits the split-panel
jackknife bias correction those ship for dynamic panels (``SPJ_PLS``); the
Nickell bias is O(1/T) and common across units, so it shifts the level of a
lagged-dependent coefficient without creating or destroying a grouping.
"""
from __future__ import annotations
import numpy as np, cvxpy as cp


def demean(A, N, T):
    """Within transformation: subtract each unit's own time mean."""
    B = A.reshape(N, T, -1) if A.ndim > 1 else A.reshape(N, T, 1)
    return (B - B.mean(axis=1, keepdims=True)).reshape(N * T, -1)


def _solve_k(yb, Xb, gamma, N, T, p, lam):
    """One convex block: (N, T) residual matrix, so no vec() ordering ambiguity."""
    b = cp.Variable((N, p))
    a = cp.Variable((1, p))
    # fitted[i, t] = Xb[i, t, :] @ b[i, :]  -- built column by column over p
    fitted = sum(cp.multiply(Xb[:, :, j], cp.reshape(b[:, j], (N, 1), order="C"))
                 for j in range(p))
    pen = gamma @ cp.norm(b - cp.vstack([a] * N), axis=1)
    prob = cp.Problem(cp.Minimize(
        cp.sum_squares(yb - fitted) / (N * T) + (lam / N) * pen))
    prob.solve(solver=cp.CLARABEL)
    return b.value, np.asarray(a.value).ravel()


def classo_pls(y, X, N, T, K, lam, max_iter=100, tol=1e-4, seed=0):
    """Returns (labels 0-based, alpha (K,p), beta (N,p), converged)."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float).reshape(N * T, -1)
    p = X.shape[1]
    Xb = X.reshape(N, T, p)
    yb = y.reshape(N, T)

    # individual OLS as the initial value (init_est)
    beta0 = np.stack([np.linalg.lstsq(Xb[i], yb[i], rcond=None)[0] for i in range(N)])

    b = np.repeat(beta0[:, None, :], K, axis=1)        # (N, K, p) -- b^{(k)}
    alpha = np.zeros((K, p))
    # spread the initial centres over the OLS cloud so the K blocks differ
    q = np.linspace(10, 90, K)
    for k in range(K):
        alpha[k] = np.percentile(beta0, q[k], axis=0)

    a_old, b_old = alpha[K - 1].copy(), b[:, K - 1, :].copy()
    converged = False
    for _ in range(max_iter):
        for k in range(K):
            others = [j for j in range(K) if j != k]
            gamma = np.prod([np.linalg.norm(b[:, j, :] - alpha[j], axis=1)
                             for j in others], axis=0) if others else np.ones(N)
            bk, ak = _solve_k(yb, Xb, gamma, N, T, p, lam)
            alpha[k], b[:, k, :] = ak, bk
        if (np.max(np.abs(alpha[K-1] - a_old)) < tol
                and np.max(np.abs(b[:, K-1, :] - b_old)) < tol):
            converged = True
            break
        a_old, b_old = alpha[K-1].copy(), b[:, K-1, :].copy()

    dist = np.stack([np.linalg.norm(b[:, k, :] - alpha[k], axis=1) for k in range(K)], 1)
    labels = dist.argmin(axis=1)

    # post-Lasso: pooled OLS within each estimated group
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx):
            Ak = Xb[idx].reshape(-1, p); yk = yb[idx].ravel()
            alpha[k] = np.linalg.lstsq(Ak, yk, rcond=None)[0]
    return labels, alpha, alpha[labels], converged


def info_criterion(y, X, N, T, labels, alpha, rho=None):
    """SSP eq. (2.9): post-Lasso MSE + rho * p * K."""
    X = np.asarray(X, float).reshape(N * T, -1); p = X.shape[1]
    yb = y.reshape(N, T); Xb = X.reshape(N, T, p)
    resid = np.concatenate([yb[i] - Xb[i] @ alpha[labels[i]] for i in range(N)])
    if rho is None:
        rho = (2.0 / 3.0) * (N * T) ** -0.5
    return float(np.mean(resid ** 2) + rho * p * len(np.unique(labels)))
