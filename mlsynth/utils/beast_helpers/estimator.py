"""Core numerics for the BEAST immunized doubly-robust synthetic control.

Bléhaut, D'Haultfœuille, L'Hour & Tsybakov (2021), *"An alternative to synthetic
control for models with many covariates under sparsity"* (arXiv 2005.12225). A
NumPy port of the authors' R (``CalibrationLasso.R`` / ``OrthogonalityReg.R`` /
``LassoFISTA.R`` / ``ImmunizedATT.R`` in ``jeremylhour/alternative-synthetic-
control-sparsity``), validated against a live run of that code on the Abadie,
Diamond & Hainmueller (2010) Proposition 99 panel: the calibration weights match
to five decimals and the immunized ATT path to ~0.15 packs (mean to ~0.02).

Three steps:

* :func:`calibration_lasso` -- covariate-balancing (exponential-tilting) weights
  ``W = (1 - d) exp(X beta) / n_1`` with an ℓ₁-penalised calibration
  (Neyman-orthogonal, sparse in high dimensions);
* :func:`orthogonality_reg` -- a weighted-least-squares Lasso outcome model
  ``mu`` (the immunizing regression of the outcome on ``X`` for the controls);
* :func:`immunized_att` -- the doubly-robust ATT
  ``theta = mean[(d - (1 - d) exp(X beta))(y - X mu)] / pi`` with a closed-form
  asymptotic standard error.

The calibration is a valid synthetic control only when the weights balance the
constant, i.e. ``sum(W) == 1``; :func:`balance_ok` exposes that check (the exact
signal that flags the over-saturated high-dimensional regime the method is not
built for).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_ACTIVE = 1e-6


def _soft(x: np.ndarray, lam: float, nopen: np.ndarray) -> np.ndarray:
    """Soft-threshold ``x`` by ``lam``; entries in ``nopen`` are left unpenalised."""
    y = np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)
    y[nopen] = x[nopen]
    return y


def _bch_lambda(c: float, n: int, p: int, g: float = 0.05) -> float:
    """Belloni-Chernozhukov-Hansen penalty level ``c * Phi^{-1}(1 - g/2p) / sqrt(n)``."""
    return c * norm.ppf(1.0 - 0.5 * g / p) / np.sqrt(n) if c > 0 else 0.0


def calibration_lasso(d: np.ndarray, X: np.ndarray, c: float = 0.03,
                      max_iter_pen: int = 10000, tol_psi: float = 0.01) -> np.ndarray:
    """ℓ₁-penalised covariate-balancing (calibration) coefficients ``beta``.

    Minimises ``mean[(1 - d) exp(X beta) - d (X beta)] + lambda ||Psi beta||_1``
    (the intercept, column 0, is unpenalised) by proximal-gradient descent, with
    the penalty loadings ``Psi`` iterated as in the reference. The first column of
    ``X`` must be a constant.
    """
    d = np.asarray(d, float).ravel()
    X = np.asarray(X, float)
    n, p = X.shape
    lam = _bch_lambda(c, n, p)
    n1, n0 = d.sum(), (1.0 - d).sum()
    beta = np.concatenate([[np.log(n1 / n0)], np.zeros(p - 1)])
    nopen = np.array([0])

    def psi_of(b):
        r = (d - np.exp(X @ b) * (1.0 - d)) ** 2
        return np.sqrt((r[:, None] * X ** 2).mean(axis=0))

    psi = psi_of(beta)
    for _ in range(max_iter_pen):
        Xs = X / psi[None, :]                       # X %*% solve(Psi)
        th = psi * beta                             # Psi %*% beta

        def smooth(t):
            z = Xs @ t
            return float(np.mean((1.0 - d) * np.exp(z) - d * z))

        def grad(t):
            z = Xs @ t
            return (((1.0 - d) * np.exp(z) - d)[:, None] * Xs).mean(axis=0)

        step = 1.0
        for _inner in range(2000):                  # prox-grad with backtracking
            g, f = grad(th), smooth(th)
            while True:
                thn = _soft(th - step * g, lam * step, nopen)
                if (smooth(thn) <= f + g @ (thn - th)
                        + (0.5 / step) * np.sum((thn - th) ** 2) or step < 1e-12):
                    break
                step *= 0.5
            if np.max(np.abs(thn - th)) < 1e-10:
                th = thn
                break
            th = thn
            step = min(step * 1.2, 1.0)
        beta = th / psi
        prepsi, psi = psi, psi_of(beta)
        if not np.all(np.isfinite(np.exp(X @ beta))):
            break
        if np.max(np.abs(psi - prepsi)) < tol_psi:
            break
    return beta


def _fista_wls(y: np.ndarray, X: np.ndarray, W: np.ndarray, lam: float,
               nopen, max_iter: int = 200000, tol: float = 1e-6) -> np.ndarray:
    """FISTA for the weighted-least-squares Lasso ``mean[W (y - X b)^2] + lam||b||_1``."""
    sw = np.sqrt(W)
    yw, Xw = sw * y, X * sw[:, None]
    n = X.shape[0]
    eta = 1.0 / np.max(2.0 * np.linalg.eigvalsh(Xw.T @ Xw) / n)
    nopen = np.asarray(nopen)
    pen = np.array([j for j in range(X.shape[1]) if j not in set(nopen.tolist())])
    beta = np.zeros(X.shape[1])
    v = beta.copy()
    theta = 1.0

    def obj(b):
        l1 = np.sum(np.abs(b[pen])) if pen.size else 0.0
        return float(np.mean((yw - Xw @ b) ** 2) + lam * l1)

    for _ in range(max_iter):
        th_o, theta = theta, (1.0 + np.sqrt(1.0 + 4.0 * theta ** 2)) / 2.0
        delta = (1.0 - th_o) / theta
        b_o = beta
        g = -2.0 * (yw - Xw @ v) @ Xw / n
        beta = _soft(v - eta * g, lam * eta, nopen)
        v = (1.0 - delta) * beta + delta * b_o
        if abs(obj(beta) - obj(b_o)) < tol:
            break
    return beta


def orthogonality_reg(y: np.ndarray, d: np.ndarray, X: np.ndarray, beta: np.ndarray,
                      c: float = 0.3, max_iter_pen: int = 1000,
                      tol_psi: float = 0.01) -> np.ndarray:
    """Weighted-Lasso immunizing outcome model ``mu`` (regress ``y`` on ``X``,
    control weights ``W = (1 - d) exp(X beta)``)."""
    d = np.asarray(d, float).ravel()
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    n, p = X.shape
    lam = _bch_lambda(c, n, p)
    W = (1.0 - d) * np.exp(X @ beta)

    def psi_of(mu):
        r = W * (y - X @ mu) ** 2
        return np.sqrt((r @ (X * np.sqrt(W)[:, None]) ** 2) / n)

    m_y = (W @ y) / W.sum()
    psi = np.sqrt((W * (y - m_y) ** 2 @ (X * np.sqrt(W)[:, None]) ** 2) / n)
    mu = np.zeros(p)
    for _ in range(max_iter_pen):
        th = _fista_wls(y, X / psi[None, :], W, lam, nopen=[0])
        mu = th / psi
        prepsi, psi = psi, psi_of(mu)
        if np.max(np.abs(psi - prepsi)) < tol_psi:
            break
    return mu


def immunized_att(y: np.ndarray, d: np.ndarray, X: np.ndarray, beta: np.ndarray,
                  mu: np.ndarray | None = None):
    """Immunized doubly-robust ATT ``theta`` and its asymptotic standard error.

    ``theta = mean[(d - (1 - d) exp(X beta)) eps] / pi`` with ``eps = y - X mu``
    (or ``eps = y`` when ``mu`` is None, the non-immunized plug-in).
    """
    d = np.asarray(d, float).ravel()
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    eps = y if mu is None else y - X @ mu
    pi = d.mean()
    w = d - (1.0 - d) * np.exp(X @ beta)
    theta = float(np.mean(w * eps) / pi)
    psi = w * eps - d * theta
    sigma = float(np.sqrt(np.mean(psi ** 2) / pi ** 2) / np.sqrt(len(y)))
    return theta, sigma


def balance_weights(d: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Donor balancing weights ``W = (1 - d) exp(X beta) / n_1`` (should sum to 1)."""
    d = np.asarray(d, float).ravel()
    return (1.0 - d) * np.exp(X @ beta) / d.sum()


def balance_ok(weights: np.ndarray, tol: float = 0.05) -> bool:
    """True when the balancing weights are valid (``|sum(W) - 1| <= tol``)."""
    return bool(abs(float(np.sum(weights)) - 1.0) <= tol)
