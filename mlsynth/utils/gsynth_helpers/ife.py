"""Interactive fixed effects on the never-treated units -- Xu (2017) Step 1.

The control-unit model is

.. math::

   Y_{it} = \\mu + \\alpha_i + \\xi_t + x_{it}'\\beta
            + \\lambda_i' f_t + \\varepsilon_{it},
   \\qquad i \\in \\mathcal{C},

fit by least squares subject to Bai's (2009) normalization
``F'F / T = I_r`` with ``\\Lambda'\\Lambda`` diagonal. The additive effects are
swept out by demeaning first, so the factor step sees only the interactive part;
what remains alternates between a pooled regression for ``\\beta`` and principal
components for ``(F, \\Lambda)`` until the coefficient vector stops moving.

The two coefficient-free cases are closed form: with no covariates the factors
are the principal components of the demeaned panel, and with ``r = 0`` the fit
is the two-way within estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ControlFit:
    """The Step 1 fit, in the pieces Steps 2 and 3 consume.

    Parameters
    ----------
    mu : float
        Grand mean.
    alpha : np.ndarray
        Control-unit effects, shape ``(N_co,)``. Zero when ``two_way`` is off.
    xi : np.ndarray
        Period effects, shape ``(T,)``.
    beta : np.ndarray
        Covariate coefficients, shape ``(p,)``, with zeros in the positions of
        any covariate dropped for want of within variation.
    factor : np.ndarray
        Estimated factors, shape ``(T, r)``, normalized ``F'F / T = I_r``.
    lam : np.ndarray
        Control-unit loadings, shape ``(N_co, r)``.
    niter : int
        Alternating-least-squares iterations taken.
    dropped : tuple of int
        Positions of covariates dropped as collinear with the fixed effects.
    """

    mu: float
    alpha: np.ndarray
    xi: np.ndarray
    beta: np.ndarray
    factor: np.ndarray
    lam: np.ndarray
    niter: int = 0
    dropped: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def r(self) -> int:
        """Number of factors."""
        return int(self.factor.shape[1])


def panel_factor(residuals: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-``r`` principal components of a ``(T, N)`` residual matrix.

    The eigenvectors are taken from whichever second-moment matrix is smaller,
    and scaled so the pair sits in Bai's normalization: ``F'F / T = I_r`` when
    ``T < N``, and ``\\Lambda'\\Lambda / N = I_r`` otherwise.

    Parameters
    ----------
    residuals : np.ndarray
        Residual matrix, shape ``(T, N)``.
    r : int
        Number of components. ``0`` returns empty matrices.

    Returns
    -------
    factor : np.ndarray
        Shape ``(T, r)``.
    lam : np.ndarray
        Shape ``(N, r)``.
    """
    T, N = residuals.shape
    if r <= 0:
        return np.zeros((T, 0)), np.zeros((N, 0))
    if T < N:
        left, _s, _vt = np.linalg.svd(residuals @ residuals.T / (N * T))
        factor = left[:, :r] * np.sqrt(T)
        lam = residuals.T @ factor / T
    else:
        left, _s, _vt = np.linalg.svd(residuals.T @ residuals / (N * T))
        lam = left[:, :r] * np.sqrt(N)
        factor = residuals @ lam / N
    return factor, lam


def _pooled_beta(X: np.ndarray, xxinv: np.ndarray, Y: np.ndarray,
                 fitted: np.ndarray) -> np.ndarray:
    """``beta = (X'X)^{-1} X'(Y - fitted)`` for a ``(T, N, p)`` covariate stack."""
    return xxinv @ np.einsum("tnk,tn->k", X, Y - fitted)


def interactive_fixed_effects(
    Y: np.ndarray,
    X: np.ndarray,
    r: int,
    *,
    two_way: bool = True,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> ControlFit:
    """Fit the control-unit model of Xu (2017) Section 3, Step 1.

    Parameters
    ----------
    Y : np.ndarray
        Control outcomes, shape ``(T, N_co)``.
    X : np.ndarray
        Covariates, shape ``(T, N_co, p)``; ``p = 0`` is allowed.
    r : int
        Number of factors.
    two_way : bool
        Include additive unit effects alongside the period effects.
    tol : float
        Convergence tolerance on the coefficient vector.
    max_iter : int
        Iteration cap.

    Returns
    -------
    ControlFit
    """
    T, N = Y.shape
    p = int(X.shape[2]) if X.size or X.ndim == 3 else 0

    YY = np.array(Y, dtype=float, copy=True)
    XX = np.array(X, dtype=float, copy=True) if p else np.empty((T, N, 0))

    mu_Y = float(YY.mean())
    YY -= mu_Y
    mu_X = XX.reshape(-1, p).mean(axis=0) if p else np.zeros(0)
    if p:
        XX -= mu_X

    alpha_Y, alpha_X = np.zeros(N), np.zeros((N, p))
    if two_way:
        alpha_Y = YY.mean(axis=0)
        YY -= alpha_Y[None, :]
        if p:
            alpha_X = XX.mean(axis=0)
            XX -= alpha_X[None, :, :]

    xi_Y = YY.mean(axis=1)
    YY -= xi_Y[:, None]
    xi_X = np.zeros((T, p))
    if p:
        xi_X = XX.mean(axis=1)
        XX -= xi_X[:, None, :]

    # A covariate with no within variation left is collinear with the fixed
    # effects. It contributes nothing and would make X'X singular, so it is
    # dropped and reported with a zero coefficient.
    keep = [k for k in range(p) if np.abs(XX[:, :, k]).sum() >= 1e-5]
    dropped = tuple(k for k in range(p) if k not in keep)
    Xk = XX[:, :, keep] if keep else np.empty((T, N, 0))
    pk = len(keep)

    beta_k = np.zeros(pk)
    niter = 0
    if pk == 0:
        factor, lam = panel_factor(YY, r)
    else:
        xxinv = np.linalg.inv(np.einsum("tnk,tnm->km", Xk, Xk))
        beta_k = _pooled_beta(Xk, xxinv, YY, np.zeros((T, N)))
        if r == 0:
            factor, lam = panel_factor(YY, 0)
        else:
            resid = YY - np.einsum("tnk,k->tn", Xk, beta_k)
            factor, lam = panel_factor(resid, r)
            beta_old, step = beta_k, np.inf
            while step > tol and niter < max_iter:
                niter += 1
                beta_k = _pooled_beta(Xk, xxinv, YY, factor @ lam.T)
                step = float(np.linalg.norm(beta_k - beta_old))
                beta_old = beta_k
                resid = YY - np.einsum("tnk,k->tn", Xk, beta_k)
                factor, lam = panel_factor(resid, r)

    beta = np.zeros(p)
    for slot, k in enumerate(keep):
        beta[k] = beta_k[slot]

    return ControlFit(
        mu=float(mu_Y - mu_X @ beta),
        alpha=(alpha_Y - alpha_X @ beta) if two_way else np.zeros(N),
        xi=xi_Y - xi_X @ beta,
        beta=beta,
        factor=factor,
        lam=lam,
        niter=niter,
        dropped=dropped,
    )
