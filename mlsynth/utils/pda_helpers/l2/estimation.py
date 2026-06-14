"""L2-relaxation estimation (Shi & Wang 2024).

Solves the L2-relaxation primal (their Eq. 3)

    min_beta  (1/2) ||beta||_2^2   s.t.   || eta_hat - Sigma_hat beta ||_inf <= tau,

with ``Sigma_hat = X'X / T1`` and ``eta_hat = X'y / T1`` over the pre-period.

Following the authors' released ``L2relax`` (https://github.com/ishwang1/L2relax-PDA),
the treated and control series are **standardised** (demeaned and scaled to unit
variance) before forming ``Sigma`` / ``eta``, and the solution is mapped back to
the original scale: ``beta = sd_y * (beta_tilde / sd_X)`` and intercept
``alpha = mean_y - mean_X . beta``. Standardisation is the default (it is what
the replication code does and what reproduces the paper's empirical results);
set ``standardize=False`` for the raw-scale variant. The tuning parameter
``tau`` is chosen by out-of-sample validation over a log-spaced grid (the
optimal ``tau`` is often a tiny fraction of ``max|eta|``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError
from .batch import l2_relax_grid, l2_relax_solve


def _standardize(y: np.ndarray, X: np.ndarray, standardize: bool):
    """Return ``(mu_y, Mu_X, sd_y, Sd_X)`` for the L2relax (de)standardisation."""
    mu_y = float(np.mean(y))
    Mu_X = X.mean(axis=0)
    if standardize:
        sd_y = float(np.std(y, ddof=1)) or 1.0
        Sd_X = X.std(axis=0, ddof=1)
        Sd_X = np.where(Sd_X > 0, Sd_X, 1.0)
    else:
        sd_y, Sd_X = 1.0, np.ones(X.shape[1])
    return mu_y, Mu_X, sd_y, Sd_X


def _destandardize(
    beta_tilde: np.ndarray, mu_y: float, Mu_X: np.ndarray,
    sd_y: float, Sd_X: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Map a standardised-scale coefficient vector back to the original scale."""
    beta_hat = sd_y * (beta_tilde / Sd_X)
    intercept = mu_y - float(Mu_X @ beta_hat)
    return beta_hat, intercept


def l2_relax(
    y_pre: np.ndarray, X_pre: np.ndarray, tau: float, standardize: bool = True,
) -> Tuple[np.ndarray, float]:
    """Solve the L2-relaxation primal for coefficients and intercept.

    The primal objective ``||beta||^2 / 2`` is strictly convex, so the solution
    is unique (Shi & Wang 2024, Lemma 2). It is solved natively with polished
    OSQP (:func:`mlsynth.utils.pda_helpers.l2.batch.l2_relax_solve`) -- the same
    solver the multiple-treated batch uses -- which matches an interior-point
    (cvxpy/CLARABEL) solve on the unique optimum without cvxpy's per-call
    canonicalisation overhead.
    """
    T1 = X_pre.shape[0]
    mu_y, Mu_X, sd_y, Sd_X = _standardize(y_pre, X_pre, standardize)
    yt = (y_pre - mu_y) / sd_y
    Xt = (X_pre - Mu_X) / Sd_X
    Sigma = (Xt.T @ Xt) / T1
    eta = (Xt.T @ yt) / T1
    beta_tilde = l2_relax_solve(Sigma, eta, tau)
    if not np.all(np.isfinite(beta_tilde)):
        raise MlsynthEstimationError("L2-relaxation failed: OSQP did not converge.")
    return _destandardize(beta_tilde, mu_y, Mu_X, sd_y, Sd_X)


def cross_validate_tau(
    y_pre: np.ndarray, X_pre: np.ndarray, val_frac: float = 0.2,
    n_coarse: int = 40, n_fine: int = 40, standardize: bool = True,
    tau_grid: Optional[np.ndarray] = None,
) -> float:
    """Validate ``tau`` by **sequential** out-of-sample validation on the tail.

    The training window is split in time order -- fit on the earlier
    ``1 - val_frac`` fraction, validate on the most-recent ``val_frac`` tail --
    so no future period informs an earlier counterfactual (unlike a K-fold split
    over contiguous blocks, which trains on both past and future of each fold).
    ``tau_grid`` overrides the automatic log-spaced grid (e.g. to match a
    reference grid); otherwise the grid is log-spaced up to ``max|eta|`` (the
    optimal ``tau`` is often a tiny fraction of that).
    """
    T1 = y_pre.shape[0]
    n_val = max(2, int(round(val_frac * T1)))
    yt, Xt = y_pre[:-n_val], X_pre[:-n_val]
    yv, Xv = y_pre[-n_val:], X_pre[-n_val:]

    # Standardise once on the training tail; the whole grid shares the same
    # Sigma / eta, so one KKT factorization serves every candidate tau.
    Tt = Xt.shape[0]
    mu_y, Mu_X, sd_y, Sd_X = _standardize(yt, Xt, standardize)
    yts = (yt - mu_y) / sd_y
    Xts = (Xt - Mu_X) / Sd_X
    Sigma = (Xts.T @ Xts) / Tt
    eta = (Xts.T @ yts) / Tt

    def grid_mse(grid: np.ndarray) -> np.ndarray:
        betas_tilde = l2_relax_grid(Sigma, eta, grid)        # (K, N), standardised
        out = np.full(grid.shape[0], np.inf)
        for i, beta_tilde in enumerate(betas_tilde):
            if not np.all(np.isfinite(beta_tilde)):
                continue
            b, a = _destandardize(beta_tilde, mu_y, Mu_X, sd_y, Sd_X)
            out[i] = float(np.mean((yv - (Xv @ b + a)) ** 2))
        return out

    if tau_grid is not None:
        grid = np.asarray(tau_grid, dtype=float)
        return float(grid[int(np.argmin(grid_mse(grid)))])

    coarse = _auto_tau_grid(yt, Xt, standardize, n_coarse)
    k = int(np.argmin(grid_mse(coarse)))
    fine = np.linspace(coarse[max(k - 1, 0)], coarse[min(k + 1, n_coarse - 1)], n_fine)
    return float(fine[int(np.argmin(grid_mse(fine)))])


def _auto_tau_grid(y: np.ndarray, X: np.ndarray, standardize: bool, n: int) -> np.ndarray:
    """Log-spaced grid up to ``max|eta|`` on the (standardised) moments."""
    _, _, sd_y, Sd_X = _standardize(y, X, standardize)
    eta = ((X - X.mean(0)) / Sd_X).T @ ((y - y.mean()) / sd_y) / X.shape[0]
    tau_max = float(np.max(np.abs(eta)))
    return np.logspace(np.log10(max(tau_max * 1e-4, 1e-12)), np.log10(tau_max), n)


def fit_l2(
    y: np.ndarray, X: np.ndarray, T0: int, tau: Optional[float] = None,
    standardize: bool = True, tau_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Fit L2-relaxation; return ``(beta, intercept, counterfactual, tau)``."""
    y_pre, X_pre = y[:T0], X[:T0]
    if tau is None:
        tau_used = cross_validate_tau(
            y_pre, X_pre, standardize=standardize, tau_grid=tau_grid)
    else:
        tau_used = float(tau)
    beta, intercept = l2_relax(y_pre, X_pre, tau_used, standardize=standardize)
    counterfactual = X @ beta + intercept
    return beta, intercept, counterfactual, tau_used
