"""Core MC-NNM engine: soft-impute with unregularised fixed effects.

Athey, S., Bayati, M., Doudchenko, N., Imbens, G. & Khosravi, K. (2021).
*"Matrix Completion Methods for Causal Panel Data Models."* Journal of
the American Statistical Association 116(536):1716-1730.

The model is :math:`Y = L^* + \\Gamma 1_T^\\top + 1_N \\Delta^\\top +
\\varepsilon`, and the estimator (paper eq. 4.3)

.. math::

   (\\widehat L, \\widehat\\Gamma, \\widehat\\Delta)
     = \\arg\\min_{L, \\Gamma, \\Delta}
       \\frac{1}{|\\mathcal{O}|}
       \\bigl\\| P_\\mathcal{O}(Y - L - \\Gamma 1_T^\\top
                               - 1_N \\Delta^\\top) \\bigr\\|_F^2
       + \\lambda \\|L\\|_*,

regularises only the low-rank part :math:`L` (the unit/time fixed
effects :math:`\\Gamma, \\Delta` are estimated explicitly, unregularised,
to reduce bias). It is solved by the SOFT-IMPUTE iteration (eq. 4.4-4.5):
soft-threshold the singular values of the filled-in matrix, then re-fit
the fixed effects, until convergence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_EPS = 1e-12


def _shrink(A: np.ndarray, thr: float) -> np.ndarray:
    """Singular-value soft-thresholding shrink operator (paper eq. 4.4)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_shrunk = np.maximum(s - thr, 0.0)
    return (U * s_shrunk) @ Vt


def _two_way_fe(
    R: np.ndarray, mask: np.ndarray, est_u: bool, est_v: bool,
    *, max_iter: int = 100, tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit (Gamma) and time (Delta) effects fitting residual ``R`` over the
    observed entries (``mask == 1``), via alternating means (paper: FE from
    the first-order conditions of the squared-error term)."""
    N, T = R.shape
    gamma = np.zeros(N)
    delta = np.zeros(T)
    if not est_u and not est_v:
        return gamma, delta
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    for _ in range(max_iter):
        g_prev, d_prev = gamma.copy(), delta.copy()
        if est_u:
            num = (mask * (R - delta[None, :])).sum(axis=1)
            gamma = np.where(row_counts > 0, num / np.maximum(row_counts, 1), 0.0)
        if est_v:
            num = (mask * (R - gamma[:, None])).sum(axis=0)
            delta = np.where(col_counts > 0, num / np.maximum(col_counts, 1), 0.0)
        if (np.max(np.abs(gamma - g_prev)) < tol
                and np.max(np.abs(delta - d_prev)) < tol):
            break
    return gamma, delta


def mcnnm_fit(
    Y: np.ndarray,
    mask: np.ndarray,
    thr: float,
    *,
    est_u: bool = True,
    est_v: bool = True,
    max_iter: int = 400,
    tol: float = 1e-5,
) -> dict:
    """Fit MC-NNM for a given SVD threshold ``thr``.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix, shape ``(N, T)``. Missing entries may hold any
        value; only observed (``mask == 1``) entries are used.
    mask : np.ndarray
        Observation indicator, shape ``(N, T)``; ``1`` observed, ``0``
        missing (to be imputed).
    thr : float
        Singular-value soft-threshold (the regularisation strength).
    est_u, est_v : bool
        Estimate unit / time fixed effects.

    Returns
    -------
    dict with ``L`` (low-rank matrix), ``gamma`` (N,), ``delta`` (T,),
    and ``completed`` = ``L + gamma + delta`` (the full fitted matrix).
    """
    N, T = Y.shape
    L = np.zeros((N, T))
    gamma = np.zeros(N)
    delta = np.zeros(T)
    Yobs = mask * Y

    for _ in range(max_iter):
        L_prev = L
        # Fixed-effects step on the residual Y - L over observed cells.
        gamma, delta = _two_way_fe(Y - L, mask, est_u, est_v)
        fe = gamma[:, None] + delta[None, :]
        # Soft-impute step: P_O(Y - fe) + P_O^perp(L), then shrink.
        filled = mask * (Y - fe) + (1.0 - mask) * L
        L = _shrink(filled, thr)
        denom = np.linalg.norm(L_prev) + _EPS
        if np.linalg.norm(L - L_prev) / denom < tol:
            break

    completed = L + gamma[:, None] + delta[None, :]
    return {"L": L, "gamma": gamma, "delta": delta, "completed": completed}


def _lambda_grid(Y: np.ndarray, mask: np.ndarray, n_lam: int) -> np.ndarray:
    """Geometric grid of SVD thresholds from ~spectral-norm down to a small
    fraction (Mazumder et al. warm-start grid)."""
    Yobs = mask * Y
    # Demean by observed grand mean for a sensible spectral scale.
    if mask.sum() > 0:
        Yobs = mask * (Y - (Yobs.sum() / mask.sum()))
    smax = float(np.linalg.svd(Yobs, compute_uv=False)[0])
    smax = max(smax, _EPS)
    return np.geomspace(smax, smax * 1e-3, n_lam)


def mcnnm_cv(
    Y: np.ndarray,
    mask: np.ndarray,
    *,
    est_u: bool = True,
    est_v: bool = True,
    n_lam: int = 40,
    n_folds: int = 5,
    max_iter: int = 400,
    tol: float = 1e-5,
    random_state: int = 0,
) -> dict:
    """Select the threshold by K-fold cross-validation over observed cells.

    For each candidate threshold, a fraction of observed entries is held
    out, the model is fit on the rest, and out-of-sample squared error is
    averaged over folds; the threshold minimising it is chosen, then the
    final fit uses all observed entries (paper "Cross-validation").
    """
    rng = np.random.default_rng(random_state)
    grid = _lambda_grid(Y, mask, n_lam)
    obs_idx = np.argwhere(mask > 0)
    n_obs = obs_idx.shape[0]

    cv_err = np.zeros(len(grid))
    fold_assign = rng.integers(0, n_folds, size=n_obs)
    for f in range(n_folds):
        test_sel = fold_assign == f
        train_mask = mask.copy()
        test_cells = obs_idx[test_sel]
        train_mask[test_cells[:, 0], test_cells[:, 1]] = 0
        if train_mask.sum() == 0:
            continue
        for li, thr in enumerate(grid):
            fit = mcnnm_fit(Y, train_mask, thr, est_u=est_u, est_v=est_v,
                            max_iter=max_iter, tol=tol)
            pred = fit["completed"]
            err = (Y[test_cells[:, 0], test_cells[:, 1]]
                   - pred[test_cells[:, 0], test_cells[:, 1]])
            cv_err[li] += float(np.mean(err ** 2))

    best = int(np.argmin(cv_err))
    best_thr = float(grid[best])
    fit = mcnnm_fit(Y, mask, best_thr, est_u=est_u, est_v=est_v,
                    max_iter=max_iter, tol=tol)
    fit["best_lambda"] = best_thr
    fit["lambda_grid"] = grid
    fit["cv_error"] = cv_err / max(n_folds, 1)
    return fit
