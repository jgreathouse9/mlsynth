"""Two-Way Synthetic Forecasting (Shen, arXiv:2606.18512v2), readable port.

Section references are to the v2 source: ``content/algorithm.tex`` for the
one-step estimator and ``content/horizon.tex`` for the direct and recursive
multi-step extensions; sigma-hat is eq. (var.sigma.pool) in ``results.tex``.

Notation follows the paper. The target unit is N, donors are I_1, the panel is
control for t <= T_0 and the donors are treated for T_0 < t <= T while the
target stays under control. The estimand is the target's *treated* potential
outcome after T -- a forecast past the end of the panel, not an imputation
inside it.
"""
from __future__ import annotations

import numpy as np


def hsvt_factors(A: np.ndarray, k: int):
    """Rank-k SVD factors of A, for HSVT and its exact pseudo-inverse.

    ``HSVT(A, k)`` has rank exactly k by construction, so its pseudo-inverse
    must be built by inverting those k singular values -- not by calling
    ``np.linalg.pinv``, whose default tolerance keeps numerically-zero
    directions of the truncated matrix and inflates the result without bound.
    At n = 150 that turned a handful of plug-in standard errors into ~1e8.
    """
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    k = int(min(k, sv.size))
    return U[:, :k], sv[:k], Vt[:k]


def hsvt(A: np.ndarray, k: int) -> np.ndarray:
    """Hard singular value thresholding, algorithm.tex 'Spectral denoising'."""
    U, sv, Vt = hsvt_factors(A, k)
    return (U * sv) @ Vt


def _pinv_from(U, sv, Vt):
    """Pseudo-inverse of ``(U * sv) @ Vt``, exactly at its construction rank."""
    return (Vt.T / sv) @ U.T


def page_matrices(Y_post_donors: np.ndarray, L: int):
    """Return ``(Z_lag, z_next, W)`` from eqs. (page.train), (Zlag.znext), (W.block).

    ``Y_post_donors`` is ``N_1 x T_1``: the donors' post-treatment (treated)
    outcomes. Each donor contributes B-1 non-overlapping blocks of length L+1;
    within a block the first L entries are the lag vector and the last is its
    one-step response. The final L+1 post-treatment observations are excluded
    from the Page matrix and the last L of them form the forecast state W.
    """
    N1, T1 = Y_post_donors.shape
    B = T1 // (L + 1)
    if B < 2:
        raise ValueError(f"need at least 2 Page blocks; T1={T1}, L={L} gives B={B}")
    blocks = []
    for j in range(N1):
        for b in range(B - 1):                      # the first B-1 blocks
            seg = Y_post_donors[j, b * (L + 1):(b + 1) * (L + 1)]
            blocks.append(seg)
    P = np.stack(blocks, axis=1)                    # (L+1) x M, M = (B-1) N_1
    Z_lag, z_next = P[:L], P[L]
    W = Y_post_donors[:, T1 - L:]                   # N_1 x L, the terminal state
    return Z_lag, z_next, W


def companion(x: np.ndarray) -> np.ndarray:
    """Pi(x) of eq. (pi.matrix): shift rows, with x' as the last row."""
    L = x.size
    P = np.zeros((L, L))
    P[:L - 1, 1:] = np.eye(L - 1)
    P[L - 1] = x
    return P


def g_ell(x: np.ndarray, ell: int) -> np.ndarray:
    """g_ell(x) = (Pi(x)^ell)' e_L -- the initial L-lag state's lead-ell map."""
    L = x.size
    e = np.zeros(L); e[-1] = 1.0
    return np.linalg.matrix_power(companion(x), ell).T @ e


def J_ell(x: np.ndarray, ell: int) -> np.ndarray:
    """Jacobian of g_ell, by the paper's representation in horizon.tex."""
    L = x.size
    e = np.zeros(L); e[-1] = 1.0
    Pi = companion(x)
    out = np.zeros((L, L))
    for a in range(ell):
        scalar = float(e @ np.linalg.matrix_power(Pi, a) @ e)
        out += scalar * np.linalg.matrix_power(Pi, ell - 1 - a).T
    return out


def fit_weights(y_pre_target, Y_pre_donors, Z_lag, z_next, k_y, k_z):
    """Steps (a)-(c): HSVT then PCR on each side."""
    Uy, sy, Vy = hsvt_factors(Y_pre_donors, k_y)    # N_1 x T_0
    Uz, sz, Vz = hsvt_factors(Z_lag, k_z)           # L x M
    Yk, Zk = (Uy * sy) @ Vy, (Uz * sz) @ Vz
    beta = _pinv_from(Uy, sy, Vy).T @ y_pre_target  # eq. (pcr.beta.hat)
    alpha = _pinv_from(Uz, sz, Vz).T @ z_next       # eq. (pcr.alpha.hat)
    return beta, alpha, Yk, Zk


def sigma2_hat(y_pre_target, Y_pre_donors, beta, Z_lag, z_next, alpha, k_y, k_z):
    """Pooled unit- and time-side PCR residual variance, eq. (var.sigma.pool).

    The residuals use the raw design matrices, not the de-noised ones.
    """
    T0 = y_pre_target.size
    M = z_next.size
    num = (np.sum((y_pre_target - Y_pre_donors.T @ beta) ** 2)
           + np.sum((z_next - Z_lag.T @ alpha) ** 2))
    den = (T0 - k_y) + (M - k_z)
    return float(num / den)


def recursive_twsf(y_pre_target, Y_pre_donors, Y_post_donors, L, k_y, k_z, h):
    """Recursive TWSF for eta = h^{-1} 1 over leads 1..h (horizon.tex sec. 5.2).

    Returns the point estimate, the plug-in standard deviation V_hat^rec, and
    the pieces the diagnostics need.
    """
    Z_lag, z_next, W = page_matrices(Y_post_donors, L)
    beta, alpha, Yk, Zk = fit_weights(y_pre_target, Y_pre_donors, Z_lag, z_next, k_y, k_z)

    eta = np.full(h, 1.0 / h)                       # the paper's eta = h^{-1} 1
    alpha_rec = sum(eta[i] * g_ell(alpha, i + 1) for i in range(h))
    J_eta = sum(eta[i] * J_ell(alpha, i + 1) for i in range(h))

    state = W.T @ beta                              # imputed treated L-lag state
    theta = float(alpha_rec @ state)

    s2 = sigma2_hat(y_pre_target, Y_pre_donors, beta, Z_lag, z_next, alpha, k_y, k_z)
    q_beta = np.linalg.pinv(Yk) @ (W @ alpha_rec)
    q_alpha = np.linalg.pinv(Zk) @ (J_eta.T @ state)
    V2 = s2 * (np.sum(alpha_rec ** 2) * np.sum(beta ** 2)
               + np.sum(q_beta ** 2) * (1 + np.sum(beta ** 2))
               + np.sum(q_alpha ** 2) * (1 + np.sum(alpha ** 2)))
    return {"theta": theta, "V": float(np.sqrt(max(V2, 0.0))), "sigma2": s2,
            "beta": beta, "alpha": alpha, "Z_lag": Z_lag, "Y_pre_donors": Y_pre_donors,
            "W": W, "alpha_rec": alpha_rec}


def fit_once(y_pre_target, Y_pre_donors, Y_post_donors, L, k_y, k_z):
    """Fit the pieces that do not depend on the horizon.

    The recursive estimator learns one temporal rule and iterates it, so every
    horizon shares ``beta``, ``alpha``, ``sigma2`` and both pseudo-inverses.
    Computing them once is what makes the full 1000-replication grid tractable.
    """
    Z_lag, z_next, W = page_matrices(Y_post_donors, L)
    Uy, sy, Vy = hsvt_factors(Y_pre_donors, k_y)
    Uz, sz, Vz = hsvt_factors(Z_lag, k_z)
    Yk_pinv, Zk_pinv = _pinv_from(Uy, sy, Vy), _pinv_from(Uz, sz, Vz)
    beta = Yk_pinv.T @ y_pre_target
    alpha = Zk_pinv.T @ z_next
    s2 = sigma2_hat(y_pre_target, Y_pre_donors, beta, Z_lag, z_next, alpha, k_y, k_z)
    return {"beta": beta, "alpha": alpha, "sigma2": s2, "W": W,
            "Yk_pinv": Yk_pinv, "Zk_pinv": Zk_pinv, "state": W.T @ beta}


def eval_horizon(fit: dict, h: int):
    """Recursive point estimate and plug-in SD at horizon ``h``, from ``fit_once``."""
    alpha, beta, W = fit["alpha"], fit["beta"], fit["W"]
    eta = np.full(h, 1.0 / h)
    alpha_rec = sum(eta[i] * g_ell(alpha, i + 1) for i in range(h))
    J_eta = sum(eta[i] * J_ell(alpha, i + 1) for i in range(h))
    state = fit["state"]
    theta = float(alpha_rec @ state)
    q_beta = fit["Yk_pinv"] @ (W @ alpha_rec)
    q_alpha = fit["Zk_pinv"] @ (J_eta.T @ state)
    V2 = fit["sigma2"] * (np.sum(alpha_rec ** 2) * np.sum(beta ** 2)
                          + np.sum(q_beta ** 2) * (1 + np.sum(beta ** 2))
                          + np.sum(q_alpha ** 2) * (1 + np.sum(alpha ** 2)))
    return theta, float(np.sqrt(max(V2, 0.0)))
