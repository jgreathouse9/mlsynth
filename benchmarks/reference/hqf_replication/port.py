"""Line-for-line Python port of the authors' RPCA-HQF MATLAB code.

Source: https://github.com/bestzywang/Robust-PCA-via-non-convex-half-quadratic-regularization
@ ca4d23a (files RPCA_HQF.m, Gaussian_noise.m, main.m; no licence file, so the
code is not vendored here). Paper: Wang, Li, So & Liu (2023), "Robust PCA via
non-convex half-quadratic regularization", Signal Processing 204, 108816.

`rpca_hqf` follows RPCA_HQF.m statement by statement, keeping two MATLAB
behaviours a reader could take for typos, because they decide the numbers:

* `inv(V*V' - lam_1)` and `inv(U'*U - lam_2)` subtract the scalar from every
  entry of the r x r matrix (MATLAB matrix-minus-scalar), not lam * I, and the
  proximal terms enter with a minus sign (`- lam_1*U_p`). Paper eq. (16)-(17)
  print the same signs with I; the proximal step that minimises (14)-(15) has
  plus signs and lam * I.
* `pinv(A, 1e-8)` uses an absolute singular-value tolerance; numpy's `rcond`
  is relative, so `_pinv_abs` reimplements MATLAB's.

The `variant` switches turn each of mlsynth's deliberate changes on one at a
time (see run_tableone.py), so the gap between the authors' numbers and
mlsynth's can be attributed change by change.
"""
from __future__ import annotations

import numpy as np

MAD_CONST = 1.4815


def _pinv_abs(A: np.ndarray, tol: float) -> np.ndarray:
    """MATLAB pinv(A, tol): singular values <= tol (absolute) treated as zero."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > tol
    return (Vt[keep].T / s[keep]) @ U[:, keep].T


def gaussian_noise_gm(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian_noise.m, model 'GM': 10% of entries with 10x the standard deviation."""
    signal_power = np.sum(np.abs(signal) ** 2) / signal.size
    s_n = 10 ** (snr_db / 10)
    c_1, c_2 = 0.9, 0.1
    sigma_v_2 = signal_power / s_n
    sigma_1 = np.sqrt(sigma_v_2 / (c_1 + 10 ** 2 * c_2))
    flag = rng.binomial(1, c_1, signal.shape)
    return (10 * sigma_1 * rng.standard_normal(signal.shape) * (1 - flag)
            + sigma_1 * rng.standard_normal(signal.shape) * flag)


def rpca_hqf(M_noise: np.ndarray, rak: int, maxiter: int, ip: float, lam_1: float,
             U0: np.ndarray, *, proper_prox: bool = False, monotone: bool = True,
             start10: bool = True, reset_stall: bool = False):
    """RPCA_HQF.m. Returns (X, iterations). Defaults reproduce the authors' code.

    proper_prox  -- (V V' + lam I)^-1 and + lam U_p, as mlsynth's hqf.py does
    monotone     -- scale = min(previous, ip * MAD); False recomputes each time
    start10      -- initial scale 10 * MAD; False starts at ip * MAD
    reset_stall  -- reset the stagnation counter on an improving step
    """
    m, n = M_noise.shape
    U = U0.copy()
    lam_2 = lam_1
    for _ in range(3):                                   # Ip = 3, PF initialisation
        V = _pinv_abs(U, 1e-8) @ M_noise
        U = M_noise @ _pinv_abs(V, 1e-8)
    X = U @ V
    T = M_noise - X
    t = T.ravel()
    scale = (10.0 if start10 else ip) * MAD_CONST * np.median(np.abs(t - np.median(t)))
    N = T * ~(np.abs(T - np.median(t)) - scale < 0)
    U_p, V_p = U, V
    rmse = []
    stall = 0
    it = 0
    I = np.eye(rak)
    for it in range(1, maxiter + 1):
        D = M_noise - N
        if proper_prox:
            U = (D @ V.T + lam_1 * U_p) @ np.linalg.inv(V @ V.T + lam_1 * I)
            V = np.linalg.inv(U.T @ U + lam_2 * I) @ (U.T @ D + lam_2 * V_p)
        else:
            U = (D @ V.T - lam_1 * U_p) @ np.linalg.inv(V @ V.T - lam_1)
            V = np.linalg.inv(U.T @ U - lam_2) @ (U.T @ D - lam_2 * V_p)
        U_p, V_p = U, V
        X = U @ V
        T = M_noise - X
        t = T.ravel()
        new = ip * MAD_CONST * np.median(np.abs(t - np.median(t)))
        scale = min(scale, new) if monotone else new
        N = T * ~(np.abs(T - np.median(t)) - scale < 0)
        rmse.append(np.linalg.norm(M_noise - X) / np.sqrt(m * n))
        if it != 1:
            if rmse[-2] - rmse[-1] < 1e-6:
                stall += 1
            elif reset_stall:
                stall = 0
            if stall > 1:
                break
    return X, it
