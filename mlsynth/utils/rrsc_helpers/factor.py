"""Factor-analysis kernels for RRSC (He, Li, Shi & Miao 2026).

Two factor routines back the two asymptotic regimes:

* :func:`fa_em` -- the Bai & Li (2012) EM quasi-maximum-likelihood factor
  analysis used in the large-N regime (reference: the authors'
  ``LargeN/factor_functions.R`` ``fa.em``, itself adapted from Wang et al.
  2017). Transcribed to match the reference's exact iteration so its
  early-stop trajectory is reproduced value-for-value.
* :func:`factanal_mle` -- an MLE factor analysis on the correlation matrix,
  matching R's ``stats::factanal`` (``rotation="none"``), used in the fixed-N
  regime.

:func:`select_nfactors` implements the Bai & Ng (2002) information criteria the
authors use to choose the number of factors.
"""
from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError

_EPS = np.finfo(float).eps


def _eig_desc(M: np.ndarray):
    """Symmetric eigendecomposition with eigenvalues in DESCENDING order
    (matching R's ``eigen``)."""
    w, V = np.linalg.eigh((M + M.T) / 2)
    return w[::-1], V[:, ::-1]


def fa_pc(Y: np.ndarray, r: int):
    """Principal-component factor analysis (initialiser for :func:`fa_em`).

    Parameters
    ----------
    Y : np.ndarray
        ``(n_obs x p)`` data (samples in rows, variables in columns).
    r : int
        Number of factors.

    Returns
    -------
    (Gamma, Sigma) : (np.ndarray, np.ndarray)
        Loadings ``(p x r)`` and residual variances ``(p,)``.
    """
    n, p = Y.shape
    if n <= p:
        S = (Y @ Y.T) / n
        vals, vecs = _eig_desc(S)
        U = vecs[:, :r]
        d = np.sqrt(np.maximum(0.0, n * vals[:r]))
        Z = np.sqrt(n) * U
        Gamma = (Y.T @ U) / np.sqrt(n)
    else:
        S2 = (Y.T @ Y) / n
        vals, vecs = _eig_desc(S2)
        V = vecs[:, :r]
        d = np.sqrt(np.maximum(0.0, n * vals[:r]))
        Gamma = V * d / np.sqrt(n)
        nz = d > _EPS * max(d.max() if d.size else 1.0, 1.0)
        invd = np.where(nz, 1.0 / np.where(d == 0, 1.0, d), 0.0)
        Z = np.sqrt(n) * (Y @ V * invd)
    Resid = Y - Z @ Gamma.T
    Sigma = np.mean(Resid ** 2, axis=0)
    return Gamma, Sigma


def fa_em(Y: np.ndarray, r: int, tol: float = 1e-6, maxiter: int = 1000):
    """Bai & Li (2012) EM quasi-MLE factor analysis on ``(n_obs x p)`` data.

    Faithful transcription of the reference ``fa.em`` including its exact
    per-iteration algebra (the non-standard ``YEZ (VE diag(1/w)) VE'`` loading
    update), so the reference's default ``tol=1e-6`` early-stop is reproduced.

    Returns ``(Gamma, Sigma)`` -- loadings ``(p x r)`` and residual variances.
    """
    if Y.ndim != 2:
        raise MlsynthDataError("fa_em: Y must be a 2D array (obs x variables).")
    n, p = Y.shape
    if not (1 <= r < p):
        raise MlsynthConfigError(f"fa_em: require 1 <= r < p; got r={r}, p={p}.")
    try:
        Gamma, Sig0 = fa_pc(Y, r)
        invSigma = 1.0 / np.maximum(Sig0, _EPS)
        sample_var = np.mean(Y ** 2, axis=0)
        tG = np.sqrt(invSigma)[:, None] * Gamma
        wM, VM = _eig_desc(np.eye(r) + tG.T @ tG)
        YSG = Y @ (invSigma[:, None] * Gamma)
        llh = -np.inf
        for _ in range(maxiter):
            inv_wM = 1.0 / np.maximum(wM, _EPS)
            varZ = VM @ (inv_wM[:, None] * VM.T)
            EZ = YSG @ varZ
            EZZ = n * varZ + EZ.T @ EZ
            wE, VE = _eig_desc(EZZ)
            YEZ = Y.T @ EZ
            G = np.sqrt(np.maximum(wE, _EPS))[:, None] * VE.T
            GG = Gamma @ G.T
            denom = (sample_var - 2.0 / n * np.sum(YEZ * Gamma, axis=1)
                     + np.sum(GG ** 2, axis=1) / n)
            invSigma = 1.0 / np.maximum(denom, _EPS)
            Gamma = YEZ @ ((1.0 / np.maximum(wE, _EPS))[:, None] * VE) @ VE.T
            siv = np.minimum(np.sqrt(invSigma), 1.0 / np.sqrt(_EPS))
            tG = siv[:, None] * Gamma
            wM, VM = _eig_desc(tG.T @ tG + np.eye(r))
            YSG = Y @ (invSigma[:, None] * Gamma)
            old = llh
            logdetY = -np.sum(np.log(invSigma)) + np.sum(np.log(np.maximum(wM, _EPS)))
            B = (1.0 / np.sqrt(np.maximum(wM, _EPS)))[:, None] * (VM.T @ YSG.T)
            llh = -logdetY - (np.sum(invSigma * sample_var) - np.sum(B ** 2) / n)
            if abs(llh - old) < tol * abs(llh):
                break
    except np.linalg.LinAlgError as exc:                       # pragma: no cover
        raise MlsynthEstimationError(f"fa_em: factor analysis failed ({exc}).") from exc
    return Gamma, 1.0 / invSigma


def factanal_mle(Xobs: np.ndarray, q: int, maxiter: int = 1000):
    """MLE factor analysis on the correlation matrix (matches R
    ``stats::factanal``, ``rotation="none"``).

    Parameters
    ----------
    Xobs : np.ndarray
        ``(n_obs x p)`` data.
    q : int
        Number of factors.

    Returns
    -------
    (loadings, uniquenesses) : (np.ndarray, np.ndarray)
        Loadings ``(p x q)`` on the covariance scale and uniquenesses ``(p,)``.
    """
    from scipy.optimize import minimize

    if Xobs.ndim != 2:
        raise MlsynthDataError("factanal_mle: Xobs must be a 2D array (obs x variables).")
    p = Xobs.shape[1]
    if not (1 <= q < p):
        raise MlsynthConfigError(f"factanal_mle: require 1 <= q < p; got q={q}, p={p}.")
    S = np.corrcoef(Xobs, rowvar=False)
    if not np.all(np.isfinite(S)):
        raise MlsynthDataError(
            "factanal_mle: correlation matrix is not finite (a series may be constant)."
        )

    def _eig(Psi):
        sc = 1.0 / np.sqrt(Psi)
        Sstar = (sc[:, None] * S) * sc[None, :]
        val, vec = _eig_desc(Sstar)
        return val, vec

    def FAfn(Psi):
        val, _ = _eig(Psi)
        e = val[q:]
        return -(np.sum(np.log(e) - e) - q + p)

    def FAgr(Psi):
        val, vec = _eig(Psi)
        load = np.sqrt(Psi)[:, None] * (vec[:, :q] * np.sqrt(np.maximum(val[:q] - 1, 0)))
        return np.diag(load @ load.T + np.diag(Psi) - S) / Psi ** 2

    try:
        start = (1 - 0.5 * q / p) / np.diag(np.linalg.inv(S))
        res = minimize(FAfn, start, jac=FAgr, method="L-BFGS-B",
                       bounds=[(0.005, 1.0)] * p,
                       options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-10))
        Psi = res.x
        val, vec = _eig(Psi)
        load = vec[:, :q] * np.sqrt(np.maximum(val[:q] - 1, 0))
    except np.linalg.LinAlgError as exc:                       # pragma: no cover
        raise MlsynthEstimationError(f"factanal_mle: optimisation failed ({exc}).") from exc
    return np.sqrt(Psi)[:, None] * load, Psi


def select_nfactors(Y: np.ndarray, k_max: int = 15) -> int:
    """Number of factors by the Bai & Ng (2002) IC_p2 criterion on the
    pre-period panel ``Y`` ``(N units x T0 periods)``.

    Uses the PC residual variance ``V(k)`` and returns the ``k`` minimising
    ``ln V(k) + k (N+T)/(NT) ln(min(N,T))``.
    """
    N, T0 = Y.shape
    k_max = int(min(k_max, N - 1, T0 - 1))
    if k_max < 1:
        raise MlsynthConfigError("select_nfactors: panel too small to select factors.")
    C = min(N, T0)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    best_k, best_ic = 1, np.inf
    for k in range(1, k_max + 1):
        Gamma, Sigma = fa_pc(Yc.T, k)              # obs x var == T0 x N
        V_k = float(np.mean(Sigma))
        ic = np.log(V_k) + k * ((N + T0) / (N * T0)) * np.log(C)
        if ic < best_ic:
            best_ic, best_k = ic, k
    return best_k
