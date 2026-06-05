"""MCMC sampler for the spatial-spillover Bayesian SCM (Sakaguchi & Tagawa 2026).

Pure-NumPy port of the authors' two-step Gibbs/Metropolis sampler
(reference: ``src/20_mcmc.cpp`` of the replication package), validated against
the compiled C++ on the paper's simulation and Sudan application.

Two steps:

* :func:`hs_alpha_gibbs` -- Bayesian horseshoe regression for the synthetic
  weights ``alpha`` on the pre-treatment outcomes
  (``Y0_pre = Yc_pre @ alpha + eps``), with the Makalic-Schmidt (2015)
  auxiliary-variable representation so every full conditional is closed form.
* :func:`sar_full_sampler` -- the spatial-autoregressive panel model on the
  control outcomes, conditional on ``alpha_hat``. Samples the spatial
  parameter ``rho`` by random-walk Metropolis (with the ``log|I - rho A|``
  Jacobian evaluated in O(N) from the eigenvalues of ``A = W + w alpha'``),
  the innovation variance ``sigma2`` (inverse-gamma), an optional covariate
  coefficient ``beta`` (ridge), and an optional AR(1) latent-factor block
  (forward-filter/backward-sample).

The treatment and spillover effects are then read off the identification
formulae (Theorems 1-2 of the paper) via :func:`treated_counterfactual` and
:func:`spillover_effects`, which depend only on ``(alpha, rho)``.
"""
from __future__ import annotations

import numpy as np


# ----------------------------- utilities ---------------------------------
def row_normalize(W: np.ndarray) -> np.ndarray:
    """Zero the diagonal and scale each row of ``W`` to sum to one."""
    W = np.array(W, dtype=float, copy=True)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1)
    rs[rs == 0] = 1.0
    return W / rs[:, None]


def _ig(rng, a, b):
    """Inverse-gamma(shape ``a``, scale ``b``) draw (vectorised)."""
    return 1.0 / rng.gamma(a, 1.0 / np.maximum(b, 1e-300))


def _chol_psd(A, jitter=1e-10):
    """Cholesky of a symmetric PSD matrix with escalating jitter fallback."""
    A = 0.5 * (A + A.T)
    for k in range(7):
        try:
            return np.linalg.cholesky(A + (jitter * 10 ** k) * np.eye(A.shape[0]))
        except np.linalg.LinAlgError:
            continue
    d, V = np.linalg.eigh(A)
    d = np.clip(d, jitter, None)
    return np.linalg.cholesky(V @ np.diag(d) @ V.T)


# ----------------------- Step 1: horseshoe alpha -------------------------
def hs_alpha_gibbs(rng, y, X, iters, burn):
    """Horseshoe-Gibbs draws of the synthetic weights ``alpha``.

    Parameters
    ----------
    rng : numpy.random.Generator
    y : np.ndarray
        Treated pre-treatment outcomes, shape ``(T0,)``.
    X : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    iters, burn : int
        Total iterations and burn-in. Returns ``iters - burn`` draws.

    Returns
    -------
    np.ndarray
        Post-burn ``alpha`` draws, shape ``(iters - burn, N)`` (rescaled to the
        original outcome units).
    """
    T0, N = X.shape
    sx = np.maximum(X.std(0, ddof=1), 1e-8)
    sy = max(y.std(ddof=1), 1e-8)
    Xs = X / sx
    ys = y / sy
    XtX = Xs.T @ Xs
    Xty = Xs.T @ ys
    s2i = np.ones(N); nus_i = np.ones(N); tau2 = 1.0; nu_tau = 1.0
    s2 = max(ys.var(ddof=1), 1.0); nu_s = 1.0
    out = np.empty((iters - burn, N))
    for it in range(iters):
        D = XtX + np.diag(s2 / np.maximum(s2i, 1e-12))
        Dinv = np.linalg.inv(D); Dinv = 0.5 * (Dinv + Dinv.T)
        L = np.linalg.cholesky(max(s2, 1e-12) * Dinv)
        alpha = Dinv @ Xty + L @ rng.standard_normal(N)
        s2i = _ig(rng, 1.0, 0.5 * alpha ** 2 + 1.0 / np.maximum(nus_i, 1e-12))
        nus_i = _ig(rng, 1.0, 1.0 / np.maximum(s2i, 1e-12) + 1.0 / max(tau2, 1e-12))
        tau2 = float(_ig(rng, 0.5 * (N + 1.0),
                         np.sum(1.0 / np.maximum(nus_i, 1e-12)) + 1.0 / max(nu_tau, 1e-12)))
        nu_tau = float(_ig(rng, 1.0, 1.0 / max(tau2, 1e-12) + 1.0 / max(s2, 1e-12)))
        sse = float(((ys - Xs @ alpha) ** 2).sum())
        s2 = float(_ig(rng, 1.0 + 0.5 * T0,
                       1.0 / max(nu_tau, 1e-12) + 1.0 / max(nu_s, 1e-12) + 0.5 * sse))
        nu_s = float(_ig(rng, 1.0, 1.0 / max(s2, 1e-12) + 1.0 / 100.0))
        if it >= burn:
            out[it - burn] = (sy / sx) * alpha
    return out


# ----------------------- Step 2: SAR sampler -----------------------------
def sar_full_sampler(rng, Yc_pre, alpha_hat, w, Wn, iters, burn,
                     X=None, p=0, step_rho=0.02, a0=1.0, b0=1.0):
    """Sample ``rho`` (+ ``sigma2``, ``beta``, AR(1) factors) given ``alpha_hat``.

    Parameters
    ----------
    rng : numpy.random.Generator
    Yc_pre : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    alpha_hat : np.ndarray
        Fixed synthetic weights from Step 1, shape ``(N,)``.
    w : np.ndarray
        Spatial-weight vector linking the treated unit to controls, ``(N,)``
        (normalised to sum to one upstream).
    Wn : np.ndarray
        Row-normalised control-to-control spatial-weight matrix, ``(N, N)``.
    iters, burn : int
    X : np.ndarray, optional
        Time-varying covariate cube ``(T0, N, K)`` (raw, no standardisation).
    p : int
        Number of AR(1) latent factors (``0`` disables the factor block).
    step_rho : float
        Random-walk Metropolis step for ``rho``.
    a0, b0 : float
        Inverse-gamma prior shape/scale for ``sigma2``.

    Returns
    -------
    dict
        ``rho``/``s2`` post-burn draw arrays, ``beta`` draws ``(M, K)`` or
        ``None``, and ``acc`` (post-burn ``rho`` acceptance rate).
    """
    T0, N = Yc_pre.shape
    K = 0 if X is None else X.shape[2]
    A = Wn + np.outer(w, alpha_hat)
    evA = np.linalg.eigvals(A)
    bnd = 0.95 / max(1.0, np.max(np.abs(np.linalg.eigvals(Wn))))
    AYc = (A @ Yc_pre.T).T

    rho = 0.0; s2 = 1.0
    beta = np.zeros(K)
    Eta = np.zeros((N, p)); Gamma = np.zeros((p, T0))
    phi_g = 0.0; s2_g = 1.0; nu_s2_g = 1.0
    omega = np.ones(p); nu_omega = np.ones(p)
    s2_eta = 1.0; nu_s2_eta = 1.0

    def loglik(r):
        if abs(r) >= bnd:
            return -np.inf
        ldet = np.sum(np.log(1.0 - r * evA).real)
        U = Yc_pre - r * AYc
        if K:
            U = U - np.einsum('tnk,k->tn', X, beta)
        if p:
            U = U - (Eta @ Gamma).T
        ss = float(np.einsum('tn,tn->', U, U))
        return T0 * ldet - 0.5 * N * T0 * np.log(s2) - 0.5 * ss / s2

    out_rho = np.empty(iters - burn); out_s2 = np.empty(iters - burn)
    out_beta = np.empty((iters - burn, K)) if K else None
    acc = 0
    for it in range(iters):
        Ystar = Yc_pre - rho * AYc
        if K:
            Ystar = Ystar - np.einsum('tnk,k->tn', X, beta)

        # (1) AR(1) factors Gamma via forward-filter/backward-sample
        if p:
            Phi = phi_g * np.eye(p); Q = s2_g * np.eye(p)
            HtH = (Eta.T @ Eta) / s2; Ht = Eta.T / s2
            m_f = [None] * T0; V_f = [None] * T0
            pred_f = [None] * T0; Ppred_f = [None] * T0
            gprev = np.zeros(p); Pprev = (s2_g / max(1e-6, 1 - phi_g ** 2)) * np.eye(p)
            for t in range(T0):
                pred = Phi @ gprev; Ppred = Phi @ Pprev @ Phi.T + Q
                Pi = np.linalg.inv(Ppred)
                V = np.linalg.inv(Pi + HtH)
                m = V @ (Pi @ pred + Ht @ Ystar[t])
                m_f[t] = m; V_f[t] = V; pred_f[t] = pred; Ppred_f[t] = Ppred
                gprev = m; Pprev = V
            Gamma[:, T0 - 1] = m_f[T0 - 1] + _chol_psd(V_f[T0 - 1]) @ rng.standard_normal(p)
            for t in range(T0 - 2, -1, -1):
                J = V_f[t] @ Phi.T @ np.linalg.inv(Ppred_f[t + 1])
                ms = m_f[t] + J @ (Gamma[:, t + 1] - pred_f[t + 1])
                Vs = V_f[t] - J @ Phi @ V_f[t]
                Gamma[:, t] = ms + _chol_psd(Vs) @ rng.standard_normal(p)
            den = sum(Gamma[:, t - 1] @ Gamma[:, t - 1] for t in range(1, T0))
            num = sum(Gamma[:, t - 1] @ Gamma[:, t] for t in range(1, T0))
            mphi = num / den if den > 0 else 0.0
            vphi = s2_g / den if den > 0 else 1.0
            while True:
                cand = rng.normal(mphi, np.sqrt(vphi))
                if abs(cand) < 1.0:
                    break
            phi_g = cand
            scg = 0.5 * sum(((Gamma[:, t] - phi_g * (Gamma[:, t - 1] if t else np.zeros(p))) ** 2).sum()
                            for t in range(T0))
            s2_g = float(_ig(rng, 0.5 + 0.5 * p * T0, scg + 1.0 / max(nu_s2_g, 1e-12)))
            nu_s2_g = float(_ig(rng, 1.0, 1.0 / max(s2_g, 1e-12) + 1.0 / 100.0))

            # (2) factor loadings Eta
            GtG = Gamma @ Gamma.T; Dom = np.diag(omega)
            Vrow = np.linalg.inv(GtG / s2 + Dom / max(s2_eta, 1e-12)); Lrow = _chol_psd(Vrow)
            Rr = Yc_pre - rho * AYc
            if K:
                Rr = Rr - np.einsum('tnk,k->tn', X, beta)
            for i in range(N):
                rhs = Gamma @ Rr[:, i]
                Eta[i] = Vrow @ (rhs / s2) + Lrow @ rng.standard_normal(p)
            sc_eta = sum(Eta[i] @ Dom @ Eta[i] for i in range(N))
            s2_eta = float(_ig(rng, 0.5 + 0.5 * p * N, 0.5 * sc_eta + 1.0 / max(nu_s2_eta, 1e-12)))
            nu_s2_eta = float(_ig(rng, 1.0, 1.0 / max(s2_eta, 1e-12) + 1.0 / 100.0))
            for k in range(p):
                rate = 1.0 / max(nu_omega[k], 1e-12) + 0.5 * np.sum(Eta[:, k] ** 2) / max(s2_eta, 1e-12)
                omega[k] = float(_ig(rng, 0.5 * (N + 1.0), rate))
                nu_omega[k] = float(_ig(rng, 1.0, 1.0 + 1.0 / max(omega[k], 1e-12)))

        # (3) covariate coefficient beta (ridge)
        if K:
            XtX = np.einsum('tnk,tnl->kl', X, X)
            Bt = Yc_pre - rho * AYc
            if p:
                Bt = Bt - (Eta @ Gamma).T
            Bb = np.einsum('tnk,tn->k', X, Bt)
            Ab = XtX + s2 * 1e-6 * np.eye(K)
            Ainv = np.linalg.inv(Ab)
            beta = Ainv @ Bb + _chol_psd(s2 * Ainv) @ rng.standard_normal(K)

        # (4) sigma^2
        U = Yc_pre - rho * AYc
        if K:
            U = U - np.einsum('tnk,k->tn', X, beta)
        if p:
            U = U - (Eta @ Gamma).T
        ss = float(np.einsum('tn,tn->', U, U))
        s2 = float(_ig(rng, a0 + 0.5 * T0 * N, b0 + 0.5 * ss))

        # (5) rho via random-walk Metropolis
        lc = loglik(rho)
        prop = rho + step_rho * rng.standard_normal()
        lp = loglik(prop)
        if np.log(rng.random()) < lp - lc:
            rho = prop
            if it >= burn:
                acc += 1
        if it >= burn:
            out_rho[it - burn] = rho
            out_s2[it - burn] = s2
            if K:
                out_beta[it - burn] = beta
    return {"rho": out_rho, "s2": out_s2, "beta": out_beta,
            "acc": acc / max(1, iters - burn)}


# ----------------------- effect plug-ins (Thms 1-2) ----------------------
def treated_counterfactual(Y0_post, Yc_post, Wn, w, alpha_hat, rho):
    """Treated-unit untreated counterfactual over the post period (eq. 5)."""
    N = len(alpha_hat); IN = np.eye(N)
    Ainv = np.linalg.inv(IN - rho * (np.outer(w, alpha_hat) + Wn))
    B = IN - rho * Wn
    ycf = Ainv @ (B @ Yc_post.T - rho * np.outer(w, Y0_post))
    return alpha_hat @ ycf


def spillover_effects(Y0, Yc, Wn, w, alpha_hat, rho):
    """Per-control spillover effects (eq. 6) over the supplied periods.

    Returns an ``(T, N)`` array ``Yc - Yc_counterfactual``.
    """
    N = len(alpha_hat); IN = np.eye(N)
    Ainv = np.linalg.inv(IN - rho * (np.outer(w, alpha_hat) + Wn))
    B = IN - rho * Wn
    cf = (Ainv @ (B @ Yc.T - rho * np.outer(w, Y0))).T
    return Yc - cf
