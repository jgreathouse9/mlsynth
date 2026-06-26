"""Over-identified doubly-robust proximal estimator (Qiu et al., 2024).

This is the configuration used in the authors' empirical analyses (e.g. the
Kansas 2012 tax-cut study, ``DR_Proximal_SC`` commit ``3bcb5ec``), where the
outcome bridge ``h(W)`` is instrumented by the *full* pool of donor proxies
while the treatment bridge ``q(Z)`` uses only a small selected subset -- i.e.
``#instruments != #donors``. mlsynth's :func:`estimate_dr` covers the
just-identified (square) case; this module covers the over-identified one.

Numerics. The identity-weight GMM objective ``||g_bar(theta)||^2`` is a
nonlinear least squares. The system decouples at the optimum: the moments
``g4``/``g5`` are zeroed by ``phi``/``psi_minus`` for any ``(alpha, beta)``, so

* ``alpha`` (outcome bridge) is the *exact* closed-form least-squares minimiser
  of the linear ``g1`` block -- no optimiser, fully deterministic;
* only the small treatment-bridge block ``(beta, psi)`` is nonlinear, solved by
  trust-region NLS with a multistart guard (and an optional ridge) for the flat
  ``exp(Z beta)`` valley that arises with few instruments;
* ``psi_minus = mean_pre[q (Y-h)]`` and ``phi = mean_post[Y-h] - psi_minus``.

Convergence note. The authors' published table is *under-converged* (R's
``optim(BFGS)`` stops early); this estimator targets the genuine optimum. With
few instruments the ``q``-block is ill-conditioned -- ``converged`` and
``n_basins`` flag when the multistart disagrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..bridges import augment


@dataclass
class DROveridResult:
    att: float                      # phi: doubly-robust ATT
    se: float                       # over-identified GMM sandwich SE (Bartlett-HAC)
    counterfactual: np.ndarray      # h(W), the outcome bridge
    alpha: np.ndarray               # outcome-bridge coefficients
    beta: np.ndarray                # treatment-bridge coefficients
    psi_minus: float                # E_pre[q (Y-h)]
    objective: float                # ||g_bar||^2 at the optimum
    converged: bool                 # multistart agreement (q-block well-conditioned)
    n_basins: int                   # distinct phi basins found across restarts


def _alpha_closed(Y, Wc, GH, pre):
    """Exact LS minimiser of the linear g1 block: argmin_a ||GH_pre'(Y-Wc a)||."""
    M = GH[pre].T @ Wc[pre]
    return np.linalg.lstsq(M, GH[pre].T @ Y[pre], rcond=None)[0]


def _logit(X, y, maxit=200):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        step = np.linalg.solve((X * w[:, None]).T @ X + 1e-9 * np.eye(X.shape[1]),
                               X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


def estimate_dr_overid(
    outcome_vector: np.ndarray,
    donor_outcomes: np.ndarray,
    outcome_instruments: np.ndarray,
    treatment_instruments: np.ndarray,
    num_pre_treatment_periods: int,
    hac_bandwidth: int,
    ridge: float = 0.0,
    n_starts: int = 8,
    seed: int = 0,
) -> DROveridResult:
    """Over-identified DR proximal ATT.

    Parameters
    ----------
    outcome_vector : (T,)
        Treated unit outcome.
    donor_outcomes : (T, nW)
        Donor outcomes ``W`` (the outcome-bridge regressors).
    outcome_instruments : (T, nZh)
        Instruments for the outcome bridge ``h`` (the *full* proxy pool).
    treatment_instruments : (T, nZq)
        Instruments for the treatment bridge ``q`` (the selected subset).
    num_pre_treatment_periods : int
        ``T0``.
    hac_bandwidth : int
        Bartlett-HAC bandwidth for the sandwich SE.
    ridge : float, default 0.0
        L2 penalty on ``beta`` (excluding intercept) to regularise the flat
        ``exp(Z beta)`` valley with few instruments. ``0`` reproduces the
        unregularised optimum.
    n_starts : int, default 8
        Multistart restarts for the ``q``-block.
    seed : int, default 0
        RNG seed for the multistart perturbations (reproducible).
    """
    Y = np.asarray(outcome_vector, float)
    W = np.asarray(donor_outcomes, float)
    Zh = np.asarray(outcome_instruments, float)
    Zq_raw = np.asarray(treatment_instruments, float)
    T = len(Y)
    T0 = int(num_pre_treatment_periods)
    t = np.arange(T)
    pre = t < T0
    post = ~pre
    T1 = int(post.sum())

    Wc = augment(W)
    GH = augment(Zh)
    gq = Wc
    Zq = augment(Zq_raw)
    nZq = Zq.shape[1]

    # --- outcome bridge alpha: exact closed form (linear g1 block) ----------
    alpha = _alpha_closed(Y, Wc, GH, pre)
    h = Wc @ alpha
    Yh = Y - h

    # --- treatment-bridge (beta, psi) block: minimise ||g2_bar||^2+||g3_bar||^2
    m_post = gq[post].sum(0) / T                       # (1/T) sum_post gq

    rng = np.random.default_rng(seed)
    b0 = _logit(Zq, post.astype(float))
    b0[0] += np.log(T1 / (T - T1)) if (T - T1) > 0 else 0.0
    ridge_vec = np.full(nZq, np.sqrt(ridge)); ridge_vec[0] = 0.0

    def resid(beta):
        q = np.exp(np.clip(Zq @ beta, -30, 30))
        m_pre_q = (q[pre, None] * gq[pre]).sum(0) / T  # (1/T) sum_pre q gq
        denom = (T1 / T) ** 2 + (T0 / T) ** 2
        psi = ((T1 / T) * m_post + (T0 / T) * m_pre_q) / denom   # profiled psi
        g2 = (T1 / T) * psi - m_post
        g3 = m_pre_q - (T0 / T) * psi
        return np.concatenate([g2, g3, ridge_vec * beta])

    best = None
    for s in range(max(n_starts, 1)):
        bs = b0 if s == 0 else b0 + rng.normal(0, 2.0, size=nZq)
        try:
            r = least_squares(resid, bs, method="trf", max_nfev=20000,
                              ftol=1e-15, xtol=1e-15, gtol=1e-15)
        except Exception:
            continue
        obj = float(r.fun @ r.fun)
        if best is None or obj < best[0] - 1e-14:
            best = (obj, r.x)
    obj_beta, beta = best

    q = np.exp(np.clip(Zq @ beta, -30, 30))
    psi_minus = float((q[pre] * Yh[pre]).mean())
    phi = float(Yh[post].mean() - psi_minus)

    # convergence / conditioning diagnostic across restarts
    phis = []
    for s in range(max(n_starts, 1)):
        bs = b0 if s == 0 else b0 + rng.normal(0, 2.0, size=nZq)
        try:
            r = least_squares(resid, bs, method="trf", max_nfev=20000,
                              ftol=1e-13, xtol=1e-13, gtol=1e-13)
        except Exception:
            continue
        qq = np.exp(np.clip(Zq @ r.x, -30, 30))
        phis.append(Yh[post].mean() - float((qq[pre] * Yh[pre]).mean()))
    phis = np.array(phis)
    n_basins = int(len({round(p, 4) for p in phis})) if len(phis) else 1
    converged = bool(n_basins == 1)

    se = _overid_se(Y, Wc, GH, Zq, gq, alpha, beta, psi_minus, phi,
                    pre, post, T0, T1, T, hac_bandwidth)

    # full-system objective at the solution (alpha exact, g4/g5 == 0)
    objective = obj_beta + float((GH[pre].T @ (Yh[pre]) / T) @ (GH[pre].T @ (Yh[pre]) / T))
    return DROveridResult(att=phi, se=se, counterfactual=h, alpha=alpha, beta=beta,
                          psi_minus=psi_minus, objective=objective,
                          converged=converged, n_basins=n_basins)


def _bartlett_hac(U, bw):
    T = U.shape[0]
    omega = U.T @ U / T
    for lag in range(1, max(bw, 0) + 1):
        if lag >= T:
            break
        w = 1.0 - lag / (bw + 1)
        auto = U[:-lag].T @ U[lag:] / T
        omega += w * (auto + auto.T)
    return omega


def _overid_se(Y, Wc, GH, Zq, gq, alpha, beta, psi_minus, phi,
               pre, post, T0, T1, T, bw):
    """Over-identified GMM sandwich SE for phi: (G'G)^-1 G' Omega G (G'G)^-1 / T."""
    nA = Wc.shape[1]; nB = Zq.shape[1]; nP = gq.shape[1]
    theta = np.concatenate([alpha, beta, gq[post].mean(0), [phi], [psi_minus]])
    iA = slice(0, nA); iB = slice(nA, nA + nB)
    iPs = slice(nA + nB, nA + nB + nP); iphi = nA + nB + nP; ipm = iphi + 1

    def moments(th):
        a = th[iA]; b = th[iB]; ps = th[iPs]; ph = th[iphi]; pm = th[ipm]
        h = Wc @ a; q = np.exp(np.clip(Zq @ b, -30, 30)); Yh = Y - h
        g1 = pre[:, None] * (Yh[:, None] * GH)
        g2 = post[:, None] * (ps[None, :] - gq)
        g3 = pre[:, None] * (q[:, None] * gq - ps[None, :])
        g4 = (post * (ph - Yh + pm))[:, None]
        g5 = (pre * (pm - q * Yh))[:, None]
        return np.column_stack([g1, g2, g3, g4, g5])

    U = moments(theta)
    base = U.mean(0)
    p = len(theta)
    G = np.zeros((len(base), p))
    eps = 1e-6
    for j in range(p):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        G[:, j] = (moments(tp).mean(0) - moments(tm).mean(0)) / (2 * eps)
    omega = _bartlett_hac(U, bw)
    try:
        GtG_inv = np.linalg.inv(G.T @ G)
        bread = GtG_inv @ G.T
        cov = bread @ omega @ bread.T / T
        var = cov[iphi, iphi]
        return float(np.sqrt(var)) if var >= 0 else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")
