"""Why whitening the CWZ refit residuals by an AR(1) does not restore coverage.

Under factors with AR coefficient 0.8 the refit residuals' lag-1 autocorrelation
is only about +0.27, yet every permutation arm stays short. Candidate causes:

1. ``seam`` -- cyclic shifts join the end of the series to its start, so the
   blocks that wrap sum two distant stretches and are less variable than a
   contiguous block such as the post window.
2. ``long-run variance`` -- the residual is a persistent factor component plus
   white noise, an ARMA-like process whose autocorrelations start small and
   decay slowly; an AR(1) fitted to it matches lag 1 and misses the long-run
   variance a 13-period total depends on.
3. ``reference depth`` -- 79 overlapping blocks of one series are few
   effectively independent ones, whatever their shape.

Measurements, at the true total on every draw:

* variance ratios ``E[S_0^2] / E[mean S_k^2]`` of the post-window sum against
  the contiguous (non-wrapping) and the wrapping reference blocks: a ratio
  above one says the reference understates the post sum's spread;
* coverage of the signed-sum test on non-wrapping blocks only (not a group, so
  not exact under anything, but free of the seam);
* coverage after whitening by AR(p) with ``p`` chosen by AIC up to 8, and by an
  ARMA(1, 1) fitted by conditional least squares, levels rebuilt as in
  ``whiten.py``.

Run from the repository root::

    python -m benchmarks.studies.marex_cumulative_conformal.whiten_diag [draws] [dgp]
"""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import minimize

from benchmarks.studies.marex_cumulative_conformal.designs import design, panel
from benchmarks.studies.marex_cumulative_conformal.inversion import blocks
from benchmarks.studies.marex_cumulative_conformal.whiten import (
    fit_ar, level_pvalue, perms)
from mlsynth.utils.conformal.refit import conformal_refit_gaps

ALPHAS = (0.05, 0.2)
TE, TB, H = 40, 26, 13


def aic_order(x, pmax=8):
    best = (np.inf, 1)
    for p in range(1, pmax + 1):
        mu, rho = fit_ar(x, p)
        K, N = p, x.size
        u = x[K:] - mu - sum(rho[j] * x[K - j - 1:N - j - 1] for j in range(K))
        # compare on a common sample so the orders are comparable
        u = u[pmax - p:]
        aic = u.size * np.log(u @ u / u.size) + 2 * (p + 1)
        best = min(best, (aic, p))
    return best[1]


def arma_fit(x):
    """ARMA(1, 1) with an intercept by conditional least squares."""
    def resid(par):
        mu, phi, theta = par
        u = np.zeros_like(x)
        for t in range(1, x.size):
            u[t] = x[t] - mu - phi * x[t - 1] - theta * u[t - 1]
        return u[1:]

    def ssr(par):
        if abs(par[1]) >= 0.99 or abs(par[2]) >= 0.99:
            return 1e12
        r = resid(par)
        return r @ r

    r1 = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    res = minimize(ssr, x0=[x.mean() * (1 - r1), r1, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    mu, phi, theta = res.x
    return mu, phi, theta, np.r_[0.0, resid(res.x)]


def arma_level_pvalue(x, mu, phi, theta, u, P):
    """Permute the ARMA innovations ``u[1:]`` and rebuild levels."""
    U = u[1:][P[1:]]
    R, M = U.shape
    X = np.empty((R, M + 1))
    X[:, 0] = x[0]
    prev = np.zeros(R)
    for t in range(M):
        X[:, t + 1] = mu + phi * X[:, t] + U[:, t] + theta * prev
        prev = U[:, t]
    S = np.abs(X[:, -H:].sum(axis=1))
    s0 = abs(x[-H:].sum())
    return (1 + np.sum(S >= s0 - 1e-9 * (1 + s0))) / P.shape[0]


def run(dgp, draws, seed0=0, R=999):
    T0 = TE + TB
    N = T0 + H
    idx = blocks(T0, H, "cyclic")
    starts = (T0 + np.arange(N)) % N            # block k starts at period starts[k]
    wraps = starts + H > N
    contig = ~wraps
    ends = starts + H                           # exclusive end of a non-wrapping block
    in_fit = contig & (ends <= TE)
    in_blank = contig & (starts >= TE) & (ends <= T0)
    s0_sq, ref_contig, ref_wrap = [], [], []
    ref_fit, ref_blank, pos_var = [], [], []
    hit, orders, phis = {}, [], []
    for d in range(draws):
        rng = np.random.default_rng(seed0 + d)
        Y = panel(N, dgp, rng)
        w, v = design(Y, TE, TB, H)
        controls = np.flatnonzero(w <= 1e-8)
        y = Y @ w + np.r_[np.zeros(T0), np.ones(H)]
        x = conformal_refit_gaps(np.r_[y[:T0], y[T0:] - 1.0], Y[:, controls], rule="sc")
        S = x[idx].sum(axis=1)
        s0_sq.append(S[0] ** 2)
        ref_contig.append(np.mean(S[contig] ** 2))
        ref_wrap.append(np.mean(S[wraps] ** 2))
        ref_fit.append(np.mean(S[in_fit] ** 2))
        ref_blank.append(np.mean(S[in_blank] ** 2) if in_blank.any() else np.nan)
        pos_var.append(x ** 2)
        p = {}
        A = np.abs(S)
        p["cyclic"] = float(np.mean(A >= A[0] - 1e-9 * (1 + A[0])))
        Ac = A[contig]                            # row 0 (the post window) is contiguous
        p["no_wrap"] = float(np.mean(Ac >= A[0] - 1e-9 * (1 + A[0])))
        k = aic_order(x)
        orders.append(k)
        mu, rho = fit_ar(x, k)
        p["white_aic"] = level_pvalue(x, H, mu, rho, perms(N - k, R, rng))
        mu, rho = fit_ar(x, 6)
        p["white_ar6"] = level_pvalue(x, H, mu, rho, perms(N - 6, R, rng))
        mu, phi, theta, u = arma_fit(x)
        phis.append((phi, theta))
        p["white_arma"] = arma_level_pvalue(x, mu, phi, theta, u, perms(N - 1, R, rng))
        for a in ALPHAS:
            for arm, pv in p.items():
                hit[(arm, a)] = hit.get((arm, a), 0) + (pv > a)
    for a in ALPHAS:
        cells = "  ".join(f"{k[0]}={c / draws:.3f}" for k, c in sorted(hit.items()) if k[1] == a)
        print(f"diag {dgp} alpha={a} draws={draws}: {cells}", flush=True)
    ph = np.array(phis)
    pv = np.mean(pos_var, axis=0)
    seg = lambda a, b: float(np.mean(pv[a:b]))
    print(f"diag {dgp}: post sum^2 over blocks inside the fitting periods "
          f"{np.mean(s0_sq) / np.mean(ref_fit):.2f}, inside the blank periods "
          f"{np.mean(s0_sq) / np.nanmean(ref_blank):.2f}; mean squared residual by stretch: "
          f"fit {seg(0, TE):.2f}, blank {seg(TE, T0):.2f}, post {seg(T0, N):.2f}", flush=True)
    print(f"diag {dgp}: variance ratio post / contiguous {np.mean(s0_sq) / np.mean(ref_contig):.2f}, "
          f"post / wrapping {np.mean(s0_sq) / np.mean(ref_wrap):.2f}, "
          f"contiguous / wrapping {np.mean(ref_contig) / np.mean(ref_wrap):.2f}; "
          f"AIC order median {np.median(orders):.0f}; ARMA(1,1) median phi {np.median(ph[:, 0]):+.2f} "
          f"theta {np.median(ph[:, 1]):+.2f}", flush=True)


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    dgps = (sys.argv[2],) if len(sys.argv) > 2 else ("ar", "ar5", "iid")
    for g in dgps:
        run(g, draws)
