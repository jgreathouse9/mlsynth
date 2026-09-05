"""Perturbation-based non-regular inference for DROSC (Koo & Guo 2026, helpers.R
``DRoSC`` with ``Inference = TRUE``).

The DROSC ATT's limiting law is non-normal (it is defined by an inequality-
constrained optimisation). Inference perturbs the pre-treatment second moments,
the treated-donor cross moments, and the post-treatment means from their
(HAC-)sampling distributions, re-solves the moment-band problem for each draw,
forms a normal interval around each draw's effect, and returns the *union* of
those intervals -- a possibly disjoint confidence set.

The per-draw solve reuses a precompiled DPP/OSQP problem so the M-draw pass is
fast (a few seconds at M = 500), instead of re-canonicalising cvxpy each draw.
"""
from __future__ import annotations

from typing import List

import cvxpy as cp
import numpy as np
from scipy.stats import norm


def hac_cov(Y: np.ndarray, lag: int = None):
    """Newey-West / Bartlett HAC covariance (helpers.R ``HAC_cov_matrix``)."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T0, N = Y.shape
    if lag is None:
        lag = int(np.floor(4 * (T0 / 100) ** (2 / 9)))
    Yb = Y.mean(0)
    S = np.zeros((N, N))
    for h in range(lag + 1):
        if h == 0:
            S += (Y - Yb).T @ (Y - Yb) / (T0 - 1)
        else:
            G = (Y[:T0 - h] - Yb).T @ (Y[h:] - Yb) / (T0 - h)
            S += (1 - h / (lag + 1)) * (G + G.T)
    return S if N > 1 else float(S[0, 0])


def _vecl(A: np.ndarray) -> np.ndarray:
    """Lower triangle (incl. diagonal), column-major (helpers.R ``vecl``)."""
    N = A.shape[0]
    return np.concatenate([A[j:, j] for j in range(N)])


def _unvecl_sym(v: np.ndarray, N: int) -> np.ndarray:
    M = np.zeros((N, N))
    k = 0
    for j in range(N):
        for i in range(j, N):
            M[i, j] = M[j, i] = v[k]
            k += 1
    return M


def _mvn_sampler(mu, cov):
    """Return ``(sample(rng), sd)`` for ``N(mu, cov)`` with the factor precomputed
    once (``numpy.multivariate_normal`` re-factorises the covariance every call,
    which dominates the M-draw loop for the high-dimensional moment vector)."""
    cov = np.atleast_2d(np.asarray(cov, float))
    mu = np.asarray(mu, float).ravel()
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ev, V = np.linalg.eigh((cov + cov.T) / 2.0)
    factor = V @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))    # PSD-safe square root

    def sample(rng):
        return mu + factor @ rng.standard_normal(mu.shape[0])

    return sample, sd


class _BandSolver:
    """Precompiled DPP problem re-solved (OSQP, warm-started) across draws."""

    def __init__(self, N: int) -> None:
        self.b = cp.Variable(N)
        self.a = cp.Parameter(N)
        self.ey = cp.Parameter()
        self.Sig = cp.Parameter((N, N), PSD=True)
        self.lo = cp.Parameter(N)
        self.hi = cp.Parameter(N)
        cons = [cp.sum(self.b) == 1, self.b >= 0,
                self.Sig @ self.b >= self.lo, self.Sig @ self.b <= self.hi]
        self.prob = cp.Problem(cp.Minimize(cp.square(self.a @ self.b - self.ey)), cons)

    def solve(self, a, ey, Sig, lo, hi):
        self.a.value = a
        self.ey.value = float(ey)
        self.Sig.value = Sig
        self.lo.value = lo
        self.hi.value = hi
        try:
            # CLARABEL (interior-point) is robust on the rank-1 objective where
            # OSQP's ADMM wanders; DPP caches the canonicalisation so the M-draw
            # loop re-solves without re-canonicalising each time.
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:  # pragma: no cover - solver hiccup on a degenerate draw
            return None
        if self.b.value is None:
            return None
        w = np.clip(np.asarray(self.b.value, float).ravel(), 0.0, None)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            return None
        return w / s          # renormalise to the simplex (guards against a
        #                       slightly-infeasible OSQP point producing an
        #                       out-of-range interval center)


def drosc_union_ci(Y0, Y1, X0, X1, robustness_lambda=0.0, alpha=0.05, alpha0=0.01,
                   c_sample=0.01, n_perturbations=500, gamma=1.0, conditional=False,
                   dependent=False, seed=0) -> List[List[float]]:
    """Perturbation union confidence interval for the DROSC ATT.

    A normal interval is invalid here for two reasons the paper separates. The
    optimum typically sits on the boundary of the compatibility band, and small
    sampling changes flip which constraints are active, giving a mixture limit
    (non-regularity). And correlated controls make the band nearly flat in some
    directions, so small errors in the estimated moments move it a lot
    (instability). The two arrive together.

    The remedy: perturb the moments and the post-treatment means from their
    sampling distributions, re-solve for each draw, and form a normal interval
    around each. Some draw nearly recovers the population problem, and for that
    one the residual uncertainty is almost all in ``mean(Y1)``, which is
    asymptotically normal. Since that draw cannot be identified, the plausible
    ones are kept and their intervals unioned.

    Returns a sorted list of disjoint ``[lo, hi]`` intervals (their union is the
    confidence set, so read it as the effects the data cannot reject, not as a
    point estimate plus symmetric error). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    Y0 = np.asarray(Y0, float).ravel()
    Y1 = np.asarray(Y1, float).ravel()
    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)
    T0, N = X0.shape
    T1 = len(Y1)
    M = int(n_perturbations)
    Sig = X0.T @ X0 / T0
    gam = (Y0[:, None] * X0).mean(0)
    EY1 = Y1.mean()
    EX1 = X1.mean(0)
    VarY1 = float(hac_cov(Y1) if dependent else np.var(Y1, ddof=1))
    CovX1 = hac_cov(X1) if dependent else np.cov(X1, rowvar=False)

    vec_sig = _vecl(Sig)
    vecl_X0 = np.array([_vecl(X0[t][:, None] @ X0[t][None, :]) for t in range(T0)])
    cov_vecl = hac_cov(vecl_X0) if dependent else np.cov(vecl_X0, rowvar=False)
    dW = 0.0 if T0 > N * (N + 1) / 2 else max(gamma * np.max(np.abs(cov_vecl)), 1.0)
    Sigma_cov = cov_vecl / T0 + dW / T0 * np.eye(cov_vecl.shape[0])
    Y0X0_cov = hac_cov(Y0[:, None] * X0) if dependent else np.cov(Y0[:, None] * X0, rowvar=False)
    dZ = 0.0 if T0 > N else max(gamma * np.max(np.abs(Y0X0_cov)), 1.0)
    dX = 0.0 if T1 > N else max(gamma * np.max(np.abs(CovX1)), 1.0)
    XY_cov = Y0X0_cov / T0 + dZ / T0 * np.eye(N)
    EX_cov = CovX1 / T1 + dX / T1 * np.eye(N)

    def outlier(x, mu, sd):
        return np.max(np.abs(x - mu) / sd) > 1.05 * norm.ppf(1 - alpha0 / (2 * np.size(x)))

    xx_sampler, xx_sd = _mvn_sampler(vec_sig, Sigma_cov)
    xy_sampler, xy_sd = _mvn_sampler(gam, XY_cov)
    ex_sampler, ex_sd = _mvn_sampler(EX1, EX_cov)
    band_scale = max(np.max(xy_sd), np.max(np.sqrt(np.diag(Sigma_cov))))

    solver = _BandSolver(N)
    prop = 0.0
    betas: List[np.ndarray] = []
    mus: List[np.ndarray] = []
    rounds = 0
    # The moment band starts far tighter than the perturbation scale, so c_sample
    # inflates (x1.25/round, as in the reference) until >= 10% of draws are
    # feasible -- for a 16-donor panel this is ~26 rounds (c_sample ~ 3.3). Cap
    # generously so it always terminates; degrade gracefully below.
    while prop < 0.1 and rounds < 60:
        rounds += 1
        betas, mus = [], []
        for _ in range(M):
            xx = xx_sampler(rng)
            if outlier(xx, vec_sig, xx_sd):
                continue
            if conditional:
                Sig_m = Sig
            else:
                ev, V = np.linalg.eigh(_unvecl_sym(xx, N))
                Sig_m = V @ np.diag(np.maximum(ev, 1e-3)) @ V.T
            xy = xy_sampler(rng)
            if outlier(xy, gam, xy_sd):
                continue
            ey = rng.normal(EY1, np.sqrt(VarY1 / T1))
            if abs(ey - EY1) / np.sqrt(VarY1 / T1) > 1.05 * norm.ppf(1 - alpha0 / 2):
                continue
            if conditional:
                ex = EX1
            else:
                ex = ex_sampler(rng)
                if outlier(ex, EX1, ex_sd):
                    continue
            lam_hat = robustness_lambda + c_sample / np.sqrt(T0) * (
                np.log(min(T0, T1)) / M) ** (1 / (1 + N * (N + 5) / 2)) * band_scale
            bm = solver.solve(ex, ey, Sig_m, xy - lam_hat, xy + lam_hat)
            if bm is None:
                continue
            betas.append(bm)
            mus.append(ex)
        prop = len(betas) / M if M else 0.0
        if prop < 0.1:
            c_sample *= 1.25
    if not betas:  # pragma: no cover - no feasible draw survived (degenerate panel)
        return []

    SE = np.sqrt(VarY1)
    z = norm.ppf(1 - (alpha - alpha0) / 2)
    ints = sorted([[EY1 - mu @ b - z * SE / np.sqrt(T1),
                    EY1 - mu @ b + z * SE / np.sqrt(T1)] for b, mu in zip(betas, mus)])
    union: List[List[float]] = []
    for lo, hi in ints:
        if union and lo <= union[-1][1]:
            union[-1][1] = float(max(union[-1][1], hi))
        else:
            union.append([float(lo), float(hi)])
    return union
