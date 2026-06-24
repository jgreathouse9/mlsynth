"""Numerical engine for the multivariate Bayesian structural time series
(MBSTS) causal model of Menchetti and Bojinov (2022).

This is a faithful, dependency-light (NumPy/SciPy) port of the ``CausalMBSTS``
R package (Bojinov and Menchetti 2020), validated cell-by-cell against it on the
package's own vignette and on the Florence supermarket study. The pieces:

* :func:`build_ssm` assembles the multivariate structural state space for a
  subset of ``{"trend", "slope", "seasonal", "cycle"}`` components, with the
  ``d`` series sharing each component's disturbance covariance.
* :func:`ffbs` draws the latent state path by a forward-filter / backward-sample
  (Carter-Kohn) pass -- the Durbin-Koopman simulation smoother used by the
  reference package, written so it never silently under-disperses.
* :func:`run_gibbs` is the Gibbs sampler: conjugate Inverse-Wishart updates for
  each state-disturbance covariance and the observation covariance, plus the
  spike-and-slab variable-selection branch when regressors are supplied.
* :func:`causal_effect` forecasts the counterfactual from the posterior
  predictive distribution and returns the observed-minus-predicted effect, per
  series, with credible bands.

References
----------
Menchetti, F. and Bojinov, I. (2022). "Estimating the effectiveness of permanent
price reductions for competing products using multivariate Bayesian structural
time series models." Annals of Applied Statistics 16(1): 414-435.
Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N. and Scott, S. L. (2015).
"Inferring causal impact using Bayesian structural time-series models." Annals
of Applied Statistics 9(1): 247-274.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import invwishart
from scipy.special import multigammaln

# The component identifiers understood by :func:`build_ssm`.
VALID_COMPONENTS = ("trend", "slope", "seasonal", "cycle")


def _sym(matrix: np.ndarray) -> np.ndarray:
    """Symmetrize, guarding against floating-point asymmetry."""
    return 0.5 * (matrix + matrix.T)


def _inv(matrix: np.ndarray) -> np.ndarray:
    """SPD inverse via Cholesky (mirrors the reference package's ``inv``)."""
    L = np.linalg.cholesky(_sym(matrix))
    return np.linalg.inv(L).T @ np.linalg.inv(L)


def _mvn(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Robust multivariate-normal draw.

    Tries a Cholesky factor; on a (numerically) non-PSD covariance falls back to
    an eigen-clip. This avoids the silent variance loss ``numpy``'s sampler
    incurs on a non-PSD covariance, which otherwise shrinks the forecast fan.
    """
    cov = _sym(cov)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        L = V * np.sqrt(np.clip(w, 0.0, None))
    return mean + L @ rng.standard_normal(mean.shape[0])


def build_ssm(
    d: int,
    components: List[str],
    seas_period: Optional[int] = None,
    cycle_period: Optional[int] = None,
    rho_cycle: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[slice, str]], int]:
    """Assemble the multivariate structural state space.

    Parameters
    ----------
    d : int
        Number of jointly modelled outcome series.
    components : list of str
        Subset of ``{"trend", "slope", "seasonal", "cycle"}``. ``"slope"``
        requires ``"trend"`` (it upgrades the local level to a local linear
        trend).
    seas_period, cycle_period : int, optional
        Required when ``"seasonal"`` / ``"cycle"`` is present.
    rho_cycle : float
        Cycle damping factor (``1.0`` = undamped, the reference default).

    Returns
    -------
    T : np.ndarray
        ``(M, M)`` state-transition matrix.
    Z : np.ndarray
        ``(d, M)`` observation/selection matrix.
    dist : list of (slice, str)
        One entry per disturbance block: the state indices it occupies and the
        covariance key (``"level"``/``"slope"``/``"seasonal"``/``"cycle1"``/
        ``"cycle2"``) whose ``(d, d)`` covariance drives it.
    M : int
        Total number of states.
    """
    eye = np.eye(d)
    has_slope = "slope" in components
    specs: List[Tuple[str, int, int]] = []
    pos = 0
    if "trend" in components:
        size = 2 * d if has_slope else d
        specs.append(("trend2" if has_slope else "trend1", pos, size))
        pos += size
    if "seasonal" in components:
        specs.append(("seasonal", pos, (seas_period - 1) * d))
        pos += (seas_period - 1) * d
    if "cycle" in components:
        specs.append(("cycle", pos, 2 * d))
        pos += 2 * d
    M = pos
    T = np.zeros((M, M))
    Z = np.zeros((d, M))
    dist: List[Tuple[slice, str]] = []
    for name, p, _ in specs:
        if name == "trend1":
            T[p:p + d, p:p + d] = eye
            Z[:, p:p + d] = eye
            dist.append((slice(p, p + d), "level"))
        elif name == "trend2":
            T[p:p + d, p:p + d] = eye
            T[p:p + d, p + d:p + 2 * d] = eye          # level += slope
            T[p + d:p + 2 * d, p + d:p + 2 * d] = eye  # slope random walk
            Z[:, p:p + d] = eye
            dist.append((slice(p, p + d), "level"))
            dist.append((slice(p + d, p + 2 * d), "slope"))
        elif name == "seasonal":
            nseas = seas_period - 1
            for b in range(nseas):                     # gamma^1 = -sum gamma + w
                T[p:p + d, p + b * d:p + (b + 1) * d] = -eye
            for b in range(1, nseas):                  # shift the lags
                T[p + b * d:p + (b + 1) * d, p + (b - 1) * d:p + b * d] = eye
            Z[:, p:p + d] = eye
            dist.append((slice(p, p + d), "seasonal"))
        elif name == "cycle":
            lam = 2 * np.pi / cycle_period
            c, s = rho_cycle * np.cos(lam), rho_cycle * np.sin(lam)
            T[p:p + d, p:p + d] = c * eye
            T[p:p + d, p + d:p + 2 * d] = s * eye
            T[p + d:p + 2 * d, p:p + d] = -s * eye
            T[p + d:p + 2 * d, p + d:p + 2 * d] = c * eye
            Z[:, p:p + d] = eye                        # observe psi
            dist.append((slice(p, p + d), "cycle1"))
            dist.append((slice(p + d, p + 2 * d), "cycle2"))
    return T, Z, dist, M


def build_Q(M: int, dist: List[Tuple[slice, str]], sig: Dict[str, np.ndarray]) -> np.ndarray:
    """State-disturbance covariance: ``sig[key]`` on each block's diagonal."""
    Q = np.zeros((M, M))
    for sl, key in dist:
        Q[sl, sl] = sig[key]
    return Q


def ffbs(
    y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    rng: np.random.Generator,
    P1: float = 1e6,
) -> np.ndarray:
    """Forward-filter / backward-sample draw of the latent state path.

    Equivalent to the reference package's Durbin-Koopman ``simulateSSM`` for
    sampling ``p(alpha | y)``. Uses the Joseph-form covariance update and a
    robust backward draw for numerical stability under a diffuse initial state.
    """
    n, _ = y.shape
    M = T.shape[0]
    Im = np.eye(M)
    a_f = np.zeros((n, M))
    P_f = np.zeros((n, M, M))
    a_p = np.zeros(M)
    P_p = P1 * Im                                       # approximate diffuse init
    for k in range(n):
        S = Z @ P_p @ Z.T + H
        K = P_p @ Z.T @ np.linalg.inv(S)
        a_f[k] = a_p + K @ (y[k] - Z @ a_p)
        ImKZ = Im - K @ Z
        P_f[k] = _sym(ImKZ @ P_p @ ImKZ.T + K @ H @ K.T)
        a_p = T @ a_f[k]
        P_p = _sym(T @ P_f[k] @ T.T + Q)
    alpha = np.zeros((n, M))
    alpha[n - 1] = _mvn(a_f[n - 1], P_f[n - 1], rng)
    for k in range(n - 2, -1, -1):
        Pp = _sym(T @ P_f[k] @ T.T + Q)
        J = P_f[k] @ T.T @ np.linalg.pinv(Pp)
        m = a_f[k] + J @ (alpha[k + 1] - T @ a_f[k])
        V = _sym(P_f[k] - J @ Pp @ J.T)
        alpha[k] = _mvn(m, V, rng)
    return alpha


def _lpy_X(y: np.ndarray, X: np.ndarray, Hsub: np.ndarray, nu0: float, s0: np.ndarray) -> float:
    """Marginal log-likelihood ``p(y | included regressors)`` (lpy.X in mbsts)."""
    t, d = y.shape
    if X.shape[1] == 0:
        nu = nu0 + t
        se = s0 + y.T @ y
        return float(-(d * t / 2) * np.log(np.pi) + (nu0 / 2) * np.linalg.slogdet(s0)[1]
                     + multigammaln(nu / 2, d) - multigammaln(nu0 / 2, d)
                     - (nu / 2) * np.linalg.slogdet(se)[1])
    Hinv = _inv(Hsub)
    W = _inv(X.T @ X + Hinv)
    Mm = W @ X.T @ y
    nu = nu0 + t
    se = s0 + y.T @ y - Mm.T @ _inv(W) @ Mm
    return float((d / 2) * np.linalg.slogdet(Hinv @ W)[1] - (d * t / 2) * np.log(np.pi)
                 + (nu0 / 2) * np.linalg.slogdet(s0)[1] + multigammaln(nu / 2, d)
                 - multigammaln(nu0 / 2, d) - (nu / 2) * np.linalg.slogdet(se)[1])


def _matrixnormal(Mmean: np.ndarray, U: np.ndarray, Vcol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw beta ~ MN(Mmean, U rows, Vcol cols), Mmean shape (k, d)."""
    k, d = Mmean.shape
    Lu = np.linalg.cholesky(_sym(U) + 1e-12 * np.eye(k))
    Lv = np.linalg.cholesky(_sym(Vcol) + 1e-12 * np.eye(d))
    return Mmean + Lu @ rng.standard_normal((k, d)) @ Lv.T


def run_gibbs(
    Y_pre: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    dist: List[Tuple[slice, str]],
    M: int,
    s0_state: np.ndarray,
    s0_obs: np.ndarray,
    nu0: float,
    niter: int,
    burn: int,
    rng: np.random.Generator,
    X_pre: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Gibbs sampler for the pre-intervention MBSTS model.

    Returns a dict with the post-burn-in per-draw forecast inputs
    (``last_state``, ``Q``, ``Sigma_eps``, ``beta``), the posterior-mean
    in-sample fit ``prefit_mean`` (``T0 x d``), and the inclusion frequencies.
    """
    n, d = Y_pre.shape
    keys = list(dict.fromkeys(k for _, k in dist))
    sig = {k: s0_state.copy() for k in keys}
    Se = s0_obs.copy()
    has_X = X_pre is not None
    if has_X:
        H = _inv(X_pre.T @ X_pre)                       # Zellner g-prior (c = 1)
        P = X_pre.shape[1]
        incl = np.zeros(P)
    draws: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    prefit_sum = np.zeros((n, d))
    kept = 0
    for it in range(niter):
        Q = build_Q(M, dist, sig)
        alpha = ffbs(Y_pre, T, Z, Q, Se, rng)
        deta = alpha[1:] - alpha[:-1] @ T.T
        ne = deta.shape[0]
        for key in keys:
            acc = np.zeros((d, d))
            for sl, kk in dist:
                if kk == key:
                    acc += deta[:, sl].T @ deta[:, sl]
            sig[key] = np.atleast_2d(invwishart.rvs(nu0 + ne, _sym(s0_state + acc), random_state=rng))
        beta: Optional[np.ndarray] = None
        if not has_X:
            eps = Y_pre - alpha @ Z.T
            Se = np.atleast_2d(invwishart.rvs(nu0 + n, _sym(s0_obs + eps.T @ eps), random_state=rng))
        else:
            ystar = Y_pre - alpha @ Z.T
            z = np.ones(P, dtype=int)
            lpc = _lpy_X(ystar, X_pre[:, z == 1], H[np.ix_(z == 1, z == 1)], nu0, s0_obs)
            for j in rng.permutation(P):
                zp = z.copy()
                zp[j] = 1 - zp[j]
                lpp = _lpy_X(ystar, X_pre[:, zp == 1], H[np.ix_(zp == 1, zp == 1)], nu0, s0_obs)
                re = (lpp - lpc) * ((-1.0) ** (zp[j] == 0))
                z[j] = rng.binomial(1, 1.0 / (1.0 + np.exp(-re)))
                if z[j] == zp[j]:
                    lpc = lpp
            beta = np.zeros((P, d))
            if z.sum() == 0:
                Se = np.atleast_2d(invwishart.rvs(nu0 + n, _sym(s0_obs + ystar.T @ ystar), random_state=rng))
            else:
                Xs = X_pre[:, z == 1]
                W = _inv(Xs.T @ Xs + _inv(H[np.ix_(z == 1, z == 1)]))
                Mm = W @ Xs.T @ ystar
                Se = np.atleast_2d(invwishart.rvs(
                    nu0 + n, _sym(s0_obs + ystar.T @ ystar - Mm.T @ _inv(W) @ Mm), random_state=rng))
                beta[z == 1] = _matrixnormal(Mm, W, Se, rng)
            if it >= burn:
                incl += z
        if it >= burn:
            draws.append((alpha[-1].copy(), build_Q(M, dist, sig), Se.copy(), beta))
            prefit_sum += alpha @ Z.T
            kept += 1
    out: Dict[str, Any] = {"draws": draws, "prefit_mean": prefit_sum / max(kept, 1), "n_kept": kept}
    if has_X:
        out["inclusion"] = incl / max(kept, 1)
    return out


def causal_effect(
    Y_post: np.ndarray,
    fit: Dict[str, Any],
    T: np.ndarray,
    Z: np.ndarray,
    rng: np.random.Generator,
    X_post: Optional[np.ndarray] = None,
    excl_post: Optional[np.ndarray] = None,
    ci_alpha: float = 0.05,
    horizon: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Posterior-predictive counterfactual and per-series causal effect.

    Forecasts each post-burn-in draw forward, differences against the observed
    post-period outcome, and summarizes per series: temporal-average effect with
    credible bounds, cumulative effect with bounds, the per-period effect path,
    and the posterior-mean counterfactual over the post window.

    ``horizon`` restricts the temporal-average and cumulative summaries to the
    first ``horizon`` retained post periods (the per-period paths stay full);
    ``None`` summarizes the whole post window.
    """
    draws = fit["draws"]
    h, d = Y_post.shape
    Mst = T.shape[0]
    a_lo, a_hi = 100 * (ci_alpha / 2), 100 * (1 - ci_alpha / 2)
    ydiff = np.empty((len(draws), h, d))
    cf_post_sum = np.zeros((h, d))
    for mi, (a_last, Q, Se, beta) in enumerate(draws):
        al = a_last.copy()
        state_fc = np.empty((h, d))
        for k in range(h):
            al = _mvn(T @ al, Q, rng)
            state_fc[k] = Z @ al
        if beta is not None and X_post is not None:
            state_fc = state_fc + X_post @ beta
        cf_post_sum += state_fc
        # predict_mbsts.R STEP 2.2: one obs-noise draw per posterior draw,
        # recycled (column-major) across the forecast horizon.
        ev = _mvn(np.zeros(d), Se, rng)
        ystar = np.reshape(np.resize(ev, h * d), (h, d), order="F")
        ydiff[mi] = Y_post - (state_fc + ystar)
    cf_post_mean = cf_post_sum / max(len(draws), 1)
    keep = np.ones(h, dtype=bool) if excl_post is None else (np.asarray(excl_post) == 0)
    if horizon is not None:                              # first `horizon` retained periods
        kept_idx = np.where(keep)[0][:horizon]
        keep = np.zeros(h, dtype=bool)
        keep[kept_idx] = True
    ydiff_k = ydiff[:, keep, :]
    tavg = ydiff_k.mean(axis=1)                          # (n_draws, d)
    cum = ydiff_k.sum(axis=1)
    return {
        "att_mean": tavg.mean(axis=0),
        "att_lower": np.percentile(tavg, a_lo, axis=0),
        "att_upper": np.percentile(tavg, a_hi, axis=0),
        "att_samples": tavg,
        "cum_mean": cum.mean(axis=0),
        "cum_lower": np.percentile(cum, a_lo, axis=0),
        "cum_upper": np.percentile(cum, a_hi, axis=0),
        "effect_path": ydiff.mean(axis=0),               # (h, d) per-period mean
        "effect_lower": np.percentile(ydiff, a_lo, axis=0),
        "effect_upper": np.percentile(ydiff, a_hi, axis=0),
        "cf_post_mean": cf_post_mean,                    # (h, d) counterfactual
    }
