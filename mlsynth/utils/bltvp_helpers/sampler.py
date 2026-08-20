"""Gibbs sampler for BLTVP (Klinenberg 2023, sec. 3 and appendix A.1).

The model is the noncentered state space of eqs (10)-(13)::

    y_t = sum_j (beta_j + btil_{j,t} sqrt_theta_j) x_{j,t} + eps_t,  eps ~ N(0, sig2)
    btil_{j,t} = btil_{j,t-1} + eta_t,   eta ~ N(0, 1)
    btil_{j,0} ~ N(0, P0_jj)

Writing the drift scale as a *signed* root puts zero in the interior of its
prior. That is what lets the time-varying half of a coefficient be shrunk
away: an inverse gamma on ``theta_j`` has zero at the edge of its support and
so cannot, which is why CausalImpact's dynamic-regression mode has to make
every coefficient dynamic and pays for it in interval width (paper, sec. 2.2).

One sweep:

1. FFBS for ``btil_{0:T0}`` -- forward Kalman filter, backward sample.
2. A joint conjugate draw of ``(beta, sqrt_theta)``. In the noncentered form
   the two blocks are one linear regression on ``[X, X * btil]``, so they come
   from a single multivariate normal instead of two blocked draws.
3. The ASIS interweave (Yu & Meng 2011): move to the centered parameterisation,
   redraw ``theta_j`` and ``beta_j`` there, and move back. Only the mixing
   changes, not the target.
4. Hyperparameters, eqs (25)-(31). The Bayesian Lasso conditionals for
   ``alpha2`` and ``xi2`` are drawn in the inverse-Gaussian form of Park &
   Casella (2008), which is the same law as the GIG the paper writes.

``sqrt_theta_j`` and ``btil_{j,.}`` are identified only up to a joint sign
flip, so a random sign switch is applied each sweep. Report ``abs(sqrt_theta)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import geninvgauss, invgauss

_JITTER = 1e-10
_FLOOR = 1e-12


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _rgig(p: float, a: float, b: float, rng: np.random.Generator) -> float:
    """Draw from GIG(p, a, b): density proportional to x^(p-1) exp(-(a x + b/x)/2)."""
    a = max(float(a), _FLOOR)
    b = max(float(b), _FLOOR)
    return float(geninvgauss.rvs(p, np.sqrt(a * b), scale=np.sqrt(b / a),
                                 random_state=rng))


def _rinvgauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """Draw from InverseGaussian(mu, lam)."""
    return float(invgauss.rvs(mu / lam, scale=lam, random_state=rng))


def ffbs(y_star: np.ndarray, H: np.ndarray, sig2: float, P0: np.ndarray,
         rng: np.random.Generator) -> np.ndarray:
    """Sample the state path given everything else.

    The observation is ``y_star_t = H[t] @ btil_t + N(0, sig2)`` and the state
    is a unit-variance random walk, so the transition matrix is the identity
    and the state covariance is ``I``.

    Parameters
    ----------
    y_star : np.ndarray
        Length-``T0`` outcome with the time-invariant part already subtracted.
    H : np.ndarray
        Shape ``(T0, m)`` state design, ``H[t, j] = sqrt_theta_j x_{j,t}``.
    sig2 : float
        Observation variance.
    P0 : np.ndarray
        Length-``m`` prior variances of the initial state.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape ``(T0 + 1, m)`` path, row 0 being the initial state.
    """
    T, m = H.shape
    a_f = np.zeros((T + 1, m))
    R_f = np.zeros((T + 1, m, m))
    R_p = np.zeros((T + 1, m, m))
    R_f[0] = np.diag(P0)

    for t in range(1, T + 1):
        Rp = _sym(R_f[t - 1] + np.eye(m))
        R_p[t] = Rp
        h = H[t - 1]
        Q = float(h @ Rp @ h) + sig2
        K = (Rp @ h) / Q
        resid = y_star[t - 1] - float(h @ a_f[t - 1])
        a_f[t] = a_f[t - 1] + K * resid
        R_f[t] = _sym(Rp - np.outer(K, K) * Q)

    btil = np.zeros((T + 1, m))
    L = np.linalg.cholesky(_sym(R_f[T]) + _JITTER * np.eye(m))
    btil[T] = a_f[T] + L @ rng.standard_normal(m)
    for t in range(T - 1, -1, -1):
        Rt = _sym(R_f[t])
        G = np.linalg.solve(R_p[t + 1], Rt).T
        mean = a_f[t] + G @ (btil[t + 1] - a_f[t])
        V = _sym(Rt - G @ Rt)
        w, U = np.linalg.eigh(V)
        btil[t] = mean + (U * np.sqrt(np.clip(w, 0.0, None))) @ rng.standard_normal(m)
    return btil


def _one_chain(y, X, T0, n_iter, burn_in, rng, time_varying,
               a1, g0, G0, z1, z2, z1_dyn, z2_dyn):
    """Run a single chain and return its retained draws."""
    T, m = X.shape
    Xp, yp = X[:T0], y[:T0]

    beta = np.zeros(m)
    sqth = np.zeros(m)
    btil = np.zeros((T0 + 1, m))
    sig2 = float(np.var(yp)) or 1.0
    a2 = 1.0
    al2 = np.ones(m)
    xi2 = np.ones(m)
    lam2 = kap2 = 1.0
    P0 = np.ones(m)

    keep = n_iter - burn_in
    pred = np.zeros((T, keep))
    beta_d = np.zeros((m, keep))
    sqth_d = np.zeros((m, keep))
    sig2_d = np.zeros(keep)
    k = 0

    for it in range(n_iter):
        if time_varying:
            btil = ffbs(yp - Xp @ beta, Xp * sqth, sig2, P0, rng)
            Z = np.hstack([Xp, Xp * btil[1:]])
            prior = np.concatenate([al2, xi2])
        else:
            Z = Xp
            prior = al2

        A = Z.T @ Z / sig2 + np.diag(1.0 / prior)
        L = np.linalg.cholesky(_sym(A))
        centre = np.linalg.solve(A, Z.T @ yp / sig2)
        draw = centre + np.linalg.solve(L.T, rng.standard_normal(Z.shape[1]))
        beta = draw[:m]
        if time_varying:
            sqth = draw[m:]

            # ASIS: redraw (beta, theta) through the centered parameterisation.
            centered = beta + sqth * btil
            for j in range(m):
                diffs = np.diff(centered[:, j])
                S = float(diffs @ diffs) + (centered[0, j] - beta[j]) ** 2 / P0[j]
                th = _rgig(-(T0 - 1) / 2.0, 1.0 / xi2[j], max(S, _FLOOR), rng)
                denom = al2[j] + th * P0[j]
                var = al2[j] * th * P0[j] / denom
                mean = centered[0, j] * al2[j] / denom
                beta[j] = mean + np.sqrt(max(var, 0.0)) * rng.standard_normal()
                root = np.sqrt(max(th, _FLOOR))
                sqth[j] = root if rng.random() < 0.5 else -root
                btil[:, j] = (centered[:, j] - beta[j]) / sqth[j]

            # sqrt_theta and the state are identified only jointly in sign.
            flip = rng.random(m) < 0.5
            sqth[flip] *= -1.0
            btil[:, flip] *= -1.0

        for j in range(m):
            al2[j] = 1.0 / _rinvgauss(
                np.sqrt(lam2 / max(beta[j] ** 2, _FLOOR)), lam2, rng)
        lam2 = rng.gamma(z1 + m, 1.0 / (z2 + 0.5 * al2.sum()))
        if time_varying:
            for j in range(m):
                xi2[j] = 1.0 / _rinvgauss(
                    np.sqrt(kap2 / max(sqth[j] ** 2, _FLOOR)), kap2, rng)
            kap2 = rng.gamma(z1_dyn + m, 1.0 / (z2_dyn + 0.5 * xi2.sum()))

        fitted = Xp @ beta
        if time_varying:
            fitted = fitted + np.sum(Xp * btil[1:] * sqth, axis=1)
        resid = yp - fitted
        sig2 = 1.0 / rng.gamma(a1 + T0 / 2.0,
                               1.0 / (a2 + 0.5 * float(resid @ resid)))
        a2 = rng.gamma(g0 + a1, 1.0 / (G0 + 1.0 / sig2))
        if time_varying:
            for j in range(m):
                P0[j] = 1.0 / rng.gamma(20.5, 1.0 / (19.0 + 0.5 * btil[0, j] ** 2))

        if it >= burn_in:
            mu = X @ beta
            if time_varying:
                # Carry the walk forward through the post window; the state is
                # unobserved there, so it diffuses and the band widens.
                path_all = np.zeros((T, m))
                path_all[:T0] = btil[1:]
                cur = btil[T0].copy()
                for t in range(T0, T):
                    cur = cur + rng.standard_normal(m)
                    path_all[t] = cur
                mu = mu + np.sum(X * path_all * sqth, axis=1)
            pred[:, k] = mu + np.sqrt(sig2) * rng.standard_normal(T)
            beta_d[:, k] = beta
            sqth_d[:, k] = sqth
            sig2_d[k] = sig2
            k += 1

    return pred, beta_d, sqth_d, sig2_d


def gibbs_bltvp(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    n_iter: int = 10000,
    burn_in: int = 5000,
    chains: int = 1,
    seed: Optional[int] = None,
    intercept: bool = False,
    time_varying: bool = True,
    a1: float = 2.5,
    g0: float = 5.0,
    G0: float = 5.0 / 1.5,
    z1: float = 0.001,
    z2: float = 0.001,
    z1_dyn: float = 0.001,
    z2_dyn: float = 0.001,
) -> Dict[str, np.ndarray]:
    """Draw from the BLTVP posterior.

    The model is fit on ``t < T0`` and the untreated potential outcome is
    predicted over every period, so the returned ``predictive`` covers the pre
    window (fitted) and the post window (forecast) alike.

    Parameters
    ----------
    y : np.ndarray
        Length-``T`` treated outcome.
    X : np.ndarray
        Shape ``(T, N)`` donor matrix.
    T0 : int
        Number of pre-treatment periods.
    n_iter, burn_in, chains : int
        Sampler settings. Chains are pooled into one set of draws.
    seed : int, optional
        Seed; chain ``c`` runs from ``seed + c`` so pooling stays reproducible.
    intercept : bool
        Append a column of ones to ``X``.
    time_varying : bool
        Allow the coefficients to drift. ``False`` pins every drift scale to
        zero, giving the static Bayesian Lasso nesting (paper, p. 1067).
    a1, g0, G0, z1, z2, z1_dyn, z2_dyn : float
        Prior hyperparameters; see :class:`BLTVPConfig`.

    Returns
    -------
    dict
        ``predictive`` ``(T, n_samples)``, ``beta`` and ``sqrt_theta``
        ``(m, n_samples)``, ``sigma2`` ``(n_samples,)``.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, N); got shape {X.shape}.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"len(y)={y.shape[0]} must equal X's row count {X.shape[0]}.")
    if not 2 <= T0 <= X.shape[0]:
        raise ValueError(f"T0={T0} must lie in [2, {X.shape[0]}].")
    if burn_in >= n_iter:
        raise ValueError(f"burn_in={burn_in} must be below n_iter={n_iter}.")

    if intercept:
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    parts = []
    for c in range(chains):
        rng = np.random.default_rng(None if seed is None else seed + c)
        parts.append(_one_chain(y, X, T0, n_iter, burn_in, rng, time_varying,
                                a1, g0, G0, z1, z2, z1_dyn, z2_dyn))
    return {
        "predictive": np.hstack([p[0] for p in parts]),
        "beta": np.hstack([p[1] for p in parts]),
        "sqrt_theta": np.hstack([p[2] for p in parts]),
        "sigma2": np.concatenate([p[3] for p in parts]),
    }
