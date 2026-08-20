"""BL-TVP: Klinenberg (2023, JBES 41:4, 1065-1076) Model 1, ported from the paper.

The readable oracle for ``benchmarks/cases/bltvp_prop99.py``. Kept here, not in
``mlsynth/``, because this is a replication spike: it establishes that the method
reproduces before any estimator is built to the library contract.

Noncentered state space (eqs 10-13):

    y_t = sum_j (beta_j + btil_{j,t} sqrt(theta)_j) x_{j,t} + eps_t,  eps ~ N(0, sig2)
    btil_{j,t} = btil_{j,t-1} + eta_t,  eta ~ N(0, 1),  btil_{j,0} ~ N(0, P0_jj)

Signing the root of theta so that zero is an interior point of its prior is what
makes the shrinkage work; an inverse gamma on theta cannot place mass at zero,
which is why CausalImpact's dynamic-regression mode forces every coefficient to
be dynamic and returns the intervals the paper is reacting to.

Gibbs, per sec 3 and appendix A.1:

  1. FFBS for btil_{0:T0}                        (McCausland-Miller-Pelletier)
  2. joint conjugate draw of (beta, sqrt_theta)  -- in the noncentered form the
     two blocks are one linear regression on [X, X*btil]
  3. ASIS interweave through the centered parameterisation (Yu-Meng 2011), with
     the sign switch that resolves the sqrt(theta)/btil label switching
  4. hyperparameters, eqs (25)-(31)

Hyperparameters default to shrinkTVP's, which is what the author's
``bitto_model_revised.R`` calls with ``a_xi = a_tau = 1`` and learning off --
the setting that reduces the double gamma to the Bayesian Lasso.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import invgauss, geninvgauss


def _sym(M):
    return 0.5 * (M + M.T)


def _rgig(p, a, b, rng):
    """GIG(p,a,b): density prop to x^{p-1} exp(-(a x + b/x)/2)."""
    a = max(float(a), 1e-12); b = max(float(b), 1e-12)
    return float(geninvgauss.rvs(p, np.sqrt(a * b), scale=np.sqrt(b / a),
                                 random_state=rng))


def _rinvgauss(mu, lam, rng):
    return float(invgauss.rvs(mu / lam, scale=lam, random_state=rng))


def ffbs(y_star, H, sig2, P0, rng):
    """Sample btil_{0:T} | rest.  y_star_t = H[t] @ btil_t + N(0, sig2)."""
    T, m = H.shape
    a = np.zeros(m); R = np.diag(P0)
    a_f = np.zeros((T + 1, m)); R_f = np.zeros((T + 1, m, m))
    R_p = np.zeros((T + 1, m, m))
    a_f[0] = a; R_f[0] = R
    for t in range(1, T + 1):
        Rp = _sym(R_f[t - 1] + np.eye(m))          # identity transition, unit Q
        R_p[t] = Rp
        h = H[t - 1]
        Q = float(h @ Rp @ h) + sig2
        K = (Rp @ h) / Q
        e = y_star[t - 1] - float(h @ a_f[t - 1])
        a_f[t] = a_f[t - 1] + K * e
        R_f[t] = _sym(Rp - np.outer(K, K) * Q)
    btil = np.zeros((T + 1, m))
    L = np.linalg.cholesky(_sym(R_f[T]) + 1e-10 * np.eye(m))
    btil[T] = a_f[T] + L @ rng.standard_normal(m)
    for t in range(T - 1, -1, -1):
        Rt = _sym(R_f[t]); Rp = R_p[t + 1]
        G = np.linalg.solve(Rp, Rt).T              # Rt @ inv(Rp)
        mean = a_f[t] + G @ (btil[t + 1] - a_f[t])
        V = _sym(Rt - G @ Rt)
        w, U = np.linalg.eigh(V)
        w = np.clip(w, 0.0, None)
        btil[t] = mean + (U * np.sqrt(w)) @ rng.standard_normal(m)
    return btil


def bltvp(y, X, T0, *, niter=10000, nburn=5000, seed=0,
          a1=2.5, g0=5.0, G0=5.0 / 1.5,
          z1=0.001, z2=0.001, z1p=0.001, z2p=0.001,
          asis=True, sign_switch=True, sig2_full=True):
    """Fit on t=0..T0-1, forecast t=T0..T-1.  Returns (T, ndraws) predictive draws."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    T, m = X.shape
    Xp, yp = X[:T0], y[:T0]

    beta = np.zeros(m); sqth = np.zeros(m)
    btil = np.zeros((T0 + 1, m))
    sig2 = float(np.var(yp)) or 1.0
    a2 = 1.0
    al2 = np.ones(m); xi2 = np.ones(m)
    lam2 = 1.0; kap2 = 1.0
    P0 = np.ones(m)

    keep = niter - nburn
    out = np.zeros((T, keep))
    k = 0
    for it in range(niter):
        # --- 1. states -------------------------------------------------------
        H = Xp * sqth                                   # h_{j,t} = sqrt(theta)_j x_{j,t}
        btil = ffbs(yp - Xp @ beta, H, sig2, P0, rng)

        # --- 2. (beta, sqrt_theta) jointly ----------------------------------
        Z = np.hstack([Xp, Xp * btil[1:]])
        prior = np.concatenate([al2, xi2])
        A = Z.T @ Z / sig2 + np.diag(1.0 / prior)
        L = np.linalg.cholesky(_sym(A))
        bhat = np.linalg.solve(A, Z.T @ yp / sig2)
        draw = bhat + np.linalg.solve(L.T, rng.standard_normal(2 * m))
        beta, sqth = draw[:m], draw[m:]

        # --- 3. ASIS: redraw (beta, theta) in the centered parameterisation --
        if asis:
            bc = beta + sqth * btil[1:]                 # centered path, t=1..T0
            bc0 = beta + sqth * btil[0]
            for j in range(m):
                d = np.diff(np.concatenate([[bc0[j]], bc[:, j]]))
                S = float(d @ d) + (bc0[j] - beta[j]) ** 2 / P0[j]
                th = _rgig(-(T0 - 1) / 2.0, 1.0 / xi2[j], max(S, 1e-12), rng)
                v = al2[j] * th * P0[j] / (al2[j] + th * P0[j])
                mu = bc0[j] * al2[j] / (al2[j] + th * P0[j])
                beta[j] = mu + np.sqrt(max(v, 0.0)) * rng.standard_normal()
                s = np.sqrt(max(th, 1e-300))
                sqth[j] = s if rng.random() < 0.5 else -s
                btil[:, j] = (np.concatenate([[bc0[j]], bc[:, j]]) - beta[j]) / sqth[j]

        # --- sign switch (the sqrt(theta), btil label-switching move) --------
        if sign_switch:
            flip = rng.random(m) < 0.5
            sqth[flip] *= -1.0
            btil[:, flip] *= -1.0

        # --- 4. hyperparameters, eqs (25)-(31) -------------------------------
        for j in range(m):
            al2[j] = 1.0 / _rinvgauss(np.sqrt(lam2 / max(beta[j] ** 2, 1e-12)), lam2, rng)
            xi2[j] = 1.0 / _rinvgauss(np.sqrt(kap2 / max(sqth[j] ** 2, 1e-12)), kap2, rng)
        lam2 = rng.gamma(z1 + m, 1.0 / (z2 + 0.5 * al2.sum()))
        kap2 = rng.gamma(z1p + m, 1.0 / (z2p + 0.5 * xi2.sum()))
        resid = yp - (Xp @ beta + np.sum(Xp * btil[1:] * sqth, axis=1))
        n_eff = T0 if sig2_full else T0 - 1
        r = resid if sig2_full else resid[:T0 - 1]
        sig2 = 1.0 / rng.gamma(a1 + n_eff / 2.0, 1.0 / (a2 + 0.5 * float(r @ r)))
        a2 = rng.gamma(g0 + a1, 1.0 / (G0 + 1.0 / sig2))
        for j in range(m):
            P0[j] = 1.0 / rng.gamma(20.5, 1.0 / (19.0 + 0.5 * btil[0, j] ** 2))

        # --- store posterior predictive over all T periods --------------------
        if it >= nburn:
            path = np.zeros((T, m))
            path[:T0] = btil[1:]
            cur = btil[T0].copy()
            for t in range(T0, T):
                cur = cur + rng.standard_normal(m)
                path[t] = cur
            mu = X @ beta + np.sum(X * path * sqth, axis=1)
            out[:, k] = mu + np.sqrt(sig2) * rng.standard_normal(T)
            k += 1
    return out
