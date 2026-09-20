"""Python port of Goh & Yu (2022) Bayes SCM, validated on the Basque application.

Algorithm 1:1 from the authors' sec6_Empirical_Application.R:

  MAP (Monte Carlo EM)
    M-step: min_w  sum_k v_k (X1 - X0 w)_k^2   s.t.  sum_{i>=2} w_i = 1,
            w_i >= 0 for i >= 2, w_1 free          -- the W_ps-conv constraint
    E-step: Gibbs over (nu, xi) at the current w, then v = (1/nu_bar) * xi_bar
            with xi fixed at 1 on the T0 outcome rows and sampled on the p
            covariate rows (spike-and-slab, prob0 = 0.5).

  Inference (Gibbs)
    w_1     ~ N,          the free intercept
    w_i     ~ TN[0, U_i], for active donors other than the last
    w_M     = 1 - sum of the others
    nu      ~ InvGamma,   xi ~ Bernoulli

Run against the R-dumped design so the algorithm is compared, not the dataprep.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm

C0 = D0 = 0.5
PROB0 = 0.5


def _solve_map_step(X1, X0, v):
    """M-step: the weighted QP over W_ps-conv. Returns w of length N."""
    import cvxpy as cp

    N = X0.shape[1]
    w = cp.Variable(N)
    r = cp.multiply(np.sqrt(v), X1 - X0 @ w)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(r)),
                      [cp.sum(w[1:]) == 1, w[1:] >= 0])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(w.value, float)


def _e_step(X1, X0, w, T0, p, rng, n_draws=15000, burn=5000):
    """MC-E-step: Gibbs over (nu, xi) at fixed w; return xi_bar and E[1/nu]."""
    resid2 = (X1 - X0 @ w) ** 2
    xi = np.ones(p)
    xi_all = np.concatenate([np.ones(T0), xi])
    nu_a = T0 / 2.0 + C0
    keep_xi = np.empty((n_draws, T0 + p))
    keep_inv_nu = np.empty(n_draws)
    Xc, X1c = X0[T0:], X1[T0:]
    for i in range(n_draws):
        nu_b = 0.5 * float(xi_all @ resid2) + D0
        nu = 1.0 / rng.gamma(shape=nu_a, scale=1.0 / nu_b)
        # phi = N(X1_cov | X0_cov w, nu); xi_prob = p0 / ((1-p0)/phi + p0)
        mu = Xc @ w
        phi = np.exp(-0.5 * (X1c - mu) ** 2 / nu) / np.sqrt(2 * np.pi * nu)
        with np.errstate(divide="ignore", over="ignore"):
            xi_prob = PROB0 / ((1 - PROB0) / phi + PROB0)
        xi = rng.binomial(1, np.clip(np.nan_to_num(xi_prob, nan=1.0), 0.0, 1.0))
        xi_all = np.concatenate([np.ones(T0), xi])
        keep_xi[i] = xi_all
        keep_inv_nu[i] = 1.0 / nu
    return keep_xi[burn:].mean(axis=0), float(keep_inv_nu[burn:].mean())


def map_estimate(X1, X0, T0, p, rng, max_em=1000, tol=1e-4):
    """The MAP weights under W_ps-conv, with V marginalised by Monte Carlo EM."""
    N = X0.shape[1]
    inv_nu = 1.0
    xi_all = np.ones(T0 + p)
    w = np.zeros(N)
    naive = None
    for r in range(1, max_em + 1):
        w_prev = w
        w = _solve_map_step(X1, X0, inv_nu * xi_all)
        if r == 1:
            naive = w.copy()          # the paper's "naive Bayes SCM"
        if np.max(np.abs(w_prev - w)) < tol:
            break
        xi_all, inv_nu = _e_step(X1, X0, w, T0, p, rng)
    return w, naive, r, xi_all, inv_nu


def gibbs(X1, X0, w_map, xi_all, inv_nu, T0, p, Y0, rng,
          n_draws=20000, burn=5000):
    """Posterior draws of the counterfactual path, on the MAP's active set."""
    N = X0.shape[1]
    T = Y0.shape[0]
    active = np.flatnonzero(np.round(w_map, 5) != 0)
    omega = np.zeros(N)
    donors = active[active != 0]
    omega[donors] = w_map[donors] / w_map[donors].sum()
    omega[0] = w_map[0]
    M = int(active.max())
    nu_a = T0 / 2.0 + C0
    Xc, X1c = X0[T0:], X1[T0:]
    path = np.empty((n_draws, T))
    pred = np.empty((n_draws, T))
    for it in range(n_draws):
        v = inv_nu * xi_all
        # intercept
        s2 = 1.0 / float(v @ (X0[:, 0] ** 2))
        mu = float(X0[:, 0] @ (v * (X1 - X0[:, 1:] @ omega[1:]))) * s2
        omega[0] = rng.normal(mu, np.sqrt(s2))
        # donors, in the authors' order; the last active donor absorbs the rest
        for i in donors:
            if i == M:
                omega[i] = 1.0 - sum(omega[j] for j in donors if j != i)
                continue
            xs = X0[:, i] - X0[:, M]
            s2 = 1.0 / float(v @ (xs ** 2))
            other = [j for j in range(1, N) if j not in (i, M)]
            resid = (X1 - X0[:, M] - omega[0] * X0[:, 0]
                     - (X0[:, other] - X0[:, [M]]) @ omega[other])
            mu = float(xs @ (v * resid)) * s2
            ub = 1.0 - sum(omega[j] for j in donors if j not in (i, M))
            sd = np.sqrt(s2)
            if ub <= 0:
                omega[i] = 0.0
            else:
                omega[i] = float(truncnorm.rvs(
                    (0.0 - mu) / sd, (ub - mu) / sd, loc=mu, scale=sd,
                    random_state=rng))
        # nu
        nu_b = 0.5 * float(xi_all @ ((X1 - X0 @ omega) ** 2)) + D0
        nu = 1.0 / rng.gamma(shape=nu_a, scale=1.0 / nu_b)
        inv_nu = 1.0 / nu
        # xi
        mu_c = Xc @ omega
        phi = np.exp(-0.5 * (X1c - mu_c) ** 2 / nu) / np.sqrt(2 * np.pi * nu)
        with np.errstate(divide="ignore", over="ignore"):
            xi_prob = PROB0 / ((1 - PROB0) / phi + PROB0)
        xi = rng.binomial(1, np.clip(np.nan_to_num(xi_prob, nan=1.0), 0.0, 1.0))
        xi_all = np.concatenate([np.ones(T0), xi])

        path[it] = Y0 @ omega[1:] + omega[0]
        pred[it] = rng.normal(path[it], np.sqrt(nu))
    return path[burn:], pred[burn:]


def hdi(x, cred=0.95):
    """Highest-density interval: the narrowest window holding ``cred`` of the draws."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    m = max(2, int(np.ceil(cred * n)))
    widths = x[m - 1:] - x[: n - m + 1]
    k = int(np.argmin(widths))
    return float(x[k]), float(x[k + m - 1])
