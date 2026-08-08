"""Gibbs sampler for the dynamic multilevel latent factor model.

A port of ``pblasso`` 1.0.8, the package behind Pang, Liu & Xu (2022). Each
routine names the source it follows: ``blasso_core.R`` for the sweep,
``src/blasso.cpp`` for the individual draws. The reference tarball ships with
the authors' replication package; see
``benchmarks/reference/dmlfm_germany/README.md``.

The model, in the paper's notation (eqs. 5-7):

    y_it(0) = x_it' beta + z_it' (omega_a . alpha_i)
                         + a_it' (omega_xi . xi_t)
                         + (omega_g . gamma_i)' f_t + eps_it

with ``xi_t`` and ``f_t`` following independent AR(1) processes and Laplacian
shrinkage optionally applied to each of ``beta``, ``omega_a``, ``omega_xi`` and
``omega_g``. Shrinking ``omega_g`` toward zero switches a factor off, which is
how the factor count is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# elementary draws (src/blasso.cpp)
# --------------------------------------------------------------------------
def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """arma::mvnrnd. Symmetrised, with a jitter retry on a non-PSD covariance."""
    cov = 0.5 * (cov + cov.T)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical guard
        w, V = np.linalg.eigh(cov)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
    return mean + L @ rng.standard_normal(mean.shape[0])


def _group_moments(X: np.ndarray, Y: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """``genXY`` (blasso.cpp:148): stack ``X_g' Y_g`` over contiguous groups."""
    k = X.shape[1]
    g = len(starts)
    out = np.zeros((g * k, Y.shape[1]))
    ends = np.append(starts[1:], X.shape[0])
    for i in range(g):
        s, e = starts[i], ends[i]
        out[i * k:(i + 1) * k] = X[s:e].T @ Y[s:e]
    return out


def sample_normal(X: np.ndarray, y: np.ndarray, cov0: np.ndarray,
                  sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """``sampleN`` (blasso.cpp:214).

    Coefficient 0 is given a flat prior -- the loop over the prior precision
    starts at ``j = 1`` (blasso.cpp:224), so the intercept is unpenalised even
    when shrinkage is on.
    """
    k = X.shape[1]
    prec0 = np.zeros((k, k))
    for j in range(1, k):
        prec0[j, j] = 1.0 / cov0[j, j]
    S = np.linalg.inv(X.T @ X / sigma2 + prec0)
    mu = S @ (X.T @ y.ravel() / sigma2)
    if k == 1:
        return np.array([rng.normal(mu[0], np.sqrt(S[0, 0]))])
    return _mvnrnd(mu, S, rng)


def _conditional_normal(mu: np.ndarray, S: np.ndarray, tail: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """``sampleCN`` (blasso.cpp:250): draw the leading block given the tail."""
    n1, n2 = mu.shape[0], tail.shape[0]
    n3 = n1 - n2
    s11, s22 = S[:n3, :n3], S[n3:, n3:]
    s12, s21 = S[:n3, n3:], S[n3:, :n3]
    inv22 = np.linalg.inv(s22)
    m = mu[:n3] + s12 @ inv22 @ (tail - mu[n3:])
    s = s11 - s12 @ inv22 @ s21
    if n3 == 1:
        return np.array([rng.normal(m[0], np.sqrt(max(s[0, 0], 0.0)))])
    return _mvnrnd(m, s, rng)


def _sample_sub_alpha(mu: np.ndarray, S: np.ndarray, k2: int, r: int,
                      unit_index: int, rng: np.random.Generator) -> np.ndarray:
    """``sampleSubAlpha`` (blasso.cpp:293).

    Rotation of the factor space is pinned by a lower-triangular restriction:
    unit ``i`` (1-based) loads on at most the first ``i`` factors. The tail
    entries stay exactly zero and the leading block is drawn conditional on
    them. This makes the fit depend on the order of units, which is a property
    of the estimator and is reproduced here rather than smoothed away.
    """
    k = mu.shape[0]
    out = np.zeros(k)
    if r == 0:
        return _mvnrnd(mu, S, rng) if k > 1 else np.array(
            [rng.normal(mu[0], np.sqrt(S[0, 0]))])
    if k2 == 0:                                    # loadings only
        if r == 1 or unit_index > r:
            return _mvnrnd(mu, S, rng) if k > 1 else np.array(
                [rng.normal(mu[0], np.sqrt(S[0, 0]))])
        if unit_index == 1:
            out[0] = rng.normal(mu[0], np.sqrt(S[0, 0]))
            return out
        if unit_index <= r - 1:
            out[:unit_index] = _conditional_normal(mu, S, out[unit_index:], rng)
            return out
        return _mvnrnd(mu, S, rng)
    # unit-level random effects alongside the loadings
    if unit_index >= r:
        return _mvnrnd(mu, S, rng)
    cut = k - r + unit_index
    out[:cut] = _conditional_normal(mu, S, out[cut:], rng)
    return out


def sample_loadings(X: np.ndarray, y: np.ndarray, cov0: np.ndarray,
                    unit_starts: np.ndarray, n_units: int, k2: int, r: int,
                    sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """``sampleAlpha`` / ``iterGenAlpha`` (blasso.cpp:367, 413)."""
    k = X.shape[1]
    XY = _group_moments(X, y, unit_starts)
    XX = _group_moments(X, X, unit_starts)
    inv0 = np.linalg.inv(cov0)
    out = np.zeros((n_units, k))
    for i in range(n_units):
        A1 = np.linalg.inv(XX[i * k:(i + 1) * k] / sigma2 + inv0)
        mu = A1 @ XY[i * k:(i + 1) * k].ravel() / sigma2
        out[i] = _sample_sub_alpha(mu, A1, k2, r, i + 1, rng)
    return out


def sample_time_states(X: np.ndarray, y: np.ndarray, tau_old: np.ndarray,
                       phi: np.ndarray, time_starts: np.ndarray, n_periods: int,
                       sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """``sampleXi`` / ``iterGenXi`` (blasso.cpp:619, 694).

    A single-site sweep over ``t`` under the AR(1) prior
    ``tau_t ~ N(Phi tau_{t-1}, I)`` with ``tau_0 = 0``. Both neighbours are read
    from ``tau_old``, never from the freshly drawn value, so the sweep is
    Jacobi-style; and the first period conditions on its forward neighbour only.
    """
    k = X.shape[1]
    XY = _group_moments(X, y, time_starts)
    XX = _group_moments(X, X, time_starts)
    I = np.eye(k)
    mphi = np.diag(phi.ravel())
    pp = mphi @ mphi
    out = np.zeros((n_periods, k))
    for t in range(n_periods):
        xx = XX[t * k:(t + 1) * k]
        xy = XY[t * k:(t + 1) * k].ravel()
        if t < n_periods - 1:
            prec = I + xx / sigma2 + pp
            mu = xy / sigma2
            if t == 0:
                mu = mu + mphi @ tau_old[1]
            else:
                mu = mu + mphi @ (tau_old[t - 1] + tau_old[t + 1])
        else:
            prec = I + xx / sigma2
            mu = xy / sigma2 + mphi @ tau_old[t - 1]
        S = np.linalg.inv(prec)
        out[t] = _mvnrnd(S @ mu, S, rng)
    return out


def sample_phi(tau: np.ndarray, prior_var: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """``samplePhi`` (blasso.cpp:717). Independent normal draw per series.

    The draw is not truncated to the stationary region, matching the reference.
    """
    lag = np.vstack([np.zeros((1, tau.shape[1])), tau[:-1]])
    xx = lag.T @ lag
    xy = lag.T @ tau
    k = tau.shape[1]
    out = np.zeros(k)
    for i in range(k):
        v = 1.0 / (xx[i, i] + 1.0 / prior_var[i])
        out[i] = rng.normal(v * xy[i, i], np.sqrt(v))
    return out


def sample_sigma2(res: np.ndarray, c0: float, d0: float,
                  rng: np.random.Generator) -> float:
    """``sampleSigmaE2`` (blasso.cpp:785). Inverse-gamma via a gamma draw."""
    c1 = c0 + res.shape[0]
    d1 = d0 + float(np.sum(res ** 2))
    return 1.0 / rng.gamma(c1 / 2.0, 2.0 / d1)


def _rinvgauss(mu: float, lam: float, rng: np.random.Generator) -> float:
    """``rrinvgauss`` (blasso.cpp:820), Michael-Schucany-Haas."""
    z = rng.standard_normal()
    y = z * z
    x = mu + 0.5 * mu * mu * y / lam - 0.5 * (mu / lam) * np.sqrt(
        4 * mu * lam * y + mu * mu * y * y)
    return x if rng.random() <= mu / (mu + x) else mu * mu / x


def _lasso_update(coef: np.ndarray, lam2: float, prior_var: np.ndarray,
                  shape: float, rate: float, rng: np.random.Generator) -> float:
    """One Bayesian-lasso block: local variances then the global penalty.

    Mirrors the ``xlasso``/``zlasso``/``alasso``/``flasso`` blocks of
    ``blasso_core.R`` (lines 235-251 and 367-444).
    """
    for j in range(coef.shape[0]):
        c2 = max(float(coef[j]) ** 2, 1e-12)
        prior_var[j] = 1.0 / _rinvgauss(np.sqrt(lam2 / c2), lam2, rng)
    return rng.gamma(shape + coef.shape[0], 1.0 / (rate + prior_var.sum() / 2.0))


def _sign_flip(omega: np.ndarray, states: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``permute`` (blasso.cpp:454).

    A random sign per column, applied to both the scale and its state series.
    The product is invariant, so this moves between the equivalent modes without
    changing the fit -- which is why only ``|omega|`` is comparable across runs.
    """
    coef = np.where(rng.standard_normal(states.shape[1]) < 0, -1.0, 1.0)
    return omega * coef, states * coef[None, :]


# --------------------------------------------------------------------------
@dataclass
class DMLFMDraws:
    """Retained posterior draws."""

    counterfactual: np.ndarray                  # (n_treated_rows, ndraw)
    sigma2: np.ndarray
    beta: np.ndarray
    omega_gamma: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    gamma: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    factors: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))
    omega_xi: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    phi: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def run_gibbs(inputs, rng: np.random.Generator) -> DMLFMDraws:
    """The sweep of ``bayesLasso`` (blasso_core.R:225-549).

    ``inputs`` is a :class:`~mlsynth.utils.dmlfm_helpers.setup.DMLFMInputs`.
    """
    X, Z, A = inputs.X, inputs.Z, inputs.A
    Xt, Zt, At = inputs.X_tr, inputs.Z_tr, inputs.A_tr
    y = inputs.y.reshape(-1, 1)
    n_units, n_periods, r = inputs.n_units, inputs.n_periods, inputs.r
    k1, k2, k3 = X.shape[1], Z.shape[1], A.shape[1]
    unit_of, time_of = inputs.unit_index, inputs.time_index
    unit_tr, time_tr = inputs.unit_index_tr, inputs.time_index_tr
    nobs = y.shape[0]

    beta = rng.standard_normal(k1)
    Alpha = rng.standard_normal((n_units, k2)) if k2 else np.zeros((n_units, 0))
    Xi = rng.standard_normal((n_periods, k3)) if k3 else np.zeros((n_periods, 0))
    Gamma = rng.standard_normal((n_units, r)) if r else np.zeros((n_units, 0))
    F = rng.standard_normal((n_periods, r)) if r else np.zeros((n_periods, 0))

    omega_a = np.ones(k2)
    omega_xi = np.ones(k3)
    omega_g = np.ones(r)
    phi = np.zeros(k3 + r)
    prior_var_phi = np.full(k3 + r, 100.0)

    var_beta = np.full(k1, 100.0)
    var_a, var_xi, var_g = (np.full(k2, 100.0), np.full(k3, 100.0), np.full(r, 100.0))
    lam_b = lam_a = lam_xi = lam_g = 10.0

    def re_fit(design, eff, index):
        return np.einsum("ij,ij->i", design, eff[index]) if design.shape[1] else np.zeros(
            design.shape[0])

    def factor_fit(g, f, iu, it):
        return np.einsum("ij,ij->i", g[iu], f[it]) if r else np.zeros(iu.shape[0])

    xfit = X @ beta
    zfit = re_fit(Z, Alpha * omega_a, unit_of)
    afit = re_fit(A, Xi * omega_xi, time_of)
    ffit = factor_fit(Gamma * omega_g, F, unit_of, time_of)
    sigma2 = sample_sigma2(y.ravel() - xfit - zfit - afit - ffit,
                           inputs.e1, inputs.e2, rng)

    tau = np.hstack([Xi, F]) if k3 + r else np.zeros((n_periods, 0))
    keep = inputs.niter - inputs.burn
    d_cf = np.zeros((Xt.shape[0], keep))
    d_sig = np.zeros(keep)
    d_beta = np.zeros((k1, keep))
    d_wg = np.zeros((r, keep))
    d_wxi = np.zeros((k3, keep))
    d_gamma = np.zeros((n_units, r, keep))
    d_f = np.zeros((n_periods, r, keep))
    d_phi = np.zeros((k3 + r, keep))

    for it in range(inputs.niter):
        keep_it = it >= inputs.burn
        j = it - inputs.burn

        # 1. constant coefficients
        res = y.ravel() - zfit - afit - ffit
        beta = sample_normal(X, res, np.diag(var_beta), sigma2, rng)
        xfit = X @ beta
        if inputs.xlasso:
            lam_b = _lasso_update(beta, lam_b, var_beta, inputs.a1, inputs.a2, rng)

        # 2. unit-level effects and factor loadings, jointly
        if k2 + r:
            res = y.ravel() - xfit - afit
            tZ = _tilde(Z, F, np.concatenate([omega_a, omega_g]), time_of)
            drawn = sample_loadings(tZ, res.reshape(-1, 1), np.eye(k2 + r),
                                    inputs.unit_starts, n_units, k2, r, sigma2, rng)
            if k2:
                Alpha = drawn[:, :k2]
                zfit = re_fit(Z, Alpha * omega_a, unit_of)
            if r:
                Gamma = drawn[:, k2:k2 + r]

        # 3. time-level effects and factors, jointly, under the AR(1) prior
        if k3 + r:
            res = (y.ravel() - xfit - zfit)[inputs.order_by_time]
            tA = _tilde(A, Gamma, np.concatenate([omega_xi, omega_g]),
                        unit_of)[inputs.order_by_time]
            tau = sample_time_states(tA, res.reshape(-1, 1), tau, phi,
                                     inputs.time_starts, n_periods, sigma2, rng)
            if k3:
                Xi = tau[:, :k3]
            if r:
                F = tau[:, k3:k3 + r]

        # 4. the scales, then shrinkage, then the sign move
        if k2 + k3 + r:
            res = y.ravel() - xfit
            tTau = _tilde_tau(Z, A, Alpha, Xi, Gamma, F, unit_of, time_of)
            cov0 = np.diag(np.concatenate([var_a, var_xi, var_g]))
            omega = sample_normal(tTau, res, cov0, sigma2, rng)
            if k2:
                omega_a = omega[:k2]
                if inputs.zlasso:
                    lam_a = _lasso_update(omega_a, lam_a, var_a, inputs.b1, inputs.b2, rng)
                omega_a, Alpha = _sign_flip(omega_a, Alpha, rng)
            if k3:
                omega_xi = omega[k2:k2 + k3]
                if inputs.alasso:
                    lam_xi = _lasso_update(omega_xi, lam_xi, var_xi,
                                           inputs.c1, inputs.c2, rng)
                omega_xi, Xi = _sign_flip(omega_xi, Xi, rng)
            if r:
                omega_g = omega[k2 + k3:]
                if inputs.flasso:
                    lam_g = _lasso_update(omega_g, lam_g, var_g, inputs.p1, inputs.p2, rng)
                omega_g, F = _sign_flip(omega_g, F, rng)

        # 5. AR(1) coefficients.
        # ``tau`` stays as ``sample_time_states`` returned it. The sign move in
        # step 4 flips Xi and F for the fit only; blasso_core.R:483 draws Phi
        # from the pre-flip Tau and carries that same Tau into the next sweep,
        # so rebuilding tau from the flipped blocks would feed sign noise into
        # the AR(1) history.
        if inputs.ar1 and k3 + r:
            phi = sample_phi(tau, prior_var_phi, rng)

        # 6. refit and the error variance
        zfit = re_fit(Z, Alpha * omega_a, unit_of)
        afit = re_fit(A, Xi * omega_xi, time_of)
        ffit = factor_fit(Gamma * omega_g, F, unit_of, time_of)
        sigma2 = sample_sigma2(y.ravel() - xfit - zfit - afit - ffit,
                               inputs.e1, inputs.e2, rng)

        # 7. the treated counterfactual, as a posterior predictive draw
        cf = Xt @ beta
        cf = cf + re_fit(Zt, Alpha * omega_a, unit_tr)
        cf = cf + re_fit(At, Xi * omega_xi, time_tr)
        cf = cf + factor_fit(Gamma * omega_g, F, unit_tr, time_tr)
        cf = cf + rng.normal(0.0, np.sqrt(sigma2), cf.shape[0])

        if keep_it:
            d_cf[:, j] = cf
            d_sig[j] = sigma2
            d_beta[:, j] = beta
            if r:
                d_wg[:, j] = omega_g
                d_gamma[:, :, j] = Gamma
                d_f[:, :, j] = F
            if k3:
                d_wxi[:, j] = omega_xi
            if k3 + r:
                d_phi[:, j] = phi

    return DMLFMDraws(counterfactual=d_cf, sigma2=d_sig, beta=d_beta,
                      omega_gamma=d_wg, gamma=d_gamma, factors=d_f,
                      omega_xi=d_wxi, phi=d_phi)


def _tilde(block: np.ndarray, states: np.ndarray, omega: np.ndarray,
           index: np.ndarray) -> np.ndarray:
    """``genTildeZ`` / ``genTildeA`` (blasso.cpp:520, 552)."""
    expanded = states[index] if states.shape[1] else np.zeros((index.shape[0], 0))
    out = np.hstack([block, expanded]) if block.shape[1] else expanded
    return out * omega[None, :]


def _tilde_tau(Z, A, Alpha, Xi, Gamma, F, unit_of, time_of) -> np.ndarray:
    """``genTildeTau`` (blasso.cpp:583): the design for the omega draw."""
    parts = []
    if Z.shape[1]:
        parts.append(Z * Alpha[unit_of])
    if A.shape[1]:
        parts.append(A * Xi[time_of])
    if Gamma.shape[1]:
        parts.append(Gamma[unit_of] * F[time_of])
    return np.hstack(parts) if parts else np.zeros((unit_of.shape[0], 0))
