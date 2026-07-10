"""Pure-numpy batched-chains Gibbs sampler for the two BSCM priors.

Kim, Lee & Gupta (2020) propose two Bayesian synthetic-control priors on the
unconstrained donor weights:

* ``horseshoe`` -- the global-local shrinkage prior of Carvalho, Polson &
  Scott (2010), sampled here with the Makalic & Schmidt (2016) auxiliary-
  variable representation so every full conditional is inverse-gamma / normal;
* ``spike_slab`` -- the discrete mixture (George & McCulloch 1993), sampled
  with conjugate updates and a marginalised inclusion draw.

All ``chains`` chains advance in one vectorised loop via stacked Cholesky /
triangular solves. At synthetic-control scale the per-iteration cost is
Python/numpy call overhead rather than FLOPs, so the chains are close to free
relative to a single chain, and multiple chains give convergence diagnostics.

The counterfactual is ``beta_0 + X @ beta`` (intercept explicit), matching the
reference Stan model's ``generated quantities`` block. Cross-validated against
the authors' Stan implementation on the Basque panel (see
``docs/replications/bscm.rst``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_INTERCEPT_PREC = 1e-6   # diffuse Normal prior on the intercept (near-flat)
_FLOOR = 1e-10
_CEIL = 1e10


def _ig(shape, scale, rng):
    """Inverse-gamma(shape, scale) draw with a floored Gamma to avoid overflow."""
    return scale / np.maximum(rng.gamma(shape, 1.0, size=np.shape(scale)), _FLOOR)


def _draw_beta(XtX, Xty, di, prior_prec, sigma2, rng, C, q):
    """Batched beta draw for ``C`` chains.

    ``prior_prec`` is ``(C, q)``, ``sigma2`` is ``(C,)``; returns ``(C, q)``.
    Solves ``beta | . ~ N(A^-1 X'y/sigma2, A^-1)`` with
    ``A = X'X/sigma2 + diag(prior_prec)`` via a stacked Cholesky.
    """
    A = XtX[None] / sigma2[:, None, None]
    A[:, di, di] += prior_prec
    L = np.linalg.cholesky(A)                       # A = L L^T
    m = np.linalg.solve(A, (Xty[None] / sigma2[:, None])[..., None])[..., 0]
    z = rng.standard_normal((C, q, 1))
    s = np.linalg.solve(np.transpose(L, (0, 2, 1)), z)[..., 0]   # L^{-T} z, cov = A^{-1}
    return m + s


def _setup(y, X, X_all, C):
    n, p = X.shape
    q = p + 1
    Xa = np.empty((n, q)); Xa[:, 0] = 1.0; Xa[:, 1:] = X
    Xall = np.empty((X_all.shape[0], q)); Xall[:, 0] = 1.0; Xall[:, 1:] = X_all
    return n, p, q, Xa, Xall, Xa.T @ Xa, Xa.T @ y, np.arange(q)


def gibbs_bscm(
    y_pre: np.ndarray,
    X_pre: np.ndarray,
    X_all: np.ndarray,
    prior: str = "horseshoe",
    chains: int = 4,
    n_iter: int = 2000,
    burn_in: int = 1000,
    spike_scale: float = 0.001 ** 0.5,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Draw posterior samples of the BSCM regression coefficients.

    Parameters
    ----------
    y_pre : np.ndarray
        Length-``T0`` treated pre-treatment outcome.
    X_pre : np.ndarray
        Shape ``(T0, N)`` donor matrix over the pre-treatment window.
    X_all : np.ndarray
        Shape ``(T, N)`` donor matrix over all periods (unused by the sampler
        itself; accepted so callers can pass a single object -- the
        counterfactual is formed downstream in ``compute_inference``).
    prior : {"horseshoe", "spike_slab"}
        Shrinkage prior on the donor weights.
    chains : int
        Number of chains sampled together and pooled.
    n_iter, burn_in : int
        Iterations per chain and warm-up discarded per chain.
    spike_scale : float
        Standard deviation of the spike component (``spike_slab`` only).
    rng : numpy.random.Generator, optional
        Source of randomness; a fresh default generator is used if omitted.

    Returns
    -------
    dict
        ``beta`` ``(N, n_samples)``, ``beta0`` ``(n_samples,)``,
        ``sigma2`` ``(n_samples,)``, and ``gamma`` ``(N, n_samples)`` for
        ``spike_slab`` (else ``None``), where ``n_samples = chains *
        (n_iter - burn_in)``.
    """
    if prior not in ("horseshoe", "spike_slab"):
        raise ValueError(f"unknown prior {prior!r}")
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y_pre, dtype=float)
    C = int(chains)
    n, p, q, Xa, _Xall, XtX, Xty, di = _setup(y, np.asarray(X_pre, float), X_all, C)
    keep = n_iter - burn_in
    beta_draws = np.empty((C, keep, q))
    sig_draws = np.empty((C, keep))
    sigma2 = np.full(C, float(np.var(y)) or 1.0)

    if prior == "horseshoe":
        # The paper's hierarchy -- beta_j ~ N(0, lam_j^2), lam_j ~ C+(0, tau),
        # tau ~ C+(0, sigma), sigma ~ C+(0, 10) -- folds to the canonical
        # sigma^2-scaled horseshoe regression beta_j ~ N(0, sigma^2 tau^2 lam_j^2)
        # with tau, lam ~ C+(0, 1). That sigma^2 scaling *is* the tau ~ C+(0, sigma)
        # coupling, so the global shrinkage tracks the residual scale. Sampled
        # via the Makalic & Schmidt (2016) auxiliary variables (all full
        # conditionals inverse-gamma / normal). Since the prior precision on the
        # donors is 1/(sigma^2 tau^2 lam^2), _draw_beta's A = X'X/sigma^2 + diag(.)
        # equals A0/sigma^2 with A0 = X'X + diag(1/(tau^2 lam^2)), so the draw has
        # covariance sigma^2 A0^{-1} as required, with no extra machinery.
        lam2 = np.ones((C, p)); nu = np.ones((C, p))
        tau2 = np.ones(C); xi = np.ones(C); zeta = np.ones(C)
        for it in range(n_iter):
            pp = np.empty((C, q)); pp[:, 0] = _INTERCEPT_PREC
            pp[:, 1:] = 1.0 / (sigma2[:, None] * tau2[:, None] * lam2)
            beta = _draw_beta(XtX, Xty, di, pp, sigma2, rng, C, q)
            b = beta[:, 1:]; resid = y[None] - beta @ Xa.T
            rss = np.einsum("cn,cn->c", resid, resid)
            bpen = np.sum(b * b / (tau2[:, None] * lam2), 1)   # sum beta^2 / (tau^2 lam^2)
            # sigma ~ C+(0, 10) via the auxiliary zeta; sigma^2 also regularises
            # beta (the shrinkage penalty enters its scale).
            sigma2 = _ig((n + p + 1) / 2, 0.5 * (rss + bpen) + 1.0 / zeta, rng)
            zeta = _ig(1.0, 1.0 / sigma2 + 1.0 / 100.0, rng)
            s2 = sigma2[:, None]
            lam2 = np.clip(_ig(1.0, 1.0 / nu + b * b / (2 * tau2[:, None] * s2), rng), _FLOOR, _CEIL)
            nu = np.clip(_ig(1.0, 1.0 + 1.0 / lam2, rng), _FLOOR, _CEIL)
            tau2 = np.clip(_ig((p + 1) / 2, 1.0 / xi + np.sum(b * b / lam2, 1) / (2 * sigma2), rng),
                           _FLOOR, _CEIL)
            xi = np.clip(_ig(1.0, 1.0 + 1.0 / tau2, rng), _FLOOR, _CEIL)
            if it >= burn_in:
                beta_draws[:, it - burn_in] = beta
                sig_draws[:, it - burn_in] = sigma2
        gamma = None
    else:
        spike = float(spike_scale) ** 2
        tau2 = np.ones((C, p)); incl = rng.random((C, p)) < 0.5
        gamma_draws = np.empty((C, keep, p))
        for it in range(n_iter):
            slab = np.where(incl, tau2, spike)
            pp = np.empty((C, q)); pp[:, 0] = _INTERCEPT_PREC; pp[:, 1:] = 1.0 / slab
            beta = _draw_beta(XtX, Xty, di, pp, sigma2, rng, C, q)
            b = beta[:, 1:]; resid = y[None] - beta @ Xa.T
            sigma2 = _ig(n / 2 + 1e-3, 0.5 * np.einsum("cn,cn->c", resid, resid) + 1e-3, rng)
            tau2 = _ig(1.0, 0.5 + b * b / 2, rng)          # slab variance ~ IG(1/2, 1/2)
            ll_in = -0.5 * np.log(tau2) - b * b / (2 * tau2)
            ll_out = -0.5 * np.log(spike) - b * b / (2 * spike)
            pin = 1.0 / (1.0 + np.exp(np.clip(ll_out - ll_in, -700, 700)))
            incl = rng.random((C, p)) < pin
            if it >= burn_in:
                beta_draws[:, it - burn_in] = beta
                sig_draws[:, it - burn_in] = sigma2
                gamma_draws[:, it - burn_in] = incl
        gamma = gamma_draws.reshape(C * keep, p).T

    beta_flat = beta_draws.reshape(C * keep, q)
    return {
        "beta0": beta_flat[:, 0],
        "beta": beta_flat[:, 1:].T,          # (N, n_samples)
        "sigma2": sig_draws.reshape(C * keep),
        "gamma": gamma,                      # (N, n_samples) or None
    }
