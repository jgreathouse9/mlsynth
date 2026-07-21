"""NUTS sampler for BPSCS -- distance-horseshoe (dhs) and distance-spike-and-slab
(ds2) shrinkage priors (Fernandez-Morales, Oganisian & Lee 2026).

Ported from the paper's equations; the counterfactual is a free-running
autoregressive forward simulation of the treated unit's no-intervention outcome,
masked/imputed over the post-period. NumPyro is imported lazily behind
``_import_backend`` so the dependency is only needed at fit time (``[bayes]``
extra). Double precision is enabled -- the model is otherwise numerically fragile.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MlsynthEstimationError


def _import_backend():
    """Import NumPyro/JAX lazily, enabling double precision and parallel chains."""
    import numpyro
    numpyro.enable_x64()
    numpyro.set_host_device_count(4)
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import numpyro.diagnostics as diagnostics
    return numpyro, jax, jnp, dist, MCMC, NUTS, diagnostics


def run_bpscs(
    Y_std, X_cov, donor_d, rho, T0, prior, *,
    n_warmup, n_samples, n_chains, target_accept, max_tree_depth, seed, progress,
):
    """Sample the BPSCS posterior and free-run the treated counterfactual.

    Parameters mirror :class:`~mlsynth.config_models.BPSCSConfig`. ``Y_std`` is the
    pre-period-standardized outcome matrix (T x D, treated column 0); ``X_cov`` is
    the z-scored per-unit covariate matrix (D x P); ``donor_d`` is the per-donor
    utility (J,) that scales the prior; ``rho`` is the ds2 inclusion radius.

    Returns a dict of standardized counterfactual draws (n_draws x T), donor
    coefficients, and NUTS diagnostics. Raises
    :class:`~mlsynth.exceptions.MlsynthEstimationError` if NumPyro is absent.
    """
    try:
        numpyro, jax, jnp, dist, MCMC, NUTS, diagnostics = _import_backend()
    except ImportError as exc:
        raise MlsynthEstimationError(
            "BPSCS requires NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc

    Y_std = np.asarray(Y_std, dtype=float)
    y_tr = jnp.asarray(Y_std[:, 0])                      # (T,) treated, standardized
    X = jnp.asarray(Y_std[:, 1:])                        # (T, J) donors, standardized
    z = jnp.asarray(np.asarray(X_cov, dtype=float)[0])   # (P,) treated covariates
    dvec = jnp.asarray(np.asarray(donor_d, dtype=float))
    T, J = Y_std.shape[0], Y_std.shape[1] - 1
    P = z.shape[0]
    lag_pre = jnp.concatenate([jnp.zeros(1), y_tr[:-1]])  # psi * Y_{t-1}, 0 at t=1
    w_slab = (dvec > rho).astype(float)                   # ds2 inclusion indicator

    def model():
        phi = numpyro.sample("phi", dist.Normal(0.0, 2.5).expand([P]))
        psi = numpyro.sample("psi", dist.Normal(0.0, 3.0))
        sigma = numpyro.sample("sigma", dist.TruncatedDistribution(
            dist.StudentT(4.0, 0.0, 1.0), low=0.0))
        if prior == "dhs":
            beta_raw = numpyro.sample("beta_raw", dist.Normal(0.0, 1.0).expand([J]))
            lam = numpyro.sample("lambda", dist.HalfCauchy(dvec))
            zeta = numpyro.sample("zeta", dist.HalfCauchy(1.0))
            beta = numpyro.deterministic("beta", sigma * jnp.sqrt(zeta) * lam * beta_raw)
        else:  # ds2 spike-and-slab keyed by the utility cutoff rho
            nu = numpyro.sample("nu", dist.HalfCauchy(1.0))
            scale = jnp.sqrt(jnp.where(w_slab > 0, nu, 0.001))
            beta = numpyro.sample("beta", dist.Normal(0.0, scale).to_event(1))
        mu_pre = lag_pre[:T0] * psi + z @ phi + X[:T0] @ beta
        numpyro.sample("Y_lik", dist.Normal(mu_pre, sigma), obs=y_tr[:T0])

    kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "accept_prob"))
    s = mcmc.get_samples()
    beta = np.asarray(s["beta"])
    phi = np.asarray(s["phi"])
    psi = np.asarray(s["psi"])
    sigma = np.asarray(s["sigma"])
    n = beta.shape[0]

    # free-running counterfactual over all T periods (posterior predictive)
    base = np.asarray(z) @ phi.T + np.asarray(Y_std[:, 1:]) @ beta.T   # (T, n)
    key = jax.random.PRNGKey(seed + 1)
    Xall = jnp.asarray(base)

    def step(carry, t):
        Yprev, k = carry
        k, sub = jax.random.split(k)
        lag = jnp.where(t == 0, 0.0, jnp.asarray(psi) * Yprev)
        mu = lag + Xall[t]
        Yt = mu + jnp.asarray(sigma) * jax.random.normal(sub, (n,))
        return (Yt, k), Yt

    (_, _), cf = jax.lax.scan(step, (jnp.zeros(n), key), jnp.arange(T))
    cf = np.asarray(cf).T                                # (n, T) standardized counterfactual

    ex = mcmc.get_samples(group_by_chain=True)
    def _rhat(v):
        v = np.asarray(v)
        r = diagnostics.gelman_rubin(v) if v.ndim >= 2 and v.shape[0] > 1 else np.array(1.0)
        return float(np.nanmax(r))
    max_rhat = max(_rhat(ex[k]) for k in ("psi", "sigma"))
    ndiv = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    accept = float(np.mean(np.asarray(
        mcmc.get_extra_fields().get("accept_prob", np.array([np.nan])))))

    return {
        "counterfactual": cf, "beta": beta, "sigma": sigma, "psi": psi,
        "n_draws": n, "accept_prob": accept, "n_divergent": ndiv,
        "max_rhat": max_rhat, "n_included": int(np.sum(np.asarray(w_slab))),
    }
