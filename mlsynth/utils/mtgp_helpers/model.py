"""NUTS (NumPyro) sampler for MTGP -- Ben-Michael et al. (2023) multitask GP.

The model reproduces the paper's Gaussian Stan program (``normal.stan`` in the
authors' replication package): control potential outcomes are a global time-GP
trend, a low-rank intrinsic-coregionalization term whose latent factors carry a
squared-exponential smoothness prior over time (``L_f z_f k_f``), and unit
intercepts, with population-heteroskedastic Gaussian noise. The treated unit's
post-period cells are masked; the posterior of the latent mean there is the
counterfactual. Validated cell-for-cell against the author's Stan (counterfactual
correlation 0.99993, hyperparameters to ~2%) on the California APPS panel.

Runs in double precision (``enable_x64``): the GP Cholesky with a 1e-9 jitter is
numerically unusable in JAX's default float32. The NumPyro / JAX import is
isolated in :func:`_import_backend` so a missing ``[bayes]`` optional dependency
degrades to a translated error (and so tests can exercise that path).
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from ...exceptions import MlsynthEstimationError


def _import_backend():  # pragma: no cover - trivial import shim (monkeypatched in tests)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpyro
    numpyro.enable_x64()                     # GP Cholesky + 1e-9 jitter need float64
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import numpyro.diagnostics as diagnostics
    return jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics


def run_mtgp(
    Y: np.ndarray, inv_pop: np.ndarray, T0: int, treated_col: int, *,
    n_factors: int, n_warmup: int, n_samples: int, n_chains: int,
    target_accept: float, max_tree_depth: int, seed: int, progress: bool = False,
) -> Dict[str, Any]:
    """Sample the MTGP posterior; return the treated counterfactual draws.

    Parameters
    ----------
    Y : np.ndarray, shape (T, D)
        Outcomes, time x unit; the treated unit is column ``treated_col`` and its
        post-period cells are masked here.
    inv_pop : np.ndarray, shape (T, D)
        Inverse-population noise scaling (ones for homoskedastic).
    T0 : int
        Number of pre-treatment periods.
    treated_col : int
        Column index of the treated unit.
    n_factors : int
        Rank of the low-rank unit kernel (number of shared latent time-GP factors).

    Returns
    -------
    dict with ``counterfactual`` ((n_draws, T) treated counterfactual),
    ``sigma`` (n_draws,), ``accept_prob``, ``n_divergent``, ``max_rhat``,
    ``lengthscale_f``, ``lengthscale_global``.
    """
    try:
        jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics = _import_backend()
    except ImportError as exc:
        raise MlsynthEstimationError(
            "MTGP requires NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected backend failure
        raise MlsynthEstimationError(f"MTGP backend import failed: {exc}") from exc

    Y = np.asarray(Y, dtype=float)
    inv_pop = np.asarray(inv_pop, dtype=float)
    N, D = Y.shape                                  # N = time, D = units
    J = int(n_factors)
    # standardized time index (evenly-spaced periods -> matches standardized years)
    x = np.arange(N, dtype=float)
    xn = (x - x.mean()) / x.std(ddof=1)
    d2 = jnp.asarray((xn[:, None] - xn[None, :]) ** 2)
    jit = 1e-9 * jnp.eye(N)

    # control cells: everything except the treated column at/after T0
    treated_mask = np.zeros((N, D), dtype=bool)
    treated_mask[T0:, treated_col] = True
    cr, cc = np.where(~treated_mask)
    cr, cc = jnp.asarray(cr), jnp.asarray(cc)
    y_j = jnp.asarray(Y)
    invp_j = jnp.asarray(inv_pop)

    def _chol(sig, ls):
        K = sig ** 2 * jnp.exp(-d2 / (2.0 * ls ** 2)) + jit
        return jnp.linalg.cholesky(K)

    def model():
        ls_f = numpyro.sample("lengthscale_f", dist.InverseGamma(5.0, 5.0))
        sig_f = numpyro.sample("sigma_f", dist.HalfNormal(1.0))
        ls_g = numpyro.sample("lengthscale_global", dist.InverseGamma(5.0, 5.0))
        sig_g = numpyro.sample("sigma_global", dist.HalfNormal(1.0))
        sigman = numpyro.sample("sigman", dist.HalfNormal(1.0))
        global_offset = numpyro.sample("global_offset", dist.Normal(0.0, 10.0))
        state_offset = numpyro.sample("state_offset", dist.Normal(0.0, 1.0).expand([D]))
        z_global = numpyro.sample("z_global", dist.Normal(0.0, 1.0).expand([N]))
        z_f = numpyro.sample("z_f", dist.Normal(0.0, 1.0).expand([N, J]))
        k_f = numpyro.sample("k_f", dist.Normal(0.0, 1.0).expand([J, D]))

        L_f, L_g = _chol(sig_f, ls_f), _chol(sig_g, ls_g)
        mean = (global_offset + state_offset[None, :]
                + (L_g @ z_global)[:, None] + (L_f @ z_f @ k_f))       # (N, D)
        numpyro.deterministic("cf", mean[:, treated_col])
        scale = sigman * jnp.sqrt(invp_j)
        numpyro.sample("y_obs", dist.Normal(mean[cr, cc], scale[cr, cc]),
                       obs=y_j[cr, cc])

    per_chain = max(2, int(np.ceil(n_samples / n_chains)))
    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept,
                     max_tree_depth=max_tree_depth),
                num_warmup=n_warmup, num_samples=per_chain, num_chains=n_chains,
                progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(int(seed)),
             extra_fields=("accept_prob", "diverging"))

    grouped = mcmc.get_samples(group_by_chain=True)
    cf_g = np.asarray(grouped["cf"])                     # (chains, per_chain, T)
    sig_g = np.asarray(grouped["sigman"])
    cf = cf_g.reshape(-1, N)
    sig = sig_g.reshape(-1)
    samp = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    accept = float(np.mean(np.asarray(extra.get("accept_prob", np.nan))))
    n_div = int(np.sum(np.asarray(extra.get("diverging", 0))))
    if n_chains >= 2:
        rhat_cf = np.asarray(diagnostics.split_gelman_rubin(cf_g))
        rhat_s = float(diagnostics.split_gelman_rubin(sig_g[..., None])[0])
        max_rhat = float(np.nanmax(np.append(rhat_cf, rhat_s)))
    else:
        max_rhat = float("nan")
    return {"counterfactual": cf, "sigma": sig, "accept_prob": accept,
            "n_divergent": n_div, "max_rhat": max_rhat,
            "lengthscale_f": float(np.mean(np.asarray(samp["lengthscale_f"]))),
            "lengthscale_global": float(np.mean(np.asarray(samp["lengthscale_global"])))}
