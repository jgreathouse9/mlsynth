"""NUTS (NumPyro) sampler for BFSC -- Pinkney (2021) Bayesian factor SC.

The model reproduces the paper's appendix Stan program: a lower-trapezoidal
factor matrix ``F`` (identified via zeros above the diagonal), horseshoe+
loadings, year (``delta``) and unit (``kappa``) effects, the subtract-last-
pre-period / divide-by-pre-SD standardization, and the masked-treated
imputation with donors contributing all periods. Validated cell-for-cell
against the author's Stan (posterior counterfactual to <0.5%, sigma to 4
decimals) on the German reunification panel.

The NumPyro / JAX import is isolated in :func:`_import_backend` so a missing
``[bayes]`` optional dependency degrades to a translated error (and so tests
can exercise that path).
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from ...exceptions import MlsynthEstimationError


def _import_backend():  # pragma: no cover - trivial import shim (monkeypatched in tests)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import numpyro.diagnostics as diagnostics
    return jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics


def run_bfsc(
    Y: np.ndarray, T0: int, *, n_factors: int, n_warmup: int, n_samples: int,
    n_chains: int, target_accept: float, seed: int, progress: bool = False,
) -> Dict[str, Any]:
    """Sample the BFSC posterior; return the treated counterfactual draws.

    Parameters
    ----------
    Y : np.ndarray, shape (J, T)
        Outcomes, treated unit first. The treated post-period is masked here.
    T0 : int
        Number of pre-treatment periods.
    n_factors, n_warmup, n_samples, n_chains, target_accept, seed : see config.

    Returns
    -------
    dict with ``counterfactual`` ((n_draws, T), unscaled treated counterfactual),
    ``sigma`` (n_draws,), ``accept_prob``, ``n_divergent``, ``max_rhat``.
    """
    try:
        jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics = _import_backend()
    except ImportError as exc:
        raise MlsynthEstimationError(
            "BFSC requires NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected backend failure
        raise MlsynthEstimationError(f"BFSC backend import failed: {exc}") from exc

    Y = np.asarray(Y, dtype=float)
    J, T = Y.shape
    L = int(n_factors)
    n_pre = T0
    # standardization: subtract the last pre-period value, divide by pre-SD
    y_mu = Y[:, n_pre - 1].copy()
    y_sd = Y[:, :n_pre].std(axis=1, ddof=1)
    y_sd = np.where(y_sd > 0, y_sd, 1.0)
    Ys = (Y - y_mu[:, None]) / y_sd[:, None]                 # (J, T) scaled
    M = L * T - L * (L + 1) // 2

    # static index arrays for the lower-trapezoidal fill (Stan column order)
    rows, cols = [], []
    for j in range(L):
        for i in range(j + 1, T):
            rows.append(i); cols.append(j)
    rows = jnp.asarray(rows); cols = jnp.asarray(cols); diagj = jnp.arange(L)

    y_pre_treated = jnp.asarray(Ys[0, :n_pre])
    y_donors = jnp.asarray(Ys[1:, :])                        # (J-1, T) all periods

    def build_F(F_diag, F_lower):
        F = jnp.zeros((T, L))
        F = F.at[diagj, diagj].set(F_diag)
        return F.at[rows, cols].set(F_lower)

    def model():
        delta = numpyro.sample("delta", dist.Normal(0.0, 2.0).expand([T]))
        kappa = numpyro.sample("kappa", dist.Normal(0.0, 1.0).expand([J]))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        F_diag = numpyro.sample("F_diag", dist.HalfNormal(1.0).expand([L]))
        F_lower = numpyro.sample("F_lower", dist.Normal(0.0, 2.0).expand([M]))
        beta_off = numpyro.sample("beta_off", dist.Normal(0.0, 1.0).expand([J, L]))
        lam = numpyro.sample("lambda", dist.Uniform(0.0, 1.0).expand([L]))
        eta = numpyro.sample("eta", dist.Uniform(0.0, 1.0))
        tau = numpyro.sample("tau", dist.Uniform(0.0, 1.0).expand([J]))

        F = build_F(F_diag, F_lower)                         # (T, L)
        cache = jnp.tan(0.5 * jnp.pi * lam) * jnp.tan(0.5 * jnp.pi * eta)
        tau_ = jnp.tan(0.5 * jnp.pi * tau)
        beta = cache[:, None] * (beta_off * tau_[:, None]).T  # (L, J)
        mean = F @ beta + delta[:, None] + kappa[None, :]    # (T, J) scaled means

        numpyro.sample("y_donors", dist.Normal(mean[:, 1:].T, sigma), obs=y_donors)
        numpyro.sample("y_tr_pre", dist.Normal(mean[:n_pre, 0], sigma), obs=y_pre_treated)
        numpyro.sample("y_tr_post", dist.Normal(mean[n_pre:, 0], sigma))  # masked/imputed
        numpyro.deterministic("cf", mean[:, 0] * y_sd[0] + y_mu[0])

    per_chain = max(2, int(np.ceil(n_samples / n_chains)))
    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept),
                num_warmup=n_warmup, num_samples=per_chain, num_chains=n_chains,
                progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(int(seed)),
             extra_fields=("accept_prob", "diverging"))

    grouped = mcmc.get_samples(group_by_chain=True)
    cf_g = np.asarray(grouped["cf"])                          # (chains, per_chain, T)
    sig_g = np.asarray(grouped["sigma"])
    cf = cf_g.reshape(-1, T)
    sig = sig_g.reshape(-1)
    extra = mcmc.get_extra_fields()
    accept = float(np.mean(np.asarray(extra.get("accept_prob", np.nan))))
    n_div = int(np.sum(np.asarray(extra.get("diverging", 0))))
    # r-hat on the identified quantities (counterfactual + sigma); needs >= 2 chains
    if n_chains >= 2:
        rhat_cf = np.asarray(diagnostics.split_gelman_rubin(cf_g))
        rhat_s = float(diagnostics.split_gelman_rubin(sig_g[..., None])[0])
        max_rhat = float(np.nanmax(np.append(rhat_cf, rhat_s)))
    else:
        max_rhat = float("nan")
    return {"counterfactual": cf, "sigma": sig, "accept_prob": accept,
            "n_divergent": n_div, "max_rhat": max_rhat}
