"""NUTS (NumPyro) sampler for MVBBSC -- Martinez & Vives-i-Bastida (2024).

The model is the outcome-only Bayesian synthetic control of arXiv:2206.01779
(the ``bsynth`` package's ``model1`` with ``predictor_match = FALSE``):

    w      ~ Dirichlet(1_N)                 # uniform prior over the simplex
    sigma  ~ HalfNormal(1)
    y_z    ~ Normal(X_z w, sigma)           # Gaussian likelihood, pre-period

Both the treated series ``y`` and the donor matrix ``X`` are standardized by
their pre-period mean and standard deviation (``ddof = 1``, matching the
reference ``sd()``) before fitting, and the counterfactual is transformed back
to the outcome scale. The counterfactual returned here is the posterior
predictive draw of the treated outcome in absence of treatment -- the
noiseless mean ``X_z w`` plus an idiosyncratic ``Normal(0, sigma)`` shock, as in
the reference ``generated quantities`` block -- so its pointwise quantiles are a
credible band. Cross-validated against the ``bsynth`` R package on the German
reunification panel (see ``benchmarks/cases/mvbbsc_germany.py``).

The NumPyro / JAX import is isolated in :func:`_import_backend` so a missing
``[bayes]`` optional dependency degrades to a translated error (and so tests can
exercise that path).
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


def run_mvbbsc(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    n_warmup: int,
    n_samples: int,
    n_chains: int,
    target_accept: float,
    seed: int,
    progress: bool = False,
) -> Dict[str, Any]:
    """Sample the MVBBSC posterior; return counterfactual and weight draws.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Treated outcome over all periods.
    X : np.ndarray, shape (T, N)
        Donor matrix over all periods.
    T0 : int
        Number of pre-treatment periods.
    n_warmup, n_samples, n_chains, target_accept, seed : see config.
    progress : bool
        Show the NUTS progress bar.

    Returns
    -------
    dict with ``counterfactual`` ((n_draws, T) posterior-predictive draws on the
    outcome scale), ``weights`` ((n_draws, N) simplex draws), ``sigma``
    ((n_draws,) standardized scale), ``accept_prob``, ``n_divergent``,
    ``max_rhat``.
    """
    try:
        jax, jnp, numpyro, dist, MCMC, NUTS, diagnostics = _import_backend()
    except ImportError as exc:
        raise MlsynthEstimationError(
            "MVBBSC requires NumPyro (pip install 'mlsynth[bayes]')."
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected backend failure
        raise MlsynthEstimationError(
            f"MVBBSC backend import failed: {exc}"
        ) from exc

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    T, N = X.shape

    # Standardize by pre-period moments (ddof=1 to match the reference sd()).
    my = float(y[:T0].mean())
    sy = float(y[:T0].std(ddof=1))
    sy = sy if sy > 0 else 1.0
    mX = X[:T0].mean(axis=0)
    sX = X[:T0].std(axis=0, ddof=1)
    sX = np.where(sX > 0, sX, 1.0)

    yz_pre = (y[:T0] - my) / sy
    Xz_pre = (X[:T0] - mX) / sX
    Xz_all = (X - mX) / sX

    Xz_pre_j = jnp.asarray(Xz_pre)
    yz_pre_j = jnp.asarray(yz_pre)

    def model():
        w = numpyro.sample("w", dist.Dirichlet(jnp.ones(N)))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        numpyro.sample("y", dist.Normal(Xz_pre_j @ w, sigma), obs=yz_pre_j)

    per_chain = max(2, int(np.ceil(n_samples)))
    mcmc = MCMC(
        NUTS(model, target_accept_prob=target_accept),
        num_warmup=n_warmup, num_samples=per_chain, num_chains=n_chains,
        progress_bar=progress,
    )
    mcmc.run(jax.random.PRNGKey(int(seed)),
             extra_fields=("accept_prob", "diverging"))

    grouped = mcmc.get_samples(group_by_chain=True)
    w_g = np.asarray(grouped["w"])                 # (chains, per_chain, N)
    s_g = np.asarray(grouped["sigma"])             # (chains, per_chain)
    W = w_g.reshape(-1, N)                          # (n_draws, N)
    S = s_g.reshape(-1)                             # (n_draws,)
    n_draws = W.shape[0]

    # Posterior-predictive counterfactual (matches the reference normal_rng):
    # noiseless mean X_z w plus a Normal(0, sigma) shock, then back-transformed.
    mu_z = W @ Xz_all.T                             # (n_draws, T) standardized mean
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((n_draws, T)) * S[:, None]
    cf = (mu_z + noise) * sy + my                   # (n_draws, T) outcome scale

    extra = mcmc.get_extra_fields()
    accept = float(np.mean(np.asarray(extra.get("accept_prob", np.nan))))
    n_div = int(np.sum(np.asarray(extra.get("diverging", 0))))
    if n_chains >= 2:
        rhat_w = np.asarray(diagnostics.split_gelman_rubin(w_g))
        rhat_s = float(diagnostics.split_gelman_rubin(s_g[..., None])[0])
        max_rhat = float(np.nanmax(np.append(rhat_w, rhat_s)))
    else:
        max_rhat = float("nan")

    return {
        "counterfactual": cf,
        "weights": W,
        "sigma": S,
        "accept_prob": accept,
        "n_divergent": n_div,
        "max_rhat": max_rhat,
    }
