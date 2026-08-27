"""Port of Wang, Schein, Shou & Blei's Algorithm 1 from the authors' own code.

Upstream: https://github.com/Joshuashou/Synthetic-Control-Paper-Model
(``src/bayesian_factor_model.py``, ``src/deconfound_and_plot.py``).

The panel is oriented (time x unit) throughout, as upstream: ``Z`` is the
(K x N) matrix of per-unit latent loadings that the paper calls the estimated
confounding structure, ``H`` is the (T x K) matrix of per-period factors, and the
last unit column is the single treated unit.

Deviations from the upstream source, each one forced:

1. ``GAP.gibbs_sample`` upstream cannot run. ``rn = np.random.default_rn()`` is a
   typo for ``default_rng`` and also shadows the ``numpy.random as rn`` alias
   used on the preceding lines; the two ``einsum`` subscript strings
   (``'tk,nk->tnk'`` and ``'tn,kn->kt'``) contract axes whose lengths disagree
   with the array shapes the same function builds; and ``data[mask] = 0`` zeroes
   the observed cells instead of the held-out ones, inverting the convention that
   ``population_predictive_check`` and Pyro's ``likelihood.mask`` both use. The
   sampler here is the same conjugate multinomial augmentation with those four
   corrected, so it runs.
2. Upstream draws from the Pyro models with NUTS. The augmentation is conjugate,
   so this Gibbs sampler needs no gradient-based sampler and no torch.
   ``validate_gibbs.py`` checks it against data drawn from the model.
3. PPCA is fit by its EM algorithm instead of NUTS. Section 5 of the paper reports
   rSC and PPCA tracking each other while GaP separates, which is the paper's own
   evidence that the likelihood family and not the inference style drives the
   comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


@dataclass(frozen=True)
class FactorPosterior:
    """Posterior draws of the factorisation of a (time x unit) panel."""

    Z: np.ndarray  # (S, K, N) per-unit loadings
    H: np.ndarray  # (S, T, K) per-period factors
    offset: np.ndarray | None = None  # (T,) per-period mean, PPCA only

    @property
    def n_samples(self) -> int:
        return self.Z.shape[0]

    def rate(self, s: int) -> np.ndarray:
        fitted = self.H[s] @ self.Z[s]
        if self.offset is None:
            return fitted
        return fitted + self.offset[:, None]


def gap_gibbs(
    data: np.ndarray,
    latent_dim: int,
    mask: np.ndarray | None = None,
    n_samples: int = 500,
    warmup: int = 500,
    shp: float = 1.0,
    rte: float = 1.0,
    seed: int = 0,
) -> FactorPosterior:
    """Gamma-Poisson factorisation by conjugate Gibbs (upstream ``GAP``).

    ``mask`` is True where a cell is observed, matching upstream's Pyro
    convention; held-out cells contribute no likelihood term.
    """
    rng = np.random.default_rng(seed)
    n_time, n_unit = data.shape
    counts = np.rint(data).astype(np.int64)
    if counts.min() < 0:
        raise ValueError("gamma-Poisson factorisation needs non-negative counts")

    observed = np.ones((n_time, n_unit)) if mask is None else mask.astype(float)
    counts = np.where(observed > 0, counts, 0)

    H = rng.gamma(shp, 1.0 / rte, size=(n_time, latent_dim))
    Z = rng.gamma(shp, 1.0 / rte, size=(latent_dim, n_unit))

    keep_Z = np.empty((n_samples, latent_dim, n_unit))
    keep_H = np.empty((n_samples, n_time, latent_dim))

    for step in range(warmup + n_samples):
        # Allocation: split each observed count across the K components.
        rates = np.einsum("tk,kn->tnk", H, Z)
        totals = rates.sum(axis=2, keepdims=True)
        np.divide(rates, np.where(totals > 0, totals, 1.0), out=rates)
        alloc = _multinomial_allocate(rng, counts, rates)

        # H | . and Z | ., both Gamma by conjugacy.
        H = rng.gamma(shp + alloc.sum(axis=1), 1.0 / (rte + np.einsum("tn,kn->tk", observed, Z)))
        Z = rng.gamma(
            shp + alloc.sum(axis=0).T, 1.0 / (rte + np.einsum("tn,tk->kn", observed, H))
        )

        if step >= warmup:
            keep_Z[step - warmup] = Z
            keep_H[step - warmup] = H

    return FactorPosterior(Z=keep_Z, H=keep_H)


def _multinomial_allocate(rng, counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Draw (T, N, K) component allocations of each (T, N) count."""
    flat_counts = counts.reshape(-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    out = np.zeros_like(flat_probs)
    nonzero = np.flatnonzero(flat_counts)
    for i in nonzero:
        out[i] = rng.multinomial(flat_counts[i], flat_probs[i])
    return out.reshape(probs.shape)


def ppca_em(
    data: np.ndarray,
    latent_dim: int,
    mask: np.ndarray | None = None,
    n_samples: int = 500,
    n_iter: int = 300,
    tol: float = 1e-9,
    seed: int = 0,
) -> FactorPosterior:
    """Probabilistic PCA (Tipping & Bishop 1999) by EM, with posterior draws of Z.

    Units are the exchangeable data points, so each unit's column of the
    (time x unit) panel is one observation in R^T and ``H`` is the shared loading
    matrix. Missing cells are imputed from the current fit each sweep, which is
    the EM for data missing at random.

    Upstream fits this with NUTS under standard-normal priors on both factors and
    no location term. On COVID counts in the tens of thousands those priors put
    almost no mass near the data, and alternating MAP under them collapses to the
    origin. The classical formulation -- a free loading matrix, an explicit
    per-period mean, and Z marginalised in the E step -- is what the cited paper
    specifies, and it gives the Gaussian arm a fair fit instead of a hobbled one.
    """
    rng = np.random.default_rng(seed)
    n_time, n_unit = data.shape
    observed = np.ones((n_time, n_unit), dtype=bool) if mask is None else mask.astype(bool)
    if not observed.any():
        raise ValueError("PPCA needs at least one observed cell")

    row_means = np.array(
        [data[t, observed[t]].mean() if observed[t].any() else 0.0 for t in range(n_time)]
    )
    filled = np.where(observed, data, row_means[:, None])
    offset = filled.mean(axis=1)

    H = rng.normal(scale=1.0, size=(n_time, latent_dim))
    sigma2 = max(float(np.var(data[observed])), 1e-8)

    previous = np.inf
    Z = np.zeros((latent_dim, n_unit))
    for _ in range(n_iter):
        centred = filled - offset[:, None]

        # E step with Z marginalised: posterior mean and pooled second moment.
        gram = H.T @ H + sigma2 * np.eye(latent_dim)
        gram_inv = np.linalg.inv(gram)
        Z = gram_inv @ H.T @ centred                    # (K, N)
        second = n_unit * sigma2 * gram_inv + Z @ Z.T   # (K, K)

        # M step: loadings, then the noise scale.
        H = (centred @ Z.T) @ np.linalg.inv(second)
        resid = centred - H @ Z
        sigma2 = max(
            float(
                (np.sum(resid ** 2) + n_unit * sigma2 * np.trace(gram_inv @ (H.T @ H)))
                / (n_time * n_unit)
            ),
            1e-10,
        )

        # Impute the held-out cells from the current fit, then refresh the location.
        fitted = H @ Z + offset[:, None]
        filled = np.where(observed, data, fitted)
        offset = offset + np.mean(filled - fitted, axis=1)

        objective = float(np.mean((filled - (H @ Z + offset[:, None])) ** 2))
        if abs(previous - objective) < tol * max(1.0, abs(previous)):
            break
        previous = objective

    # Posterior draws of each unit's loadings; H and the location stay at the fit.
    gram = H.T @ H + sigma2 * np.eye(latent_dim)
    cov = sigma2 * np.linalg.inv(gram)
    cov = 0.5 * (cov + cov.T)
    Z_mean = np.linalg.solve(gram, H.T @ (filled - offset[:, None]))
    keep_Z = np.empty((n_samples, latent_dim, n_unit))
    for n in range(n_unit):
        keep_Z[:, :, n] = rng.multivariate_normal(Z_mean[:, n], cov, size=n_samples)
    keep_H = np.broadcast_to(H, (n_samples, n_time, latent_dim)).copy()
    return FactorPosterior(Z=keep_Z, H=keep_H, offset=offset)


def counterfactual_from_regression(
    Z: np.ndarray,
    panel: np.ndarray,
    intervention_t: int,
    include_previous_outcome: bool,
    alphas: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2),
) -> np.ndarray:
    """Upstream ``get_counterfactual_from_best_reg`` for one posterior draw.

    Regresses every unit's post-period outcome vector on its loadings, a treatment
    dummy and -- when ``include_previous_outcome`` -- its last pre-period outcome,
    then predicts the treated unit's row with the dummy switched off.

    ``include_previous_outcome`` is the switch that carries every number the paper
    reports: it is hardcoded True in upstream's ``run_semi_synthetic_experiment.py``
    and set True in ``calculate_ATT.ipynb``, and appears nowhere in equations 40-41.
    """
    design, response = _regression_arrays(
        Z, panel, intervention_t, treated_on=True, include_previous_outcome=include_previous_outcome
    )
    search = GridSearchCV(Ridge(), {"alpha": list(alphas)}, scoring="r2", cv=5)
    search.fit(design, response)

    counterfactual_design, _ = _regression_arrays(
        Z, panel, intervention_t, treated_on=False, include_previous_outcome=include_previous_outcome
    )
    return search.best_estimator_.predict(counterfactual_design)[-1]


def _regression_arrays(
    Z: np.ndarray,
    panel: np.ndarray,
    intervention_t: int,
    treated_on: bool,
    include_previous_outcome: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_unit = Z.shape[1]
    by_unit = panel.T

    treat = np.zeros((n_unit, 1))
    if treated_on:
        treat[-1] = 1.0

    blocks = [Z.T]
    if include_previous_outcome:
        blocks.append(by_unit[:, intervention_t - 1][:, None])
    blocks.append(treat)
    return np.concatenate(blocks, axis=1), by_unit[:, intervention_t:]


def population_predictive_check(
    data: np.ndarray, mask: np.ndarray, posterior: FactorPosterior, likelihood: str, seed: int = 0
) -> float:
    """Upstream ``population_predictive_check`` (equations 35-36).

    ``mask`` is True where observed, so ``~mask`` selects the held-out cells the
    check scores. Returns ``p_pop``; the paper rejects a model outside
    ``[alpha/2, 1 - alpha/2]``.
    """
    rng = np.random.default_rng(seed)
    heldout = ~mask.astype(bool)
    if not heldout.any():
        raise ValueError("predictive check needs held-out cells")

    wins = 0
    for s in range(posterior.n_samples):
        rate = np.clip(posterior.rate(s), 1e-10, None)
        if likelihood == "poisson":
            replicate = rng.poisson(rate)
            d_fake = -_poisson_logpmf(replicate[heldout], rate[heldout]).sum()
            d_true = -_poisson_logpmf(np.rint(data[heldout]).astype(np.int64), rate[heldout]).sum()
        elif likelihood == "gaussian":
            scale = max(float(np.std(data[mask.astype(bool)] - rate[mask.astype(bool)])), 1e-8)
            replicate = rng.normal(rate, scale)
            d_fake = -_normal_logpdf(replicate[heldout], rate[heldout], scale).sum()
            d_true = -_normal_logpdf(data[heldout], rate[heldout], scale).sum()
        else:
            raise ValueError(f"unknown likelihood {likelihood!r}")
        wins += int(d_fake > d_true)
    return wins / posterior.n_samples


def _poisson_logpmf(k: np.ndarray, rate: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return k * np.log(rate) - rate - gammaln(k + 1.0)


def _normal_logpdf(x: np.ndarray, loc: np.ndarray, scale: float) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi * scale**2) - 0.5 * ((x - loc) / scale) ** 2
