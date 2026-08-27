"""Probabilistic factor models for MOSC, and the score that compares them.

Two likelihoods, one interface. Each fits a ``(T, N)`` panel and returns draws of
the per-unit loadings ``Z`` -- the paper's estimated confounding structure --
together with the per-period factors and any location term, so a caller can
reconstruct the fitted mean without knowing which model produced it.

The gamma-Poisson arm is drawn by conjugate Gibbs. The augmentation that makes it
conjugate is standard: each observed count is split across the ``K`` components
by a multinomial draw, after which both factor matrices are Gamma. The authors'
own code sketches this and then draws with NUTS instead, because the sketch does
not run. Gibbs removes the sampler from the dependency surface entirely -- no
NumPyro, no ``[bayes]`` extra.

PPCA is the classical Tipping & Bishop (1999) EM: a free loading matrix, an
explicit per-period location, and the loadings marginalised in the E step. The
authors' formulation puts standard-normal priors on both factor matrices with no
location, which places almost no mass near a panel of counts in the tens of
thousands and collapses to the origin under alternating maximisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError


@dataclass(frozen=True)
class FactorDraws:
    """Posterior draws of a factorisation of a ``(T, N)`` panel.

    ``loadings`` are the per-unit latent vectors the downstream regression
    adjusts for; ``factors`` and ``offset`` exist so the fitted mean can be
    rebuilt for scoring, and carry no interpretation of their own -- the factor
    model is identified only up to an invertible transformation of ``Z``.
    """

    loadings: np.ndarray                  # (S, K, N)
    factors: np.ndarray                   # (S, T, K)
    offset: Optional[np.ndarray] = None   # (T,), PPCA only
    likelihood: str = "poisson"

    @property
    def n_draws(self) -> int:
        return int(self.loadings.shape[0])

    def mean(self, draw: int) -> np.ndarray:
        """Fitted mean of the panel under one draw, shape ``(T, N)``."""
        fitted = self.factors[draw] @ self.loadings[draw]
        return fitted if self.offset is None else fitted + self.offset[:, None]


def gap_gibbs(
    panel: np.ndarray,
    n_factors: int,
    mask: Optional[np.ndarray] = None,
    n_samples: int = 200,
    n_warmup: int = 200,
    shape: float = 1.0,
    rate: float = 1.0,
    seed: int = 0,
) -> FactorDraws:
    """Gamma-Poisson factorisation by conjugate Gibbs.

    ``mask`` is True where a cell is observed; held-out cells contribute no
    likelihood term and are imputed by the fitted rate.
    """
    rng = np.random.default_rng(seed)
    n_time, n_unit = panel.shape
    counts = np.rint(panel).astype(np.int64)
    if counts.min() < 0:  # pragma: no cover - setup refuses this earlier
        raise MlsynthEstimationError("Gamma-Poisson factorisation needs non-negative counts.")

    observed = np.ones((n_time, n_unit)) if mask is None else np.asarray(mask, dtype=float)
    counts = np.where(observed > 0, counts, 0)

    factors = rng.gamma(shape, 1.0 / rate, size=(n_time, n_factors))
    loadings = rng.gamma(shape, 1.0 / rate, size=(n_factors, n_unit))

    kept_loadings = np.empty((n_samples, n_factors, n_unit))
    kept_factors = np.empty((n_samples, n_time, n_factors))

    for sweep in range(n_warmup + n_samples):
        allocation = _allocate(rng, counts, factors, loadings)
        factors = rng.gamma(
            shape + allocation.sum(axis=1),
            1.0 / (rate + np.einsum("tn,kn->tk", observed, loadings)),
        )
        loadings = rng.gamma(
            shape + allocation.sum(axis=0).T,
            1.0 / (rate + np.einsum("tn,tk->kn", observed, factors)),
        )
        if sweep >= n_warmup:
            kept_loadings[sweep - n_warmup] = loadings
            kept_factors[sweep - n_warmup] = factors

    return FactorDraws(loadings=kept_loadings, factors=kept_factors, likelihood="poisson")


def _allocate(rng, counts: np.ndarray, factors: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Split each observed count across the ``K`` components, shape ``(T, N, K)``."""
    rates = np.einsum("tk,kn->tnk", factors, loadings)
    totals = rates.sum(axis=2, keepdims=True)
    np.divide(rates, np.where(totals > 0, totals, 1.0), out=rates)

    flat_counts = counts.reshape(-1)
    flat_rates = rates.reshape(-1, rates.shape[-1])
    out = np.zeros_like(flat_rates)
    for cell in np.flatnonzero(flat_counts):
        out[cell] = rng.multinomial(flat_counts[cell], flat_rates[cell])
    return out.reshape(rates.shape)


def ppca_em(
    panel: np.ndarray,
    n_factors: int,
    mask: Optional[np.ndarray] = None,
    n_samples: int = 200,
    n_iter: int = 300,
    tol: float = 1e-9,
    seed: int = 0,
    **_: object,
) -> FactorDraws:
    """Probabilistic PCA by EM, then draws from the Gaussian posterior over ``Z``.

    Units are the exchangeable data points, so each unit's column is one
    observation in ``R^T`` and the loading matrix is shared. Missing cells are
    imputed from the current fit each sweep, which is the EM for data missing at
    random.
    """
    rng = np.random.default_rng(seed)
    n_time, n_unit = panel.shape
    observed = np.ones((n_time, n_unit), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    row_means = np.array(
        [panel[t, observed[t]].mean() if observed[t].any() else 0.0 for t in range(n_time)]
    )
    filled = np.where(observed, panel, row_means[:, None])
    offset = filled.mean(axis=1)

    factors = rng.normal(size=(n_time, n_factors))
    sigma2 = max(float(np.var(panel[observed])), 1e-8)

    previous = np.inf
    loadings = np.zeros((n_factors, n_unit))
    for _iteration in range(n_iter):
        centred = filled - offset[:, None]

        gram = factors.T @ factors + sigma2 * np.eye(n_factors)
        gram_inv = np.linalg.inv(gram)
        loadings = gram_inv @ factors.T @ centred
        second = n_unit * sigma2 * gram_inv + loadings @ loadings.T

        factors = (centred @ loadings.T) @ np.linalg.inv(second)
        residual = centred - factors @ loadings
        sigma2 = max(
            float(
                (np.sum(residual ** 2) + n_unit * sigma2 * np.trace(gram_inv @ (factors.T @ factors)))
                / (n_time * n_unit)
            ),
            1e-10,
        )

        fitted = factors @ loadings + offset[:, None]
        filled = np.where(observed, panel, fitted)
        offset = offset + np.mean(filled - fitted, axis=1)

        objective = float(np.mean((filled - (factors @ loadings + offset[:, None])) ** 2))
        if abs(previous - objective) < tol * max(1.0, abs(previous)):
            break
        previous = objective

    gram = factors.T @ factors + sigma2 * np.eye(n_factors)
    covariance = sigma2 * np.linalg.inv(gram)
    covariance = 0.5 * (covariance + covariance.T)
    posterior_mean = np.linalg.solve(gram, factors.T @ (filled - offset[:, None]))

    kept_loadings = np.empty((n_samples, n_factors, n_unit))
    for unit in range(n_unit):
        kept_loadings[:, :, unit] = rng.multivariate_normal(
            posterior_mean[:, unit], covariance, size=n_samples
        )
    kept_factors = np.broadcast_to(factors, (n_samples, n_time, n_factors)).copy()
    return FactorDraws(
        loadings=kept_loadings, factors=kept_factors, offset=offset, likelihood="gaussian"
    )


# Dispatch by table: adding a likelihood is a row here plus a Literal in the
# config, not another branch in the pipeline.
FACTOR_MODELS: Dict[str, Callable[..., FactorDraws]] = {
    "gap": gap_gibbs,
    "ppca": ppca_em,
}

#: Likelihoods whose support excludes negative outcomes.
NON_NEGATIVE_ONLY = frozenset({"gap"})


def heldout_log_density(panel: np.ndarray, mask: np.ndarray, draws: FactorDraws) -> float:
    """Mean log predictive density per held-out cell. Higher is a better fit.

    This is the model comparison the paper reaches for its ``p_pop`` check to
    make, reported as a score. The check itself is not offered: equation 36 sums
    its discrepancy over held-out cells, so the systematic part grows like the
    cell count while its spread grows like the square root, and the verdict stops
    depending on the data. A score makes no calibration claim and so cannot make
    a false one.
    """
    heldout = ~np.asarray(mask, dtype=bool)
    if not heldout.any():  # pragma: no cover - the caller always holds cells out
        raise MlsynthEstimationError("Scoring the fit needs at least one held-out cell.")

    per_draw = []
    for draw in range(draws.n_draws):
        fitted = draws.mean(draw)
        if draws.likelihood == "poisson":
            fitted = np.clip(fitted, 1e-10, None)
            per_draw.append(float(np.mean(_poisson_logpmf(panel[heldout], fitted[heldout]))))
        else:
            scale = max(float(np.std(panel[mask.astype(bool)] - fitted[mask.astype(bool)])), 1e-8)
            per_draw.append(float(np.mean(_normal_logpdf(panel[heldout], fitted[heldout], scale))))
    return float(np.mean(per_draw))


def _poisson_logpmf(observed: np.ndarray, rate: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    counts = np.rint(observed)
    return counts * np.log(rate) - rate - gammaln(counts + 1.0)


def _normal_logpdf(observed: np.ndarray, location: np.ndarray, scale: float) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi * scale ** 2) - 0.5 * ((observed - location) / scale) ** 2
