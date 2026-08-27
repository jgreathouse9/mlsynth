"""The semi-synthetic data-generating process, in both of its published forms.

Upstream implements it as ``create_semi_synthetic_matrix`` in
``src/semi_synthetic_experiment.ipynb``. The paper states it as equations 46-51.

The effect function is the same family in both, under a reciprocal
parameterisation. Equation 49 prints as ``delta_t = [1 + log(t)] ^ (t*alpha/1000)``
-- the fraction sits in the exponent, which is the only reading consistent with
the paper's own sentence that ``alpha = 0`` gives ``delta_t = 1``. Upstream writes::

    Mu[t:, -1] *= (1 + log1p(arange(n_post))) ** (arange(n_post) / alpha)

so ``alpha_code = 1000 / alpha_paper``: the paper's largest effect, ``alpha = 4``,
is upstream's ``alpha = 250``, and upstream's ``alpha = 100000`` is ``alpha_paper
= 0.01``, which multiplies the treated path by between 1.0000 and 1.0004 over
thirty post-periods -- no effect at all.

Two things survive that reconciliation:

* Upstream's grid is {250, 500, 100000}, i.e. ``alpha_paper`` in {4, 2, 0.01},
  where the paper states {1/10, 4}. The small arm is an order of magnitude
  smaller in the code than in the paper.
* Upstream's plotting cell pairs those values with labels in the order
  ``zip([250, 500, 100000], ['Small', 'Medium', 'Large'])``, which is inverted
  against the effects they produce: it calls the largest effect Small and the
  null one Large. That cell writes into the paper's own figure directory.

``effect_form="paper"`` takes ``alpha`` on the paper's scale, ``effect_form="code"``
on upstream's. Both call the same curve.

Two departures of the code from the paper are real, and are kept here because
they are what produced the published figure:

* Equation 46 asks for a Poisson (maximum-likelihood) factorisation. Upstream
  calls ``sklearn.decomposition.NMF`` at its ``beta_loss='frobenius'`` default,
  which is the Gaussian loss. ``nmf_loss`` selects.
* Equation 50 mixes the factor model with the previous *realised* outcome, a
  recursion. Upstream draws one Poisson panel from the unmixed rate before the
  loop and mixes against that fixed draw, so its autoregressive arm never
  compounds. ``recursive_mismatch`` selects.

``ground_truth="rate"`` returns the noiseless mean, which is what upstream scores
against; ``ground_truth="draw"`` returns a Poisson draw, which is what the paper's
MRE definition writes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF


@dataclass(frozen=True)
class SemiSyntheticPanel:
    observed: np.ndarray        # (T, N) counts, treated unit is the last column
    truth: np.ndarray           # (n_post,) untreated potential outcome of the treated unit
    intervention_t: int
    multiplier: np.ndarray      # (n_post,) the delta_t actually applied


def _effect_multiplier(n_post: int, alpha: float, effect_form: str) -> np.ndarray:
    """delta_t over the post-period. ``alpha`` is on the code or paper scale."""
    steps = np.arange(n_post)
    if effect_form == "code":
        return (1.0 + np.log1p(steps)) ** (steps / alpha)
    if effect_form == "paper":
        t = steps + 1.0
        return (1.0 + np.log(t)) ** (t * alpha / 1000.0)
    raise ValueError(f"unknown effect_form {effect_form!r}")


def create_semi_synthetic_matrix(
    panel: np.ndarray,
    intervention_t: int,
    alpha: float,
    latent_dim: int = 10,
    seed: int = 617,
    model_mismatch_p: float = 0.0,
    effect_form: str = "code",
    nmf_loss: str = "frobenius",
    ground_truth: str = "rate",
    recursive_mismatch: bool = False,
) -> SemiSyntheticPanel:
    """Regenerate a count panel from a factor model fit to it, then add an effect."""
    n_time, _ = panel.shape

    solver = "cd" if nmf_loss == "frobenius" else "mu"
    nmf = NMF(
        n_components=latent_dim,
        init="random",
        random_state=seed,
        max_iter=5000,
        beta_loss=nmf_loss,
        solver=solver,
    )
    W = nmf.fit_transform(panel)
    rate = W @ nmf.components_

    rng = np.random.default_rng(seed)
    prior_draw = rng.poisson(rate)
    if model_mismatch_p > 0.0:
        for t in range(1, n_time):
            # Upstream reads t's last decimal digit, which is t mod 10 for t >= 0.
            low = int((t % 10) < 5)
            innovation = 0.9 + rng.beta(1 + 2 * low, 1 + 2 * (1 - low)) * 0.2
            previous = rng.poisson(rate[t - 1]) if recursive_mismatch else prior_draw[t - 1]
            rate[t] = (1 - model_mismatch_p) * rate[t] + model_mismatch_p * (
                previous * innovation
            )

    n_post = n_time - intervention_t
    untreated_rate = rate[intervention_t:, -1].copy()
    multiplier = _effect_multiplier(n_post, alpha, effect_form)
    rate[intervention_t:, -1] *= multiplier

    observed = rng.poisson(rate)
    if ground_truth == "rate":
        truth = untreated_rate
    elif ground_truth == "draw":
        truth = rng.poisson(untreated_rate).astype(float)
    else:
        raise ValueError(f"unknown ground_truth {ground_truth!r}")

    return SemiSyntheticPanel(
        observed=observed.astype(float),
        truth=truth,
        intervention_t=intervention_t,
        multiplier=multiplier,
    )
