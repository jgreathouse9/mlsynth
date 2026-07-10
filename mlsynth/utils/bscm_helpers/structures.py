"""Structured containers for the BSCM estimator.

Containers for the Bayesian Synthetic Control Methods of Kim, Lee & Gupta
(2020), Journal of Marketing Research 57(5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class BSCMInputs:
    """Panel data fed into the BSCM Gibbs sampler.

    BSCM fits with an explicit intercept (the reference model's
    ``beta_0``), so the donor blocks are kept on their original scale --
    no demeaning.

    Parameters
    ----------
    y_pre : np.ndarray
        Length-``T0`` treated pre-treatment outcome.
    X_pre : np.ndarray
        Shape ``(T0, N)`` donor matrix over the pre-treatment window.
    X_all : np.ndarray
        Shape ``(T, N)`` donor matrix over all periods (used to form the
        counterfactual path, pre and post).
    y_target : np.ndarray
        Length-``T`` treated outcome over all periods.
    T0 : int
        Number of pre-treatment periods.
    T : int
        Total number of periods.
    N : int
        Number of donor units.
    treated_unit_name : str
    donor_names : Sequence
    time_labels : np.ndarray
    """

    y_pre: np.ndarray
    X_pre: np.ndarray
    X_all: np.ndarray
    y_target: np.ndarray
    T0: int
    T: int
    N: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray


@dataclass(frozen=True)
class BSCMPosterior:
    """Pooled MCMC samples drawn by the BSCM Gibbs sampler.

    Parameters
    ----------
    beta0 : np.ndarray
        Length-``n_samples`` posterior samples of the intercept.
    beta : np.ndarray
        Shape ``(N, n_samples)`` posterior samples of the donor weights.
    sigma2 : np.ndarray
        Length-``n_samples`` posterior samples of the error variance.
    gamma : np.ndarray or None
        Shape ``(N, n_samples)`` 0/1 inclusion indicators for the
        ``spike_slab`` prior; ``None`` for the horseshoe.
    prior : str
        ``"horseshoe"`` or ``"spike_slab"``.
    burn_in : int
        Warm-up iterations dropped per chain.
    n_iter : int
        Total iterations per chain (including burn-in).
    chains : int
        Number of chains pooled into these samples.
    """

    beta0: np.ndarray
    beta: np.ndarray
    sigma2: np.ndarray
    gamma: Optional[np.ndarray]
    prior: str
    burn_in: int
    n_iter: int
    chains: int


@dataclass(frozen=True)
class BSCMInference:
    """Point estimate plus credible interval for the ATT.

    Parameters
    ----------
    att_mean : float
        Posterior mean ATT over the post-treatment horizon (``np.nan``
        when no post window exists).
    att_ci_lower, att_ci_upper : float
        Credible interval bounds at level ``1 - ci_alpha``.
    att_samples : np.ndarray
        Length-``n_samples`` per-draw ATT.
    ci_alpha : float
        Significance level used to build the interval.
    counterfactual_mean : np.ndarray
        Length-``T`` posterior-mean counterfactual.
    counterfactual_lower, counterfactual_upper : np.ndarray
        Length-``T`` pointwise credible bands.
    """

    att_mean: float
    att_ci_lower: float
    att_ci_upper: float
    att_samples: np.ndarray
    ci_alpha: float
    counterfactual_mean: np.ndarray
    counterfactual_lower: np.ndarray
    counterfactual_upper: np.ndarray


class BSCMResults(BaseEstimatorResults):
    """Public ``BSCM.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational
    report): it populates the standardized sub-models so the flat accessors
    (``att`` / ``att_ci`` / ``counterfactual`` / ``gap`` / ``donor_weights`` /
    ``pre_rmse``) resolve through the base contract. ``att`` is the posterior
    mean ATT and ``att_ci`` its credible interval; ``donor_weights`` are the
    posterior mean weights (which may be negative). The Bayesian detail -- the
    MCMC posterior, the per-draw ATT samples, the pointwise counterfactual
    bands, and (for ``spike_slab``) the inclusion probabilities -- stays in the
    typed fields below.

    Parameters
    ----------
    inputs : BSCMInputs
    posterior : BSCMPosterior
    inference_detail : BSCMInference
        Posterior ATT / counterfactual bands. The standardized ``inference``
        slot holds the ATT-level
        :class:`~mlsynth.config_models.InferenceResults`.
    inclusion_probs : dict or None
        ``donor_label -> P(gamma_i = 1 | y)`` for ``spike_slab``; ``None``
        for the horseshoe.
    weight_means : dict
        ``donor_label -> E[beta_i | y]`` posterior mean weights.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: BSCMInputs
    posterior: BSCMPosterior
    inference_detail: BSCMInference
    inclusion_probs: Optional[dict]
    weight_means: dict


# Resolve forward references (module uses ``from __future__ import annotations``).
BSCMResults.model_rebuild()
