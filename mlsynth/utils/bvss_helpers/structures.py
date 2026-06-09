"""Structured containers for the BVS-SS estimator.

Implements containers for the BVS-SS pipeline of Xu & Zhou (2025),
arXiv:2503.06454.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class BVSSInputs:
    """Demeaned panel data fed into the Gibbs sampler.

    Parameters
    ----------
    Y_pre_demean : np.ndarray
        Length-``T0`` demeaned treated pre-treatment outcome.
    X_pre_demean : np.ndarray
        Shape ``(T0, N)`` demeaned donor matrix over the pre-treatment
        window.
    X_post_demean : np.ndarray or None
        Shape ``(T_post, N)`` demeaned donor matrix over the post window
        (uses the pre-treatment column means). ``None`` if there is no
        post period.
    Gram : np.ndarray
        Pre-computed ``X_pre_demean.T @ X_pre_demean``.
    mean_Y : float
        Pre-treatment mean of the treated outcome, used to undo the
        demeaning when forming counterfactual paths.
    mean_X : np.ndarray
        Length-``N`` per-donor pre-treatment means.
    T0 : int
        Number of pre-treatment periods.
    T : int
        Total number of periods.
    N : int
        Number of donor units.
    treated_unit_name : str
    donor_names : Sequence
    time_labels : np.ndarray
    y_target : np.ndarray
        Original (un-demeaned) treated outcome over all ``T`` periods.
    """

    Y_pre_demean: np.ndarray
    X_pre_demean: np.ndarray
    X_post_demean: Optional[np.ndarray]
    Gram: np.ndarray
    mean_Y: float
    mean_X: np.ndarray
    T0: int
    T: int
    N: int
    treated_unit_name: str
    donor_names: Sequence
    time_labels: np.ndarray
    y_target: np.ndarray


@dataclass(frozen=True)
class BVSSPosterior:
    """MCMC samples drawn by the BVS-SS Gibbs sampler.

    Parameters
    ----------
    mu : np.ndarray
        Shape ``(N, n_samples)`` posterior samples of ``\\mu`` (after
        burn-in).
    phi : np.ndarray
        Length-``n_samples`` posterior samples of ``\\phi``.
    tau : np.ndarray
        Length-``n_samples`` posterior samples of ``\\tau``.
    gamma : np.ndarray
        Shape ``(N, n_samples)`` 0/1 inclusion indicators implied by
        ``mu``.
    burn_in : int
        Number of warm-up iterations dropped before this slice.
    n_iter : int
        Total iterations the chain ran for (including burn-in).
    """

    mu: np.ndarray
    phi: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    burn_in: int
    n_iter: int


@dataclass(frozen=True)
class BVSSInference:
    """Point estimate plus credible interval for the ATT.

    Parameters
    ----------
    att_mean : float
        Posterior mean ATT over the post-treatment horizon.
        ``np.nan`` when no post window exists.
    att_ci_lower, att_ci_upper : float
        Credible interval bounds at level ``1 - ci_alpha``.
    att_samples : np.ndarray
        Length-``n_samples`` per-MCMC-draw ATT.
    ci_alpha : float
        Significance level used to build the interval.
    counterfactual_mean : np.ndarray
        Length-``T`` posterior-mean counterfactual (in original outcome
        units, not demeaned).
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


class BVSSResults(BaseEstimatorResults):
    """Public ``BVSS.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    it populates the standardized sub-models so the flat accessors (``att`` /
    ``att_ci`` / ``counterfactual`` / ``gap`` / ``donor_weights`` /
    ``pre_rmse``) resolve through the base contract. ``att`` is the posterior
    mean ATT and ``att_ci`` its credible interval; ``donor_weights`` are the
    posterior mean weights. The full Bayesian detail -- the MCMC posterior, the
    per-draw ATT samples, the pointwise counterfactual bands, and the inclusion
    probabilities -- stays in the typed fields below.

    Parameters
    ----------
    inputs : BVSSInputs
    posterior : BVSSPosterior
    inference_detail : BVSSInference
        Posterior ATT / counterfactual bands (was ``inference`` before the
        contract migration; the standardized ``inference`` slot now holds the
        ATT-level :class:`~mlsynth.config_models.InferenceResults`).
    inclusion_probs : dict
        ``donor_label -> P(\\gamma_i = 1 | y)`` posterior inclusion
        frequencies over the post burn-in samples.
    weight_means : dict
        ``donor_label -> E[\\mu_i | y]`` posterior mean weights.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: BVSSInputs
    posterior: BVSSPosterior
    inference_detail: BVSSInference
    inclusion_probs: dict
    weight_means: dict


# Resolve forward references (module uses ``from __future__ import annotations``).
BVSSResults.model_rebuild()
