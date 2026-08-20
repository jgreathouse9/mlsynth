"""Structured containers for the BLTVP estimator.

Containers for the Time Varying Parameter Bayesian Lasso of Klinenberg (2023),
Journal of Business & Economic Statistics 41(4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class BLTVPInputs:
    """Panel data fed into the BLTVP Gibbs sampler.

    The donor blocks are kept on their original scale: the coefficients carry
    the level, and the optional intercept is added inside the sampler.

    Parameters
    ----------
    y_pre : np.ndarray
        Length-``T0`` treated pre-treatment outcome.
    X_pre : np.ndarray
        Shape ``(T0, N)`` donor matrix over the pre-treatment window.
    X_all : np.ndarray
        Shape ``(T, N)`` donor matrix over all periods.
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
class BLTVPPosterior:
    """Pooled MCMC samples drawn by the BLTVP Gibbs sampler.

    ``beta`` and ``sqrt_theta`` are the two halves of the decomposition in
    eq. (7): the coefficient on donor ``j`` at time ``t`` is
    ``beta_j + btil_{j,t} sqrt_theta_j``. A donor whose ``sqrt_theta_j`` is
    shrunk to zero has a static relationship to the treated unit; one whose
    ``beta_j`` is also shrunk to zero is irrelevant.

    ``sqrt_theta`` is signed, and only its magnitude is identified -- the
    sampler's sign switch makes the sign symmetric by construction, so read
    ``abs(sqrt_theta)`` as the scale of the drift.

    Parameters
    ----------
    beta : np.ndarray
        Shape ``(m, n_samples)`` draws of the time-invariant coefficients,
        where ``m`` is ``N`` (or ``N + 1`` with an intercept).
    sqrt_theta : np.ndarray
        Shape ``(m, n_samples)`` draws of the drift scales. All zero when the
        fit was static.
    sigma2 : np.ndarray
        Length-``n_samples`` draws of the observation variance.
    predictive : np.ndarray
        Shape ``(T, n_samples)`` posterior predictive draws of the untreated
        potential outcome over every period.
    burn_in, n_iter, chains : int
        Sampler settings these draws came from.
    intercept : bool
        Whether an intercept column was carried.
    time_varying : bool
        Whether the coefficients were allowed to drift.
    """

    beta: np.ndarray
    sqrt_theta: np.ndarray
    sigma2: np.ndarray
    predictive: np.ndarray
    burn_in: int
    n_iter: int
    chains: int
    intercept: bool
    time_varying: bool


@dataclass(frozen=True)
class BLTVPInference:
    """Point estimate plus credible interval for the ATT.

    Parameters
    ----------
    att_mean : float
        Posterior mean ATT over the post-treatment horizon (``np.nan`` when
        no post window exists).
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


class BLTVPResults(BaseEstimatorResults):
    """Public ``BLTVP.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult`: it populates the
    standardized sub-models so the flat accessors (``att`` / ``att_ci`` /
    ``counterfactual`` / ``gap`` / ``donor_weights`` / ``pre_rmse``) resolve
    through the base contract. ``att`` is the posterior mean ATT and
    ``att_ci`` its credible interval; ``donor_weights`` are the posterior mean
    time-invariant coefficients, which may be negative and need not sum to
    one.

    The decomposition the method exists for lives in ``dynamic_scales``: the
    posterior mean of ``abs(sqrt_theta_j)`` per donor, which says how much
    that donor's relationship to the treated unit drifts. Read it alongside
    ``weight_means``, the static half of the same coefficient.

    Under a Bayesian Lasso no coefficient is set exactly to zero (p. 1068), so
    there is no inclusion probability to report here and none is invented --
    these are shrinkage magnitudes, to be read as such.

    Parameters
    ----------
    inputs : BLTVPInputs
    posterior : BLTVPPosterior
    inference_detail : BLTVPInference
        Posterior ATT / counterfactual bands. The standardized ``inference``
        slot holds the ATT-level
        :class:`~mlsynth.config_models.InferenceResults`.
    weight_means : dict
        ``donor_label -> E[beta_j | y]``, the time-invariant half.
    dynamic_scales : dict
        ``donor_label -> E[|sqrt(theta)_j| | y]``, the drift scale.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: BLTVPInputs
    posterior: BLTVPPosterior
    inference_detail: BLTVPInference
    weight_means: dict
    dynamic_scales: dict


# Resolve forward references (module uses ``from __future__ import annotations``).
BLTVPResults.model_rebuild()
