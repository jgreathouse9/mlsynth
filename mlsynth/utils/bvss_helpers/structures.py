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


@dataclass(frozen=True)
class BVSSSimplexDiagnostics:
    """How far the data let the weights drift off the simplex.

    Equation (2) of Xu and Zhou (2025) gives the actual weights

    .. math::

        \\mathbf{w}_\\gamma \\mid \\gamma, \\boldsymbol{\\mu}_\\gamma, \\tau, \\phi
        \\sim N(\\boldsymbol{\\mu}_\\gamma, (\\tau / \\phi) I),

    so they are centred on a point of the simplex and scattered around it with
    standard deviation :math:`\\sqrt{\\tau / \\phi}` per coordinate. Section 2.1
    reads that scatter as the data's verdict on the constraint: a posterior for
    :math:`\\tau` concentrating near zero says the simplex is appropriate, and
    one staying away from zero says it is violated.

    Every field is analytic in the draws of :math:`\\tau`, :math:`\\phi` and
    :math:`|\\gamma|`, so none of it depends on how the counterfactual is built
    -- which matters, because that construction is an open question recorded in
    ``docs/bvss.rst``.

    Parameters
    ----------
    tau_mean, tau_median, tau_q025, tau_q975 : float
        Posterior summaries of :math:`\\tau`.
    deviation_scale : float
        :math:`E[\\sqrt{\\tau / \\phi}]`, the per-coordinate standard deviation
        of a weight about its simplex centre, in weight units.
    weight_sum_sd : float
        :math:`E[\\sqrt{|\\gamma| \\tau / \\phi}]`. The coordinates are
        independent given the parameters, so the variances add and the sum of
        the weights departs from one on this scale.
    relative_deviation : float
        ``deviation_scale`` against a typical weight :math:`1 / |\\gamma|`. A
        scatter of 0.05 means one thing across three donors and another across
        thirty, and this is the comparable number.
    model_size_mean : float
        Posterior mean :math:`|\\gamma|`, carried so the two scales above can
        be read without a second pass over the draws.
    """

    tau_mean: float
    tau_median: float
    tau_q025: float
    tau_q975: float
    deviation_scale: float
    weight_sum_sd: float
    relative_deviation: float
    model_size_mean: float


def simplex_diagnostics(posterior: "BVSSPosterior") -> BVSSSimplexDiagnostics:
    """Summarise the posterior's verdict on the simplex constraint.

    Parameters
    ----------
    posterior : BVSSPosterior
        Post burn-in draws of ``mu``, ``phi`` and ``tau``.

    Returns
    -------
    BVSSSimplexDiagnostics
    """
    tau = np.asarray(posterior.tau, dtype=float)
    phi = np.asarray(posterior.phi, dtype=float)
    size = (np.asarray(posterior.mu) != 0).sum(axis=0).astype(float)

    ratio = tau / phi
    deviation = float(np.sqrt(ratio).mean())
    mean_size = float(size.mean())
    return BVSSSimplexDiagnostics(
        tau_mean=float(tau.mean()),
        tau_median=float(np.median(tau)),
        tau_q025=float(np.percentile(tau, 2.5)),
        tau_q975=float(np.percentile(tau, 97.5)),
        deviation_scale=deviation,
        weight_sum_sd=float(np.sqrt(size * ratio).mean()),
        relative_deviation=deviation * mean_size,
        model_size_mean=mean_size,
    )


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
    simplex : BVSSSimplexDiagnostics
        What the posterior says about the simplex constraint the model softens.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: BVSSInputs
    posterior: BVSSPosterior
    inference_detail: BVSSInference
    inclusion_probs: dict
    weight_means: dict
    simplex: BVSSSimplexDiagnostics


# Resolve forward references (module uses ``from __future__ import annotations``).
BVSSResults.model_rebuild()
