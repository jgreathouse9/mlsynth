"""Structured containers for the CMBSTS pipeline.

Outcomes are kept time-major ``Y in R^{T x d}`` (rows = time, columns = the
``d`` jointly-modelled series, the treated series first) to match the
multivariate state-space convention of Menchetti and Bojinov (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class CMBSTSInputs:
    """Pre-processed inputs fed into the MBSTS sampler.

    Parameters
    ----------
    Y : np.ndarray
        ``(T, d)`` outcome matrix; the treated series is column 0.
    X : np.ndarray, optional
        ``(T, k)`` regression matrix (controls + covariates); None if neither.
    T0, T, d : int
        Pre-period length, total periods, number of series.
    series_names : list
        Length-``d`` labels (treated unit first, then ``group_units``).
    control_names, covariate_names : list
        Column provenance of ``X`` (control units, then covariate columns).
    time_labels : np.ndarray
        Length-``T`` time index.
    intervention_time : Any
        Label at the pre/post boundary.
    excl_post : np.ndarray, optional
        0/1 mask over the post window flagging dropped dates.
    """

    Y: np.ndarray
    X: Optional[np.ndarray]
    T0: int
    T: int
    d: int
    series_names: List[Any]
    control_names: List[Any]
    covariate_names: List[Any]
    time_labels: np.ndarray
    intervention_time: Any
    excl_post: Optional[np.ndarray]

    @property
    def pre_periods(self) -> int:
        return int(self.T0)

    @property
    def post_periods(self) -> int:
        return int(self.T - self.T0)


@dataclass(frozen=True)
class CMBSTSPosterior:
    """Summary of the Gibbs run (kept light; full draws are not retained).

    Parameters
    ----------
    inclusion_probs : dict, optional
        ``regressor -> P(included | y)`` posterior inclusion frequencies, when
        a regression block is present.
    n_draws : int
        Post-burn-in draws.
    niter, burn : int
        Total iterations and burn-in.
    components : list of str
        Structural components used.
    seas_period, cycle_period : int, optional
        Periods, when the corresponding component is present.
    """

    inclusion_probs: Optional[Dict[str, float]]
    n_draws: int
    niter: int
    burn: int
    components: List[str]
    seas_period: Optional[int]
    cycle_period: Optional[int]


@dataclass(frozen=True)
class CMBSTSInference:
    """Per-series posterior causal effects and counterfactual bands.

    All per-series arrays are length ``d`` (treated series first); path arrays
    are ``(post_periods, d)`` or ``(T, d)``.

    Parameters
    ----------
    series_names : list
        The ``d`` series labels.
    att_mean, att_lower, att_upper : np.ndarray
        Temporal-average effect and its credible bounds, per series.
    cum_mean, cum_lower, cum_upper : np.ndarray
        Cumulative effect and bounds, per series.
    effect_path, effect_lower, effect_upper : np.ndarray
        ``(post_periods, d)`` per-period effect and pointwise bands.
    counterfactual_post : np.ndarray
        ``(post_periods, d)`` posterior-mean counterfactual.
    counterfactual_full : np.ndarray
        ``(T, d)`` counterfactual over the whole horizon (in-sample fit then
        forecast) for plotting.
    att_samples : np.ndarray
        ``(n_draws, d)`` per-draw temporal-average effect.
    ci_alpha : float
        Two-sided level of the bands.
    """

    series_names: List[Any]
    att_mean: np.ndarray
    att_lower: np.ndarray
    att_upper: np.ndarray
    cum_mean: np.ndarray
    cum_lower: np.ndarray
    cum_upper: np.ndarray
    effect_path: np.ndarray
    effect_lower: np.ndarray
    effect_upper: np.ndarray
    counterfactual_post: np.ndarray
    counterfactual_full: np.ndarray
    att_samples: np.ndarray
    ci_alpha: float


class CMBSTSResults(BaseEstimatorResults):
    """Public ``CMBSTS.fit()`` return container.

    An :class:`~mlsynth.config_models.EffectResult`: the standardized sub-models
    (``effects`` / ``time_series`` / ``weights`` / ``inference`` /
    ``fit_diagnostics``) and the flat accessors (``att`` / ``att_ci`` /
    ``counterfactual`` / ``gap``) resolve over the treated series (series 0).
    ``att`` is its posterior-mean temporal-average effect and ``att_ci`` the
    credible interval. The full multivariate detail -- the per-series effects,
    cumulative effects, counterfactual bands, posterior draws, and inclusion
    probabilities -- lives in the typed fields below. CMBSTS is a state-space
    estimator with no donor weights, so ``weights`` is an empty
    :class:`~mlsynth.config_models.WeightsResults` carrying a method note.

    Parameters
    ----------
    inputs : CMBSTSInputs
    posterior : CMBSTSPosterior
    inference_detail : CMBSTSInference
        Per-series posterior effects and counterfactual bands (the standardized
        ``inference`` slot holds the treated-series ATT-level
        :class:`~mlsynth.config_models.InferenceResults`).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: CMBSTSInputs
    posterior: CMBSTSPosterior
    inference_detail: CMBSTSInference
