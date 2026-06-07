"""Frozen dataclasses for the MC-NNM estimator.

Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *Matrix Completion
Methods for Causal Panel Data Models* (JASA). MC-NNM imputes the missing
(treated) entries of the outcome matrix by nuclear-norm-regularised
low-rank matrix completion with unregularised two-way fixed effects, then
forms treatment effects as observed minus imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class MCNNMInputs:
    """Preprocessed panel for MC-NNM.

    Attributes
    ----------
    Y : np.ndarray
        Observed outcomes, shape ``(N, T)``.
    mask : np.ndarray
        Observation indicator, shape ``(N, T)``; ``1`` observed (control
        / pre-treatment), ``0`` missing (treated post-treatment).
    D : np.ndarray
        Treatment indicators, shape ``(N, T)`` (``1`` where treated).
    treated_idx : np.ndarray
        Indices of ever-treated units.
    T0 : int
        First treated period.
    unit_names : list
    time_labels : np.ndarray
    """

    Y: np.ndarray
    mask: np.ndarray
    D: np.ndarray
    treated_idx: np.ndarray
    T0: int
    unit_names: List[Any]
    time_labels: np.ndarray

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]


class MCNNMResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.MCNNM.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the MC-NNM-specific fields below it exposes the standardized
    sub-models (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``att_ci`` / ``pre_rmse``. The treated
    counterfactual path (``res.counterfactual``) is the cross-treated-unit mean
    of the imputed untreated outcome; the full ``(N, T)`` fitted matrix lives in
    ``counterfactual_matrix``.

    Parameters
    ----------
    inputs : MCNNMInputs
    counterfactual_matrix : np.ndarray
        Full fitted matrix ``L + Gamma + Delta``, shape ``(N, T)``; on
        treated cells this is the imputed untreated potential outcome.
        (Renamed from ``counterfactual``, which now returns the 1-D treated
        path per the result contract.)
    effects_matrix : np.ndarray
        Per-cell effects (observed minus imputed) on treated cells; ``NaN``
        elsewhere, shape ``(N, T)``. (Renamed from ``effects``, which is now
        the standardized ``EffectsResults`` slot.)
    att_by_period : dict
        ``{period_label: mean effect across treated units}`` post-treatment
        (calendar time -- pools cohorts at each period).
    cohort_att : dict
        ``{adoption_time_label: mean ATT for that adoption cohort}`` -- the
        cohort-specific effects under staggered adoption.
    event_study : dict
        ``{relative_time: mean effect across treated cells at that event
        time}`` where relative time is ``period - adoption period`` for each
        treated unit. Negative keys are pre-adoption (a placebo / fit-quality
        check, ~0); non-negative keys are the dynamic treatment effects.
    L : np.ndarray
        Estimated low-rank matrix, shape ``(N, T)``.
    gamma : np.ndarray
        Unit fixed effects, shape ``(N,)``.
    delta : np.ndarray
        Time fixed effects, shape ``(T,)``.
    best_lambda : float
        Cross-validation-selected singular-value threshold.
    rank : int
        Numerical rank of ``L`` (singular values > 1e-6 of the max).
    unit_factors : np.ndarray
        Unit loadings :math:`U \\Sigma^{1/2}` from the SVD of ``L``, shape
        ``(N, rank)`` -- MC-NNM's "under the hood": each unit's position in
        the latent factor space.
    time_factors : np.ndarray
        Time factors :math:`V \\Sigma^{1/2}`, shape ``(T, rank)``.
    singular_values : np.ndarray
        Singular values of ``L`` (the nuclear-norm spectrum).
    inference_jackknife : MCNNMInference, optional
        The raw leave-one-control jackknife object (``method`` / ``se`` /
        ``ci``) when ``inference=True``; else ``None``. The standardized
        :class:`~mlsynth.config_models.InferenceResults` is mirrored into the
        ``inference`` slot (so ``res.att_ci`` resolves).
    metadata : dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MCNNMInputs
    counterfactual_matrix: np.ndarray
    effects_matrix: np.ndarray
    att_by_period: Dict[Any, float]
    cohort_att: Dict[Any, float]
    event_study: Dict[int, float]
    L: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    best_lambda: float
    rank: int
    unit_factors: Optional[np.ndarray] = None
    time_factors: Optional[np.ndarray] = None
    singular_values: Optional[np.ndarray] = None
    inference_jackknife: Optional["MCNNMInference"] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)


@dataclass(frozen=True)
class MCNNMInference:
    """Leave-one-control jackknife inference for the MC-NNM ATT.

    Attributes
    ----------
    method : str
        ``"jackknife"``.
    se : float
    ci : tuple of float
    alpha_level : float
    n_jackknife : int
    """

    method: str
    se: float
    ci: tuple
    alpha_level: float
    n_jackknife: int


# Resolve the forward reference to MCNNMInference now that it is defined.
MCNNMResults.model_rebuild()
