"""Frozen dataclasses for the Synthetic Nearest Neighbors (SNN) estimator.

Agarwal, A., Dahleh, M., Shah, D. & Shen, D. (2021). *"Causal Matrix
Completion."* arXiv:2109.15154.

In the causal/panel setting, SNN treats the treated units' post-treatment
potential outcomes :math:`Y(0)` as the missing entries of the outcome
matrix and imputes them by matrix completion (synthetic nearest
neighbors), then forms treatment effects as observed minus imputed. It
generalises the Synthetic Interventions / synthetic-control estimator to
arbitrary missingness patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SNNInputs:
    """Preprocessed panel for SNN.

    Attributes
    ----------
    Y : np.ndarray
        Observed outcomes, shape ``(N, T)``.
    D : np.ndarray
        Treatment indicators, shape ``(N, T)``; ``1`` where treated.
    treated_idx : np.ndarray
        Indices of ever-treated units.
    T0 : int
        First treated period (post-treatment is ``t >= T0``).
    unit_names : list
        Length-``N`` unit identifiers.
    time_labels : np.ndarray
        Length-``T`` period labels.
    """

    Y: np.ndarray
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


class SNNResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SNN.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the SNN-specific fields below it exposes the standardized
    sub-models (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``att_ci`` / ``pre_rmse``. The treated
    counterfactual path (``res.counterfactual``) is the cross-treated-unit mean
    of the imputed :math:`Y(0)`; the full imputed ``(N, T)`` matrix lives in
    ``counterfactual_matrix``.

    Parameters
    ----------
    inputs : SNNInputs
        Preprocessed panel.
    counterfactual_matrix : np.ndarray
        Outcome matrix with treated post-treatment :math:`Y(0)` imputed,
        shape ``(N, T)``. (Renamed from ``counterfactual``, which now returns
        the 1-D treated path per the result contract.)
    effects_matrix : np.ndarray
        Per-cell treatment effects (observed minus imputed) for treated
        post cells; ``NaN`` elsewhere, shape ``(N, T)``. (Renamed from
        ``effects``, which is now the standardized ``EffectsResults`` slot.)
    att_by_period : dict
        ``{period_label: mean effect across treated units}`` for
        post-treatment periods.
    feasible : np.ndarray
        Boolean mask of cells SNN could impute, shape ``(N, T)``.
    inference_jackknife : SNNInference, optional
        The raw jackknife inference object (``method`` / ``se`` / ``ci``) when
        ``inference=True``; ``None`` otherwise. The standardized
        :class:`~mlsynth.config_models.InferenceResults` is mirrored into the
        ``inference`` slot (so ``res.att_ci`` resolves).
    metadata : dict
        Free-form diagnostics.

    Notes
    -----
    The PCR donor weights (the linear combination that builds the
    counterfactual) live in the standardized ``weights`` slot: for a single
    treated unit ``donor_weights`` maps donor name -> weight; with multiple
    treated units it holds the cross-unit average and the per-unit weights live
    in ``summary_stats['per_unit_donor_weights']``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SNNInputs
    counterfactual_matrix: np.ndarray
    effects_matrix: np.ndarray
    att_by_period: Dict[Any, float]
    feasible: np.ndarray
    inference_jackknife: Optional["SNNInference"] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)


@dataclass(frozen=True)
class SNNInference:
    """Jackknife inference for the SNN ATT.

    Leaves out one control (anchor) unit at a time, re-imputes, and uses
    the spread of the resulting ATTs to form a standard error and
    confidence interval.

    Attributes
    ----------
    method : str
        ``"jackknife"``.
    se : float
        Jackknife standard error of the ATT.
    ci : tuple of float
        Two-sided confidence interval for the ATT.
    alpha_level : float
        Level used for ``ci``.
    n_jackknife : int
        Number of leave-one-control re-fits used.
    """

    method: str
    se: float
    ci: tuple
    alpha_level: float
    n_jackknife: int


# Resolve the forward reference to SNNInference now that it is defined.
SNNResults.model_rebuild()
