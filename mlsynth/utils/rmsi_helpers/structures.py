"""Frozen dataclasses for the RMSI estimator.

Agarwal, Choi & Yuan (2026), *Robust Matrix Estimation with Side Information*
(arXiv:2603.24833). RMSI imputes the treated counterfactual of a block-adoption
causal panel by a four-component sieve + nuclear-norm matrix estimator that
exploits unit-level (row) and time-level (column) covariates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class RMSIInputs:
    """Preprocessed block-adoption panel with side information for RMSI.

    Attributes
    ----------
    Y : np.ndarray
        Outcomes, shape ``(N, T)``.
    D : np.ndarray
        Treatment indicators, shape ``(N, T)`` (``1`` where treated).
    X : np.ndarray
        Row (unit) covariates, shape ``(N, d1)`` -- one row per unit.
    Z : np.ndarray
        Column (time) covariates, shape ``(T, d2)`` -- one row per period.
    T0 : int
        Number of clean pre-treatment periods (common adoption at ``T0``).
    treated_idx, control_idx : np.ndarray
        Indices of ever-treated and never-treated units.
    unit_names, time_labels : sequence
    unit_covariates, time_covariates : list of str
        The covariate column names used to build ``X`` and ``Z``.
    """

    Y: np.ndarray
    D: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    T0: int
    treated_idx: np.ndarray
    control_idx: np.ndarray
    unit_names: List[Any]
    time_labels: np.ndarray
    unit_covariates: List[str]
    time_covariates: List[str]

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]


class RMSIResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.RMSI.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the RMSI-specific fields below it exposes the standardized
    sub-models (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``pre_rmse``. The treated counterfactual path
    (``res.counterfactual``) is the cross-treated-unit synthetic mean; the full
    imputed ``(N, T)`` matrix lives in ``counterfactual_matrix``.

    Parameters
    ----------
    inputs : RMSIInputs
    counterfactual_matrix : np.ndarray
        Estimated untreated potential outcomes ``M_hat``, shape ``(N, T)``.
        (Renamed from ``counterfactual``, which now returns the 1-D treated
        path per the result contract.)
    effects_matrix : np.ndarray
        Observed minus imputed on treated cells (NaN elsewhere), shape
        ``(N, T)``. (Renamed from ``effects``, which is now the standardized
        :class:`~mlsynth.config_models.EffectsResults` slot.)
    att_by_period : dict
        ``{time_label: ATT}`` over the post-treatment periods.
    treated_mean, synthetic_mean : np.ndarray
        Cross-treated-unit means of observed / imputed over the full timeline
        (the series the plotter draws), length ``T``.
    rank : int
        Factor rank used in the block recombination.
    components : dict, optional
        The four components of the "tall" fit (diagnostic), if retained.
    metadata : dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: RMSIInputs
    counterfactual_matrix: np.ndarray
    effects_matrix: np.ndarray
    att_by_period: Dict[Any, float]
    treated_mean: np.ndarray
    synthetic_mean: np.ndarray
    rank: int
    components: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)
