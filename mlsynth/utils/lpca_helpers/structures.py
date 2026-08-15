"""Frozen containers for the LPCA estimator.

Feng (2024), *Optimal Estimation of Large-Dimensional Nonlinear Factor Models*.
Local PCA reconstructs the treated unit's untreated outcome from a truncated
SVD of its nearest neighbours, so the treatment effect is observed minus
reconstructed over the post-treatment periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class LPCAInputs:
    """Preprocessed panel for LPCA.

    Attributes
    ----------
    Y : np.ndarray
        Observed outcomes, shape ``(N, T)``.
    Y_masked : np.ndarray
        ``Y`` with the treated post-treatment cells set to ``NaN`` -- the
        entries the method treats as missing.
    D : np.ndarray
        Treatment indicators, shape ``(N, T)``.
    treated_idx : np.ndarray
        Indices of ever-treated units.
    T0 : int
        First treated period.
    match_periods : int
        Number of leading periods reserved for nearest-neighbour matching; the
        remaining ``T - match_periods`` periods carry the local SVD and are the
        window the result reports on.
    unit_names : list
    time_labels : np.ndarray
    """

    Y: np.ndarray
    Y_masked: np.ndarray
    D: np.ndarray
    treated_idx: np.ndarray
    T0: int
    match_periods: int
    unit_names: List[Any]
    time_labels: np.ndarray

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]


class LPCAResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.LPCA.fit`.

    An :class:`~mlsynth.config_models.EffectResult`: alongside the
    LPCA-specific fields below it exposes the standardized sub-models
    (``effects``, ``time_series``, ``weights``, ``inference``,
    ``fit_diagnostics``, ``method_details``) and the flat accessors ``att`` /
    ``counterfactual`` / ``gap`` / ``att_ci`` / ``pre_rmse``.

    The reported series cover the PCA block only. Local PCA predicts the
    periods held out from matching, so the first ``match_periods`` periods have
    no counterfactual and are excluded from ``time_series`` instead of padded;
    ``time_series.time_periods`` carries the labels that remain.

    Parameters
    ----------
    inputs : LPCAInputs
    counterfactual_matrix : np.ndarray
        Reconstructed untreated outcome for the treated units on the PCA block,
        shape ``(n_treated, T - match_periods)``, on the outcome's scale.
    neighbours : list
        Names of the treated unit's matched neighbours, itself included. With
        several treated units these describe the first; the rest are in
        ``neighbours_by_unit``.
    neighbours_by_unit : dict
        ``{treated unit name: list of neighbour names}``.
    neighbour_weights : dict
        ``{neighbour name: weight}`` for the treated unit. The reconstruction
        is literally this weighted combination of the neighbourhood's rows.
        Being a projection and not a simplex fit, the weights neither sum to
        one nor stay non-negative.
    self_weight : float
        The treated unit's own weight in its reconstruction. The treated row
        belongs to its own neighbourhood, so this measures how much the
        counterfactual leans on the post-treatment cells the estimator set to
        the period mean -- the perturbation Theorem 6.1 bounds. A large value
        on a long post-period is the warning sign.
    n_neighbours : int
        Requested neighbourhood size ``K``.
    neighbourhood_size : int
        Realised size for the treated unit. Exceeds ``n_neighbours`` when the
        distance ties, which discrete panels do routinely.
    local_rank : int
        Components kept by the singular-value ratio rule. Pinned at
        ``max_components - 1`` whenever ``n_neighbours <= 15``, where the rule
        cannot fire -- see :func:`~mlsynth.utils.lpca_helpers.core.select_rank`.
    singular_values : np.ndarray
        Leading singular values of the treated unit's neighbourhood.
    period_means : np.ndarray, optional
        Per-period means removed before matching and added back to the
        counterfactual; ``None`` when ``demean`` is False.
    metadata : dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: LPCAInputs
    counterfactual_matrix: np.ndarray
    neighbours: List[Any]
    neighbours_by_unit: Dict[Any, List[Any]]
    neighbour_weights: Dict[Any, float]
    self_weight: float
    n_neighbours: int
    neighbourhood_size: int
    local_rank: int
    singular_values: np.ndarray
    period_means: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)
