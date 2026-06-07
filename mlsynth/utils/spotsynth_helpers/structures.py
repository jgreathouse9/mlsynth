"""Frozen dataclasses for the SPOTSYNTH estimator.

O'Riordan & Gilligan-Lee (2025), *Spillover detection for donor selection in
synthetic control models*, Journal of Causal Inference 13:20240036
(doi:10.1515/jci-2024-0036). SPOTSYNTH screens every candidate donor for
spillover contamination by testing whether its first post-intervention value
can be forecast from pre-intervention donor data (Theorem 3.1, Algorithm 1),
excludes the donors that fail the test, and builds a synthetic control on the
donors judged *valid*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SpotSynthInputs:
    """Preprocessed single-treated-unit panel for SPOTSYNTH.

    Attributes
    ----------
    y : np.ndarray
        Treated-unit outcome series, length ``T``.
    D : np.ndarray
        Donor-pool outcomes, shape ``(T, n_donors)`` (columns = donors).
    T0 : int
        Number of pre-intervention periods (intervention at index ``T0``).
    donor_names : list
        Names of the donor-pool columns (aligned with ``D``).
    treated_name : Any
        Name of the treated unit.
    time_labels : np.ndarray
        The ``T`` period labels.
    """

    y: np.ndarray
    D: np.ndarray
    T0: int
    donor_names: List[Any]
    treated_name: Any
    time_labels: np.ndarray

    @property
    def T(self) -> int:
        return self.y.shape[0]

    @property
    def n_donors(self) -> int:
        return self.D.shape[1]


@dataclass(frozen=True)
class SpilloverScreen:
    """Per-donor output of the Algorithm 1 spillover screen.

    Attributes
    ----------
    donor_names : list
        Donor names, in the original pool order.
    forecast_error : np.ndarray
        Procedure ``S1``: absolute (normalised) forecast error
        :math:`A_i = |x_i^t - \\hat x_i^t|` at the screened post-intervention
        horizon. Smaller = more likely a *valid* donor.
    inside_ppi : np.ndarray of bool
        Procedure ``S2``: ``True`` where the realised post-intervention value
        falls inside the donor's forecast posterior predictive interval
        (i.e. ``B = 0`` in the paper -- judged valid).
    selected_idx : np.ndarray
        Indices (into the donor pool) of the donors judged valid and used to
        build the synthetic control.
    excluded_idx : np.ndarray
        Indices of the donors flagged as spillover-contaminated.
    selection : str
        ``"S1"`` or ``"S2"`` -- which procedure drove the selection.
    forecast : str
        ``"lag"`` (paper Algorithm 1) or ``"loo"`` (leave-one-out variant).
    metadata : dict
    """

    donor_names: List[Any]
    forecast_error: np.ndarray
    inside_ppi: np.ndarray
    selected_idx: np.ndarray
    excluded_idx: np.ndarray
    selection: str
    forecast: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def selected_names(self) -> List[Any]:
        return [self.donor_names[i] for i in self.selected_idx]

    @property
    def excluded_names(self) -> List[Any]:
        return [self.donor_names[i] for i in self.excluded_idx]


class SpotSynthResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SPOTSYNTH.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report):
    in addition to the SPOTSYNTH-specific fields below it exposes the
    standardized sub-models (``effects``, ``time_series``, ``weights``,
    ``inference``, ``fit_diagnostics``, ``method_details``) and the flat
    accessors ``att`` / ``att_ci`` / ``counterfactual`` / ``gap`` /
    ``donor_weights`` / ``pre_rmse``. The screened ATT is the post-period mean
    gap; ``att_ci`` reads the Dirichlet credible interval from ``inference``.

    Parameters
    ----------
    inputs : SpotSynthInputs
    screen : SpilloverScreen
        The per-donor spillover diagnostics and the valid-donor selection.
    att_by_period : dict
        ``{time_label: gap}`` over the post-intervention periods.
    att_unscreened : float
        ATT from a synthetic control on the *full* donor pool (the ``All``
        baseline) -- for comparison.
    inference_method : str
        ``"bayes"`` (Dirichlet posterior) or ``"frequentist"`` (simplex LS).
        (Renamed from the former ``inference`` field, which now holds the
        standardized :class:`~mlsynth.config_models.InferenceResults`.)
    counterfactual_lower, counterfactual_upper : np.ndarray, optional
        Posterior-predictive credible band for the counterfactual (length
        ``T``) under ``inference_method="bayes"``, else ``None``.
    att_debiased : float, optional
        Proximal (two-stage / GMM) debiased ATT using the excluded donors as
        proximal controls (when ``debias=True``), else ``None``.
    metadata : dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SpotSynthInputs
    screen: SpilloverScreen
    att_by_period: Dict[Any, float]
    att_unscreened: float
    inference_method: str = "frequentist"
    counterfactual_lower: Optional[np.ndarray] = None
    counterfactual_upper: Optional[np.ndarray] = None
    att_debiased: Optional[float] = None
    metadata: Dict[str, Any] = PydField(default_factory=dict)
