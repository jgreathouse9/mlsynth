"""Frozen dataclasses for the Spatial Synthetic Difference-in-Differences estimator.

Serenini & Masek (2024). *"Spatial Synthetic Difference-in-Differences."*
SSRN 4736857. Extends Arkhangelsky-Athey-Hirshberg-Imbens-Wager (2021) SDID
with a spatial spillover term :math:`\\tau_s` so the estimator can disentangle
the direct ATT on the directly-treated units from the *indirect* (spillover)
effect on units exposed via a spatial weight matrix :math:`W`.

Two estimands fall out of one regression:

* :math:`\\widehat \\tau` -- direct effect on the directly-treated units
  (the ATT, identical in form to standard SDID).
* :math:`\\widehat \\tau_s` -- spillover effect per unit of neighbour-treatment
  exposure :math:`(WD)_{it} = \\sum_j w_{ij} D_{jt}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import ConfigDict, Field as PydField

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class SpSyDiDInputs:
    """Preprocessed panel + spatial weights for SpSyDiD.

    Attributes
    ----------
    outcome_matrix : np.ndarray
        ``(N, T)`` panel of outcomes, ordered by ``unit_names``.
    treatment_matrix : np.ndarray
        ``(N, T)`` panel of 0/1 treatment indicators.
    spatial_matrix : np.ndarray
        Row-standardised spatial weight matrix, shape ``(N, N)``,
        ordered consistently with ``unit_names``.
    exposure_matrix : np.ndarray
        Pre-computed spillover exposure :math:`(WD)_{it}
        = \\sum_j w_{ij} D_{jt}`, shape ``(N, T)``.
    unit_names : list
        Length-``N`` ordering of unit ids.
    time_labels : np.ndarray
        Length-``T`` ordering of time-period labels.
    T : int
        Total number of panel periods.
    T0 : int
        Number of pre-treatment periods (largest ``t`` such that
        no unit is treated for ``t' <= t``).
    direct_indices : np.ndarray
        Indices of *directly* treated units (those with ``D=1`` at some t).
    spillover_indices : np.ndarray
        Indices of *indirectly* treated units (``D=0`` always but
        ``(WD)_it > 0`` at some t).
    pure_control_indices : np.ndarray
        Indices of pure controls (``D=0`` and ``(WD)=0`` for all t).
    """

    outcome_matrix: np.ndarray
    treatment_matrix: np.ndarray
    spatial_matrix: np.ndarray
    exposure_matrix: np.ndarray
    unit_names: List[Any]
    time_labels: np.ndarray
    T: int
    T0: int
    direct_indices: np.ndarray
    spillover_indices: np.ndarray
    pure_control_indices: np.ndarray

    @property
    def N(self) -> int:
        return len(self.unit_names)

    @property
    def N_direct(self) -> int:
        return int(self.direct_indices.size)

    @property
    def N_spillover(self) -> int:
        return int(self.spillover_indices.size)

    @property
    def N_pure(self) -> int:
        return int(self.pure_control_indices.size)


class SpSyDiDResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.SpSyDiD.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (the observational report).
    SpSyDiD is a **spillover decomposition**: the **direct** effect
    :math:`\\widehat{\\tau}` (ATT) drives the standardized surface, so the flat
    accessors ``att`` / ``counterfactual`` / ``gap`` / ``pre_rmse`` describe the
    *directly-treated* group (observed mean vs the pure-control SDID synthetic).
    The **indirect** (``aite``) and **total** (``ate``) effects, which have no
    single counterfactual path, are kept as typed fields below. The pure-control
    SDID unit weights live in the standardized ``weights`` slot (with the time
    weights in ``summary_stats``).

    Parameters
    ----------
    inputs : SpSyDiDInputs
        Preprocessed panel + W matrix + auto-detected partition.
    aite : float
        Average indirect treatment effect per unit of exposure
        :math:`\\widehat{\\tau}_s`. Multiply by the average exposure to
        recover the population-level spillover.
    ate : float
        Implied population-level ATE
        :math:`\\widehat{\\tau} \\cdot (1 + \\bar{WD})` per the paper's
        eq. 14, with :math:`\\bar{WD}` the average exposure across the
        directly + indirectly treated units.
    unit_weights : dict
        Mapping ``{unit_name: omega}`` -- the per-unit weights used in
        the final WLS regression (SDID-style for pure controls,
        uniform :math:`1/N_{tr}` for directly treated and
        :math:`1/N_{sp}` for indirectly treated).
    time_weights : np.ndarray
        Length-``T0`` SDID time weights for the pre-period
        (post-period weights are uniform :math:`1/T_{\\text{post}}` and
        not stored).
    zeta : float
        SDID regularisation parameter from Arkhangelsky et al. 2021
        (used in the unit-weight QP for pure controls).
    metadata : dict
        Free-form diagnostics (mean exposure, partition sizes, etc.).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: SpSyDiDInputs
    aite: float
    ate: float
    unit_weights: Dict[Any, float]
    time_weights: np.ndarray
    zeta: float
    metadata: Dict[str, Any] = PydField(default_factory=dict)

    @property
    def tau(self) -> float:
        """Alias matching paper notation: direct effect."""
        return self.att

    @property
    def tau_s(self) -> float:
        """Alias matching paper notation: spillover coefficient."""
        return self.aite
