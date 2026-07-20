"""Frozen containers and the result object for the MEDSC estimator.

MEDSC (Mellace & Pasquini 2022) decomposes the synthetic-control treatment
effect into a direct and an indirect (through-the-mediator) channel:

.. math::

   \\hat\\tau^{tot}_t = Y_{1t} - \\hat Y^{0,M0}_{1t}, \\qquad
   \\hat\\tau^{dir}_t = Y_{1t} - \\hat Y^{0,M1}_{1t}, \\qquad
   \\hat\\tau^{ind}_t = \\hat\\tau^{tot}_t - \\hat\\tau^{dir}_t,

where :math:`\\hat Y^{0,M0}` is the ordinary synthetic control and
:math:`\\hat Y^{0,M1}` is the cross-world control that additionally matches the
treated unit's post-treatment mediator path. The three layers below (inputs,
decomposition, results) keep that pipeline pluggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class MEDSCInputs:
    """Preprocessed panel data for MEDSC estimation.

    Parameters
    ----------
    treated_outcome, treated_mediator : np.ndarray
        Treated-unit outcome and mediator series, shape ``(T,)``.
    total_donor_outcomes : np.ndarray
        Donor outcomes for the total pool, shape ``(T, J_tot)``.
    direct_donor_outcomes, direct_donor_mediators : np.ndarray
        Donor outcomes and mediators for the direct pool, shape ``(T, J_dir)``.
    total_donor_names, direct_donor_names : tuple
        Donor labels for the two pools.
    treated_covariates : np.ndarray or None
        Treated covariate vector, shape ``(L,)`` (or ``None``).
    total_covariates, direct_covariates : np.ndarray or None
        Donor covariate blocks, shape ``(L, J_tot)`` / ``(L, J_dir)``.
    covariate_names : tuple
        Covariate labels, length ``L`` (empty when no covariates).
    time_labels : np.ndarray
        Period labels, length ``T``.
    T, T0 : int
        Total and pre-treatment period counts.
    treated_name : Any
        Label of the treated unit.
    """

    treated_outcome: np.ndarray
    treated_mediator: np.ndarray
    total_donor_outcomes: np.ndarray
    direct_donor_outcomes: np.ndarray
    direct_donor_mediators: np.ndarray
    total_donor_names: Tuple[Any, ...]
    direct_donor_names: Tuple[Any, ...]
    treated_covariates: Optional[np.ndarray]
    total_covariates: Optional[np.ndarray]
    direct_covariates: Optional[np.ndarray]
    covariate_names: Tuple[Any, ...]
    time_labels: np.ndarray
    T: int
    T0: int
    treated_name: Any

    @property
    def n_post(self) -> int:
        """Number of post-treatment periods."""
        return int(self.T - self.T0)

    @property
    def L(self) -> int:
        """Number of covariates."""
        return len(self.covariate_names)


@dataclass(frozen=True)
class MediationDecomposition:
    """The total / direct / indirect decomposition produced by MEDSC.

    Parameters
    ----------
    total, direct, indirect : np.ndarray
        Per-period effect paths, shape ``(T,)`` (NaN in the pre-period for the
        direct/indirect series, which are only defined post-treatment).
    counterfactual_total : np.ndarray
        Ordinary synthetic control :math:`\\hat Y^{0,M0}`, shape ``(T,)``.
    counterfactual_direct : np.ndarray
        Cross-world (mediator-matched) control :math:`\\hat Y^{0,M1}`, shape
        ``(T,)`` (NaN pre-treatment).
    att_total, att_direct, att_indirect : float
        Post-treatment mean effects.
    pre_rmse_total : float
        Pre-treatment RMSE of the total-effect synthetic control.
    total_weights : dict
        Donor weights of the total-effect control.
    direct_weights_final : dict
        Direct-effect donor weights at the final post period.
    """

    total: np.ndarray
    direct: np.ndarray
    indirect: np.ndarray
    counterfactual_total: np.ndarray
    counterfactual_direct: np.ndarray
    att_total: float
    att_direct: float
    att_indirect: float
    pre_rmse_total: float
    total_weights: Dict[Any, float]
    direct_weights_final: Dict[Any, float]


class MEDSCResults(BaseEstimatorResults):
    """Top-level container returned by :meth:`mlsynth.MEDSC.fit`.

    An :class:`~mlsynth.config_models.EffectResult` (observational report): it
    populates the standardized sub-models so the flat accessors (``att`` /
    ``counterfactual`` / ``gap`` / ``pre_rmse``) resolve through the base
    contract. The headline ``att`` is the total effect; the mediation
    decomposition (direct / indirect channels) lives on ``decomposition``.

    Parameters
    ----------
    inputs : MEDSCInputs
        Preprocessed panel.
    decomposition : MediationDecomposition
        Total / direct / indirect effect paths and counterfactuals.
    metadata : dict
        Free-form pipeline diagnostics.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    inputs: MEDSCInputs
    decomposition: MediationDecomposition
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def att_direct(self) -> float:
        """Post-treatment mean direct effect."""
        return self.decomposition.att_direct

    @property
    def att_indirect(self) -> float:
        """Post-treatment mean indirect (through-mediator) effect."""
        return self.decomposition.att_indirect


# Resolve forward references (module uses ``from __future__ import annotations``).
MEDSCResults.model_rebuild()
