"""Frozen dataclass containers for the MAREX (synthetic experimental design)
pipeline.

Implements the containers for:

    Abadie, A., & Zhao, J. (2026). "Synthetic Controls for Experimental Design."

MAREX *designs* an experiment on aggregate units: it chooses treated weights
``w`` and control weights ``v`` (on the simplex, disjoint via ``w_j v_j = 0``)
so the synthetic treated and synthetic control units reproduce population
predictor means. All containers are frozen (immutable) per the repository
convention; inference, when requested, is computed up front and embedded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class MAREXStudy:
    """Design hyperparameters of a MAREX study (was ``StudyConfig``)."""

    design: str
    T0: int
    blank_periods: int
    beta: float = 1e-6
    lambda1: float = 0.0
    lambda2: float = 0.0
    xi: float = 0.0


@dataclass(frozen=True)
class MAREXInference:
    """Placebo/permutation inference for one synthetic treated-vs-control pair.

    Parameters
    ----------
    treated_effects : np.ndarray
        Post-period synthetic treated minus synthetic control, shape ``(T1,)``.
    placebo_effects : np.ndarray
        The same contrast on the blank (held-out pre) periods, shape ``(Tb,)``.
    fulltreated_effects : np.ndarray
        The contrast over the whole timeline, shape ``(T,)``.
    s_obs : float
        Observed test statistic (mean absolute post-period effect).
    global_p_value : float
        Permutation p-value for the global null of no effect.
    per_period_pvals : np.ndarray
        Per-post-period p-values, shape ``(T1,)``.
    ci : np.ndarray
        Split-conformal confidence band over the full timeline, shape ``(T, 2)``
        (pre-period rows are ``NaN``).
    alpha : float
        Two-sided significance level.
    """

    treated_effects: np.ndarray
    placebo_effects: np.ndarray
    fulltreated_effects: np.ndarray
    s_obs: float
    global_p_value: float
    per_period_pvals: np.ndarray
    ci: np.ndarray
    alpha: float = 0.05


@dataclass(frozen=True)
class MAREXClusterDesign:
    """Design and synthetics for a single cluster.

    Parameters
    ----------
    label : str
        Cluster label.
    members : list
        Unit labels in this cluster.
    cardinality : int
        Number of units in the cluster.
    treated_weights : np.ndarray
        Treated weights ``w`` for this cluster's column, shape ``(N,)``.
    control_weights : np.ndarray
        Control weights ``v`` for this cluster's column, shape ``(N,)``.
    selection_indicators : np.ndarray
        Binary selection mask ``z`` over the cluster's members.
    synthetic_treated : np.ndarray
        Synthetic treated outcome over the full timeline, shape ``(T,)``.
    synthetic_control : np.ndarray
        Synthetic control outcome over the full timeline, shape ``(T,)``.
    pre_treatment_means : np.ndarray
        Cluster predictor means used as the matching target.
    rmse : float
        Pre-treatment fit RMSE (synthetic treated vs control).
    unit_weight_map : dict
        ``{"Treated": {unit: w}, "Control": {unit: v}}`` for non-zero weights.
    inference : MAREXInference, optional
        Inference for this cluster (``None`` unless requested).
    """

    label: str
    members: List[Any]
    cardinality: int
    treated_weights: np.ndarray
    control_weights: np.ndarray
    selection_indicators: np.ndarray
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    pre_treatment_means: np.ndarray
    rmse: float
    unit_weight_map: Dict[str, Dict[Any, float]]
    inference: Optional[MAREXInference] = None


@dataclass(frozen=True)
class MAREXGlobalDesign:
    """Aggregated (population-level) design and synthetics.

    Parameters
    ----------
    Y_full : np.ndarray
        Observed outcome matrix, shape ``(N, T)``.
    Y_fit : np.ndarray
        Fitting slice, shape ``(N, T_fit)``.
    Y_blank : np.ndarray, optional
        Held-out blank pre-periods, shape ``(N, Tb)`` (``None`` if none).
    treated_weights_agg : np.ndarray
        Cluster-size-weighted aggregate treated weights, shape ``(N,)``.
    control_weights_agg : np.ndarray
        Cluster-size-weighted aggregate control weights, shape ``(N,)``.
    synthetic_treated : np.ndarray
        Aggregate synthetic treated outcome, shape ``(T,)``.
    synthetic_control : np.ndarray
        Aggregate synthetic control outcome, shape ``(T,)``.
    inference : MAREXInference, optional
        Aggregate inference (``None`` unless requested).
    """

    Y_full: np.ndarray
    Y_fit: np.ndarray
    Y_blank: Optional[np.ndarray]
    treated_weights_agg: np.ndarray
    control_weights_agg: np.ndarray
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    inference: Optional[MAREXInference] = None


@dataclass(frozen=True)
class MAREXResults:
    """User-facing output of the MAREX estimator.

    Parameters
    ----------
    clusters : dict of {str: MAREXClusterDesign}
        Per-cluster design (a single ``"0"`` entry when no cluster column).
    study : MAREXStudy
        Design hyperparameters.
    globres : MAREXGlobalDesign
        Aggregate design and synthetics.
    """

    clusters: Dict[str, MAREXClusterDesign]
    study: MAREXStudy
    globres: MAREXGlobalDesign

    @property
    def mode(self) -> str:
        """Solver mode reported to downstream consumers."""
        return "marex"

    @property
    def synthetic_treated(self) -> np.ndarray:
        """Aggregate synthetic treated outcome, shape ``(T,)``."""
        return self.globres.synthetic_treated

    @property
    def synthetic_control(self) -> np.ndarray:
        """Aggregate synthetic control outcome, shape ``(T,)``."""
        return self.globres.synthetic_control

    @property
    def treated_units(self) -> List[Any]:
        """Units assigned to treatment (non-zero aggregate treated weight)."""
        cd = self.clusters
        out: List[Any] = []
        for c in cd.values():
            out.extend(c.unit_weight_map.get("Treated", {}).keys())
        return out
