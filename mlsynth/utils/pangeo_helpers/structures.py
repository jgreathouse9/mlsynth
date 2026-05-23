"""Frozen dataclasses for the PANGEO design estimator.

PANGEO is a *prospective experimental design* method: from historical
(pre-treatment) sales it partitions each treatment arm's geos into
supergeo pairs whose treatment/control halves are maximally parallel over
the pre-period, so a later difference-in-differences / synthetic-control
analysis has clean parallel trends. The output is a **design**
(supergeo pairs + treatment/control assignment + achieved parallelism),
not a treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SupergeoPair:
    """One supergeo pair within an arm.

    Attributes
    ----------
    treatment : list
        Unit names assigned to treatment in this pair.
    control : list
        Unit names assigned to control in this pair.
    gap_variance : float
        Pre-period level-removed gap variance between the two halves
        (lower = more parallel; the DiD pre-period residual SS).
    parallelism_r2 : float
        R^2 of the within-pair parallel-trends fit (1 = perfectly parallel).
    treatment_mean : np.ndarray
        Pre-period mean trajectory of the treatment half.
    control_mean : np.ndarray
        Pre-period mean trajectory of the control half.
    covariate_smd : dict
        ``{covariate_name: standardized mean difference}`` between the
        treatment and control halves (empty if no covariates were used).
    """

    treatment: List[Any]
    control: List[Any]
    gap_variance: float
    parallelism_r2: float
    treatment_mean: np.ndarray
    control_mean: np.ndarray
    covariate_smd: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmDesign:
    """The supergeo-pair design for a single treatment arm.

    Attributes
    ----------
    arm : Any
        Arm label.
    pairs : list of SupergeoPair
        The chosen supergeo pairs partitioning the arm's units.
    n_units : int
        Number of geos eligible for the arm.
    total_gap_variance : float
        Sum of the pairs' gap variances -- the design objective value.
    mean_parallelism_r2 : float
        Mean within-pair parallel-trends R^2 across pairs.
    treatment_units : list
        All units assigned to treatment in the arm.
    control_units : list
        All units assigned to control in the arm.
    """

    arm: Any
    pairs: List[SupergeoPair]
    n_units: int
    total_gap_variance: float
    mean_parallelism_r2: float
    treatment_units: List[Any]
    control_units: List[Any]


@dataclass(frozen=True)
class PangeoResults:
    """Top-level container returned by :meth:`mlsynth.PANGEO.fit`.

    Attributes
    ----------
    arm_designs : dict
        ``{arm_label: ArmDesign}`` -- the design for each arm.
    max_supergeo_size : int
        The Q used (max size of either supergeo within a pair).
    assignment : dict
        Flat ``{unit_name: "treatment"|"control"}`` map across all arms.
    time_labels : np.ndarray
        Pre-period time labels the design was built on.
    metadata : dict
        Free-form diagnostics.
    """

    arm_designs: Dict[Any, ArmDesign]
    max_supergeo_size: int
    assignment: Dict[Any, str]
    time_labels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    power: Optional[Any] = None  # PangeoPower (see pangeo_helpers.power)
