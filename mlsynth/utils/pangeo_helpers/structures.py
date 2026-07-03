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
from pydantic import Field

from ...config_models import DesignResult


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
    gap_level : float
        DiD counterfactual gap level :math:`\\delta` -- the mean gap over the
        *estimation* window E (the periods the split was optimised on).
    holdout_resid : np.ndarray
        Gap residuals on the held-out blank window B (``gap[B] - gap_level``).
        B is excluded from the optimisation, so these residuals are an honest
        out-of-sample estimate of the parallel-trends noise -- the reservoir
        for conformal inference and the variance behind the MDE.
    """

    treatment: List[Any]
    control: List[Any]
    gap_variance: float
    parallelism_r2: float
    treatment_mean: np.ndarray
    control_mean: np.ndarray
    covariate_smd: Dict[str, float] = field(default_factory=dict)
    gap_level: float = 0.0
    holdout_resid: np.ndarray = field(default_factory=lambda: np.empty(0))


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


class PangeoResults(DesignResult):
    """Top-level container returned by :meth:`mlsynth.PANGEO.fit`.

    A :class:`~mlsynth.config_models.DesignResult` (the experimental-design
    family): it chooses the treatment/control assignment *before* any
    intervention, and -- once post-period outcomes exist -- resolves to an
    :class:`~mlsynth.config_models.EffectResult` via :attr:`report`.

    Inherited design fields
    -----------------------
    report : EffectResult or None
        The realized effect report (program-level Augmented-DiD ATT mapped to
        the standard effect surface); ``None`` for a design-only fit.
    assignment : dict
        Flat ``{unit_name: "treatment"|"control"}`` map across all arms.
    selected_units : list
        Units assigned to treatment (the design's chosen treated set).
    power : PangeoPower or None
        Program- and arm-level MDE / power analysis.
    metadata : dict
        Free-form design diagnostics (solver, q_sweep, solver_diagnostics, ...).

    PANGEO-specific fields
    ----------------------
    arm_designs : dict
        ``{arm_label: ArmDesign}`` -- the supergeo-pair design per arm.
    max_supergeo_size : int
        The Q used (max size of either supergeo within a pair).
    time_labels : np.ndarray
        Pre-period time labels the design was built on.
    effects : PangeoEffects or None
        The rich realized-ATT object (per-pair, per-arm, program, plus the
        design-based randomization inference) when post-period data is given.
    """

    arm_designs: Dict[Any, ArmDesign] = Field(default_factory=dict)
    max_supergeo_size: Optional[int] = None
    time_labels: Optional[np.ndarray] = None
    effects: Optional[Any] = None  # PangeoEffects when post-period data given

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        frozen = True  # preserve the immutable-result contract (use model_copy)
