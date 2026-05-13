"""Structured containers for the SCDI synthetic design pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

from ..fast_scm_helpers.structure import IndexSet


@dataclass(frozen=True)
class SCDIProblemComponents:
    """CVXPY components for an SCDI optimization formulation.

    Keeping these pieces together makes the solver thin while allowing new
    helper functions to add constraints or swap objective pieces before the
    :class:`cvxpy.Problem` is created.
    """

    mode: str
    objective: cp.Expression
    constraints: List[cp.Constraint]
    variables: Dict[str, cp.Variable]
    assignment_variable: cp.Variable

    def with_constraints(
        self, additional_constraints: List[cp.Constraint]
    ) -> "SCDIProblemComponents":
        """Return a copy with additional constraints appended."""

        return SCDIProblemComponents(
            mode=self.mode,
            objective=self.objective,
            constraints=[*self.constraints, *additional_constraints],
            variables=self.variables,
            assignment_variable=self.assignment_variable,
        )


@dataclass(frozen=True)
class SCDIInputs:
    """Prepared panel matrices and label metadata for SCDI."""

    Y_pre: np.ndarray
    Y_post: Optional[np.ndarray]
    unit_index: IndexSet
    time_index: IndexSet
    pre_time_index: IndexSet
    post_time_index: Optional[IndexSet]
    outcome: str


@dataclass(frozen=True)
class SCDIDesign:
    """Optimization output for an SCDI design solve."""

    mode: str
    objective_value: float
    lambda_value: float
    assignment: np.ndarray
    selected_unit_indices: np.ndarray
    selected_unit_labels: np.ndarray
    assignment_by_unit: Dict[Any, int]
    w: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    raw_results: Dict[str, Any] = field(default_factory=dict)
    treated_weights: Optional[np.ndarray] = None
    control_weights: Optional[np.ndarray] = None
    contrast_weights: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SCDIInference:
    """Inference output for an SCDI design."""

    atet: float
    p_value: float
    reject: bool
    alpha: float
    method: str
    null_stats: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SCDIResults:
    """Top-level result object returned by :class:`mlsynth.SCDI`."""

    design: SCDIDesign
    inputs: SCDIInputs
    inference: Optional[SCDIInference] = None

    @property
    def mode(self) -> str:
        """Return the design mode used by the optimizer."""
        return self.design.mode

    @property
    def selected_unit_labels(self) -> np.ndarray:
        """Labels for units selected into treatment by the design."""
        return self.design.selected_unit_labels
