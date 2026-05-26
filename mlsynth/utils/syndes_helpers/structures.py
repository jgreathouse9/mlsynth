"""Structured containers for the SYNDES synthetic design pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

from ..fast_scm_helpers.structure import IndexSet


@dataclass(frozen=True)
class SYNDESProblemComponents:
    """
    CVXPY components for an SYNDES optimization problem.

    This container stores the symbolic elements required to construct and
    solve the mixed-integer program underlying SYNDES.

    Parameters
    ----------
    mode : str
        SYNDES formulation ("global_2way", "global_equal_weights", "per_unit").
    objective : cp.Expression
        CVXPY objective function.
    constraints : list of cp.Constraint
        Linear and convex constraints defining the feasible set.
    variables : dict of str -> cp.Variable
        Named CVXPY decision variables (e.g., weights, assignment variables).
    assignment_variable : cp.Variable
        Binary treatment assignment variable D.

    Methods
    -------
    with_constraints(additional_constraints)
        Return a new problem with extra constraints appended.

    Notes
    -----
    This object is intentionally “solver-agnostic” and is used to separate
    model construction from optimization execution.
    """

    mode: str
    objective: cp.Expression
    constraints: List[cp.Constraint]
    variables: Dict[str, cp.Variable]
    assignment_variable: cp.Variable

    def with_constraints(
        self, additional_constraints: List[cp.Constraint]
    ) -> "SYNDESProblemComponents":
        """Append additional constraints to the current formulation.

        Parameters
        ----------
        additional_constraints : list of cp.Constraint
            Extra constraints to include in the optimization problem.

        Returns
        -------
        SYNDESProblemComponents
            New instance with extended constraint set.
        """
        return SYNDESProblemComponents(
            mode=self.mode,
            objective=self.objective,
            constraints=[*self.constraints, *additional_constraints],
            variables=self.variables,
            assignment_variable=self.assignment_variable,
        )


@dataclass(frozen=True)
class SYNDESInputs:
    """
    Preprocessed panel data for SYNDES estimation.

    Contains matrix representations of outcomes and index mappings used in
    optimization.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape (T_pre, N) -- rows are time
        periods, columns are units.
    Y_post : np.ndarray or None
        Post-treatment outcome matrix of shape (T_post, N), if available.
    unit_index : IndexSet
        Mapping from unit labels to integer indices.
    time_index : IndexSet
        Mapping from time labels to integer indices.
    pre_time_index : IndexSet
        Index set for pre-treatment periods.
    post_time_index : IndexSet or None
        Index set for post-treatment periods.
    outcome : str
        Name of outcome variable.

    Notes
    -----
    All matrices are aligned such that rows correspond to units and columns
    correspond to time periods.
    """

    Y_pre: np.ndarray
    Y_post: Optional[np.ndarray]
    unit_index: IndexSet
    time_index: IndexSet
    pre_time_index: IndexSet
    post_time_index: Optional[IndexSet]
    outcome: str


@dataclass(frozen=True)
class SYNDESDesign:
    """
    Optimized SYNDES design solution.

    Contains the outcome of solving the mixed-integer program, including
    treatment assignment and synthetic control weights.

    Parameters
    ----------
    mode : str
        Optimization mode used.
    objective_value : float
        Optimal value of the objective function.
    lambda_value : float
        Regularization parameter used in optimization.
    assignment : np.ndarray
        Binary treatment assignment vector D.
    selected_unit_indices : np.ndarray
        Integer indices of treated units.
    selected_unit_labels : np.ndarray
        Original labels of treated units.
    assignment_by_unit : dict
        Mapping from unit label to treatment indicator.

    w : np.ndarray or None, optional
        Synthetic control weights (global or per-unit depending on mode).
    q : np.ndarray or None, optional
        Auxiliary optimization variables (mode-dependent).
    z : np.ndarray or None, optional
        Additional binary or continuous decision variables.
    raw_results : dict, optional
        Raw solver output.
    treated_weights : np.ndarray or None, optional
        Normalized weights over treated units.
    control_weights : np.ndarray or None, optional
        Normalized weights over control units.
    contrast_weights : np.ndarray or None, optional
        Difference weights used in global estimators.
    contrast_series : np.ndarray or None, optional
        Pre-period fitted contrast ``Y_pre @ contrast_weights``, shape
        ``(T_pre,)`` -- the per-period treated-minus-synthetic gap the design
        balances. The design's "prediction" of the (zero) pre-period effect.
    pre_fit_rmse : float or None, optional
        Root mean squared pre-period contrast, ``sqrt(mean(contrast_series^2))``
        -- how tightly the design balances treated and control before launch.

    Notes
    -----
    This object represents the *solution to the design stage only*.
    Post-treatment estimation and inference are handled separately.
    """

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
    contrast_series: Optional[np.ndarray] = None
    pre_fit_rmse: Optional[float] = None


@dataclass(frozen=True)
class SYNDESInference:
    """
    Permutation-based inference results for SYNDES.

    Parameters
    ----------
    atet : float
        Estimated average treatment effect on treated units.
    p_value : float
        Permutation-based p-value.
    reject : bool
        Whether the null hypothesis is rejected at level alpha.
    alpha : float
        Significance level used for the test.
    method : str
        Name of inference procedure.
    null_stats : np.ndarray or None
        Empirical null distribution of test statistics.

    Notes
    -----
    Inference is currently implemented for the global_2way mode only.
    """

    atet: float
    p_value: float
    reject: bool
    alpha: float
    method: str
    null_stats: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SYNDESResults:
    """
    Complete SYNDES estimation output.

    This object bundles the optimized design, prepared inputs, and optional
    inference results.

    Parameters
    ----------
    design : SYNDESDesign
        Optimization solution.
    inputs : SYNDESInputs
        Preprocessed data used in estimation.
    inference : SYNDESInference or None
        Optional inference results.

    Notes
    -----
    This is the primary return object of SYNDES.fit().

    Properties
    ----------
    mode : str
        Alias for design.mode.
    selected_unit_labels : np.ndarray
        Labels of treated units selected by the design.
    """

    design: SYNDESDesign
    inputs: SYNDESInputs
    inference: Optional[SYNDESInference] = None

    @property
    def mode(self) -> str:
        """Return the optimization mode used."""
        return self.design.mode

    @property
    def selected_unit_labels(self) -> np.ndarray:
        """Return labels of selected treated units."""
        return self.design.selected_unit_labels


@dataclass(frozen=True)
class SYNDESMultiArmResults:
    """Per-arm SYNDES designs.

    Returned by :class:`mlsynth.estimators.SYNDES` when an ``arm`` column is
    configured: the SYNDES design problem is solved **independently within
    each arm's units**, and each arm's full result (a :class:`SYNDESResults`
    for the MIP modes, or a ``RelaxedSolverResults`` for the annealed mode)
    is collected here.

    Parameters
    ----------
    arm_designs : dict
        ``{arm_label: SYNDESResults}`` -- one independent SYNDES solution per
        arm (or ``RelaxedSolverResults`` under ``mode="two_way_global_annealed"``).
    arm : str
        Name of the arm column the units were partitioned on.
    """

    arm_designs: Dict[Any, Any]
    arm: str

    @property
    def mode(self) -> str:
        return "syndes_multiarm"

    def atet_by_arm(self) -> Dict[Any, Optional[float]]:
        """``{arm_label: ATET}`` across arms (``None`` where no inference)."""
        out: Dict[Any, Optional[float]] = {}
        for label, result in self.arm_designs.items():
            inference = getattr(result, "inference", None)
            out[label] = inference.atet if inference is not None else None
        return out

    def selected_unit_labels_by_arm(self) -> Dict[Any, Any]:
        """``{arm_label: selected_unit_labels}`` across arms."""
        return {
            label: getattr(result, "selected_unit_labels", None)
            for label, result in self.arm_designs.items()
        }
