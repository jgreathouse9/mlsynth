"""Frozen containers for the bilevel SCM optimization (Malo et al. 2024).

The synthetic-control problem is an *optimistic bilevel* program: the
upper level chooses predictor weights ``V`` (and, implicitly, donor weights
``W``) to minimize the pre-treatment outcome fit, while the lower level
chooses ``W`` to minimize the ``V``-weighted predictor discrepancy on the
donor simplex.

References
----------
Malo, P., Eskelinen, J., Zhou, X., & Kuosmanen, T. (2024). Computing
Synthetic Controls Using Bilevel Optimization. Computational Economics,
64, 1113-1136. https://doi.org/10.1007/s10614-023-10471-7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class BilevelProblem:
    """Inputs to the bilevel SCM solver (pure NumPy).

    Parameters
    ----------
    y1_pre : np.ndarray
        Treated pre-treatment outcomes, shape ``(Tpre,)`` -- the upper-level
        target ``Y_1^{pre}``.
    Y0_pre : np.ndarray
        Donor pre-treatment outcomes, shape ``(Tpre, J)`` -- ``Y_0^{pre}``.
    X1 : np.ndarray
        Treated predictor vector, shape ``(K,)`` -- ``X_1``.
    X0 : np.ndarray
        Donor predictor matrix, shape ``(K, J)`` -- ``X_0``.
    predictor_names : list
        Labels of the ``K`` predictors, for provenance.
    """

    y1_pre: np.ndarray
    Y0_pre: np.ndarray
    X1: np.ndarray
    X0: np.ndarray
    predictor_names: List[Any] = field(default_factory=list)

    @property
    def n_donors(self) -> int:
        return int(self.Y0_pre.shape[1])

    @property
    def n_predictors(self) -> int:
        return int(self.X0.shape[0])

    @property
    def Tpre(self) -> int:
        return int(self.y1_pre.shape[0])


@dataclass(frozen=True)
class BilevelSolution:
    """Output of the bilevel SCM solver.

    Attributes
    ----------
    V : np.ndarray
        Optimal predictor weights, shape ``(K,)`` on the simplex.
    W : np.ndarray
        Optimal donor weights, shape ``(J,)`` on the simplex.
    upper_loss : float
        Upper-level loss ``L_V = ||Y1_pre - Y0_pre W||^2`` (outcome fit).
    lower_loss : float
        Lower-level loss ``L_W = ||X1 - X0 W||^2_V`` (V-weighted predictor fit).
    lower_bound : float
        ``L(W**)`` from the unconstrained simplex regression -- the global
        lower bound on ``upper_loss``.
    stage : str
        Which stage produced the solution: ``"unconstrained"``, ``"corner"``,
        or ``"tykhonov"``.
    iterations : int
        Number of refinement iterations performed (0 if a corner/unconstrained
        solution was accepted).
    metadata : dict
        Free-form diagnostics (corner losses, gap, etc.).
    """

    V: np.ndarray
    W: np.ndarray
    upper_loss: float
    lower_loss: float
    lower_bound: float
    stage: str
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """Upper-level optimality gap ``L_V - L(W**)`` (>= 0)."""
        return float(self.upper_loss - self.lower_bound)
