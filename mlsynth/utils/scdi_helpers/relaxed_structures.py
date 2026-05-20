"""Structured containers for the two-way relaxed SCDI annealing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class RelaxedSwapLog:
    """Per-iteration Metropolis swap diagnostics for the annealed D-step.

    Parameters
    ----------
    n_proposals : int
        Total number of swap proposals generated during the iteration.
    n_accepted : int
        Number of proposals accepted by the Metropolis criterion.
    n_uphill : int
        Number of proposals that increased the objective.
    n_uphill_accepted : int
        Number of uphill proposals accepted (i.e., explored).
    delta_history : list of float
        Sequence of proposal energy deltas, used downstream by the adaptive
        temperature schedule.

    Notes
    -----
    The container is intentionally minimal and immutable so that successive
    iterations can be accumulated into a :class:`RelaxedAnnealingTrace`.
    """

    n_proposals: int
    n_accepted: int
    n_uphill: int
    n_uphill_accepted: int
    delta_history: List[float]

    @property
    def uphill_acceptance_rate(self) -> float:
        """Fraction of uphill proposals that were accepted."""
        return self.n_uphill_accepted / (self.n_uphill + 1e-8)


@dataclass(frozen=True)
class RelaxedAnnealingTrace:
    """Iteration-level diagnostics from :func:`solve_two_way_relaxed`.

    Parameters
    ----------
    objective_history : list of float
        Current-state energy at each outer iteration.
    rmse_history : list of float
        Current-state RMSE of the synthetic gap at each outer iteration.
    swap_logs : list of RelaxedSwapLog
        Per-iteration Metropolis swap logs.

    Notes
    -----
    This trace describes the *current* state of the chain (as committed),
    not necessarily the best state. The best state is stored separately in
    :class:`RelaxedDesign`.
    """

    objective_history: List[float] = field(default_factory=list)
    rmse_history: List[float] = field(default_factory=list)
    swap_logs: List[RelaxedSwapLog] = field(default_factory=list)


@dataclass(frozen=True)
class RelaxedDesign:
    """Best-state design solution returned by the relaxed annealing solver.

    Parameters
    ----------
    assignment : np.ndarray
        Binary treatment assignment vector ``D`` of shape ``(N,)``.
    raw_weights : np.ndarray
        Combined treated/control weights ``w`` of shape ``(N,)`` as returned
        by the convex QP w-step.
    treated_weights : np.ndarray
        Normalized weights over treated units (sums to one over ``D == 1``).
    control_weights : np.ndarray
        Normalized weights over control units (sums to one over ``D == 0``).
    contrast_weights : np.ndarray
        Signed contrast weights ``(2 D - 1) * w`` used to form the synthetic
        gap.
    synthetic_treated : np.ndarray
        Synthetic treated trajectory ``Y[:, treated] @ w[treated]`` of shape
        ``(T,)``.
    synthetic_control : np.ndarray
        Synthetic control trajectory ``Y[:, control] @ w[control]`` of shape
        ``(T,)``.
    synthetic_gap : np.ndarray
        Difference ``synthetic_treated - synthetic_control``.
    objective_value : float
        Final value of the relaxed energy at the best state.
    rmse : float
        RMSE of the synthetic gap at the best state.
    lambda_value : float
        Regularization parameter used by the solver.
    """

    assignment: np.ndarray
    raw_weights: np.ndarray
    treated_weights: np.ndarray
    control_weights: np.ndarray
    contrast_weights: np.ndarray
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray
    synthetic_gap: np.ndarray
    objective_value: float
    rmse: float
    lambda_value: float


@dataclass(frozen=True)
class RelaxedSolverResults:
    """Complete output of :func:`solve_two_way_relaxed`.

    Parameters
    ----------
    design : RelaxedDesign
        Best-state design solution found by the chain.
    trace : RelaxedAnnealingTrace
        Iteration-level diagnostics of the annealing run.

    Notes
    -----
    The split between ``design`` and ``trace`` mirrors the existing
    :class:`SCDIDesign` / :class:`SCDIResults` separation used by the
    mixed-integer SCDI pipeline.
    """

    design: RelaxedDesign
    trace: RelaxedAnnealingTrace

    @property
    def assignment(self) -> np.ndarray:
        """Alias for ``design.assignment``."""
        return self.design.assignment

    @property
    def objective_value(self) -> float:
        """Alias for ``design.objective_value``."""
        return self.design.objective_value

    @property
    def rmse(self) -> float:
        """Alias for ``design.rmse``."""
        return self.design.rmse
