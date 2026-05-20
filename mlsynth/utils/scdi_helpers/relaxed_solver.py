"""Orchestration for the two-way relaxed SCDI annealing solver.

The relaxed solver performs alternating optimization between a discrete
treatment-assignment search (simulated annealing over swaps) and a convex
weight update (QP w-step). All heavy numerical logic lives in the
``relaxed_formulation``, ``relaxed_annealing``, and
``relaxed_initialization`` modules; this file only coordinates the loop
and assembles a typed result object.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .relaxed_annealing import d_step_annealed, temperature_schedule
from .relaxed_formulation import (
    compute_energy,
    compute_rmse_gap,
    extract_weights,
    solve_weights_global,
    synthetic_paths,
)
from .relaxed_initialization import (
    default_lambda,
    init_assignment,
    validate_relaxed_inputs,
)
from .relaxed_structures import (
    RelaxedAnnealingTrace,
    RelaxedDesign,
    RelaxedSolverResults,
)


def solve_two_way_relaxed(
    Y: np.ndarray,
    K: int,
    lam: Optional[float] = None,
    max_iter: int = 40,
    decay: float = 0.97,
    verbose: bool = True,
) -> RelaxedSolverResults:
    """Two-way relaxed synthetic control with annealed D-step optimization.

    Alternates between a Metropolis D-step that proposes assignment swaps
    and a convex w-step QP that re-solves treated and control weights at
    each candidate assignment. The best state visited by the chain is
    returned as a :class:`RelaxedDesign`.

    Parameters
    ----------
    Y : np.ndarray
        Outcome matrix of shape ``(T, N)``.
    K : int
        Number of treated units to select.
    lam : float, optional
        Ridge penalty for the energy. Defaults to the mean cross-sectional
        variance of ``Y`` (see :func:`default_lambda`).
    max_iter : int, optional
        Number of outer annealing iterations.
    decay : float, optional
        Geometric decay factor for the warm-up temperature schedule.
    verbose : bool, optional
        If ``True``, prints a one-line diagnostic per outer iteration.

    Returns
    -------
    RelaxedSolverResults
        ``design`` holds the best-state solution; ``trace`` holds the
        per-iteration diagnostics.
    """

    validate_relaxed_inputs(Y, K)

    lam_value = default_lambda(Y) if lam is None else float(lam)

    D = init_assignment(Y, K)
    w = solve_weights_global(Y, D)

    best_D = D.copy()
    best_w = w.copy()
    best_E = compute_energy(Y, D, w, lam_value)
    best_rmse = compute_rmse_gap(Y, D, w)
    best_mu_T, best_mu_C, best_gap = synthetic_paths(Y, D, w)

    trace = RelaxedAnnealingTrace(
        objective_history=[],
        rmse_history=[],
        swap_logs=[],
    )
    global_delta_history: list[float] = []

    for it in range(max_iter):
        T = temperature_schedule(
            it,
            Y,
            delta_history=global_delta_history,
            decay=decay,
        )

        D_new, w_new, log = d_step_annealed(Y, D, w, K, T, lam_value)

        current_E = compute_energy(Y, D_new, w_new, lam_value)
        current_rmse = compute_rmse_gap(Y, D_new, w_new)

        D, w = D_new, w_new

        if current_E < best_E:
            best_D = D.copy()
            best_w = w.copy()
            best_E = current_E
            best_rmse = current_rmse
            best_mu_T, best_mu_C, best_gap = synthetic_paths(Y, best_D, best_w)

        trace.objective_history.append(current_E)
        trace.rmse_history.append(current_rmse)
        trace.swap_logs.append(log)
        global_delta_history.extend(log.delta_history)

        if verbose:
            print(
                f"[{it:02d}] "
                f"E={current_E:.6f} "
                f"RMSE={current_rmse:.6f} "
                f"T={T:.4f} "
                f"uphill_accept={log.uphill_acceptance_rate:.2f}"
            )

    final_weights = extract_weights(best_D, best_w)

    design = RelaxedDesign(
        assignment=best_D,
        raw_weights=best_w,
        treated_weights=final_weights["treated_weights"],
        control_weights=final_weights["control_weights"],
        contrast_weights=final_weights["contrast_weights"],
        synthetic_treated=best_mu_T,
        synthetic_control=best_mu_C,
        synthetic_gap=best_gap,
        objective_value=best_E,
        rmse=best_rmse,
        lambda_value=lam_value,
    )

    return RelaxedSolverResults(design=design, trace=trace)
