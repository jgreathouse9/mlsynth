#relaxed_global_solver

"""
Solver-facing optimization utilities for two-way relaxed synthetic control.

This module mirrors the SCDI solver architecture but targets:
    - alternating optimization (w-step + D-step)
    - projected gradient relaxation for D
    - convex QP weight updates

It is designed to plug directly into the helpers module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError

from .pgd_formulation import synthetic_paths, rmse_synthetic_gap, d_step_annealed, propose_swap, _extract_weights, init_D, solve_weights_global, energy, estimate_lambda, temperature_schedule

# ============================================================
# MAIN ANNEALING SOLVER
# ============================================================

def solve_two_way_relaxed(
        Y: np.ndarray,
        K: int,
        lam: float = None,
        max_iter: int = 40,
        decay: float = 0.97,
        verbose: bool = True,
):
    """
    Two-way relaxed synthetic control with annealed D-step optimization.
    """

    lam = float(np.mean(np.var(Y, axis=0))) if lam is None else float(lam)

    # ------------------------------------------------------------
    # INIT STATE
    # ------------------------------------------------------------
    D = init_D(Y, K)
    w = solve_weights_global(Y, D)

    # ------------------------------------------------------------
    # BEST STATE TRACKING (CRITICAL)
    # ------------------------------------------------------------
    best_D = D.copy()
    best_w = w.copy()

    best_E = energy(Y, D, w, lam)
    best_rmse = rmse_synthetic_gap(Y, D, w)

    best_mu_T, best_mu_C, best_gap = synthetic_paths(Y, D, w)

    # ------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------
    history = []
    rmse_history = []
    annealing_trace = []
    global_delta_history = []

    # ------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------
    for it in range(max_iter):

        # temperature
        T = temperature_schedule(
            it,
            Y,
            delta_history=global_delta_history,
            decay=decay,
        )

        # propose new state
        D_new, w_new, log = d_step_annealed(
            Y, D, w, K, T, lam
        )

        # evaluate CURRENT state (IMPORTANT: must use SAME (D_new, w_new))
        current_E = energy(Y, D_new, w_new, lam)
        current_rmse = rmse_synthetic_gap(Y, D_new, w_new)

        # commit move
        D, w = D_new, w_new

        # update BEST state consistently
        if current_E < best_E:
            best_D = D.copy()
            best_w = w.copy()
            best_E = current_E
            best_rmse = current_rmse

            best_mu_T, best_mu_C, best_gap = synthetic_paths(Y, best_D, best_w)

        # logging
        history.append(current_E)
        rmse_history.append(current_rmse)
        annealing_trace.append(log)
        global_delta_history.extend(log["delta_history"])

        # debug
        if verbose:
            uphill_rate = log["n_uphill_accepted"] / (log["n_uphill"] + 1e-8)

            print(
                f"[{it:02d}] "
                f"E={current_E:.6f} "
                f"RMSE={current_rmse:.6f} "
                f"T={T:.4f} "
                f"uphill_accept={uphill_rate:.2f}"
            )

    # ------------------------------------------------------------
    # FINAL OUTPUT (CONSISTENT BEST STATE ONLY)
    # ------------------------------------------------------------
    final_weights = _extract_weights(best_D, best_w)

    return {
        "assignment": best_D,
        "raw_weights": best_w,
        "weights": final_weights,

        # predictions (correctly synced with best state)
        "synthetic_treated": best_mu_T,
        "synthetic_control": best_mu_C,
        "synthetic_gap": best_gap,

        # metrics
        "best_objective": best_E,
        "best_rmse": best_rmse,
        "rmse_history": rmse_history,

        # diagnostics
        "history": history,
        "annealing_trace": annealing_trace,
        "lambda": lam,
    }
