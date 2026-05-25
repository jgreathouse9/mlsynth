"""Driver for the Malo et al. (2024) bilevel SCM algorithm.

Composes the three stages into a single solve, short-circuiting as soon as an
optimal solution is certified (as the paper recommends -- the optimum is
typically a corner found in the early stages).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .simplex import mspe
from .stages import (
    _lower_level_weights,
    corner_solutions,
    tykhonov_refine,
    unconstrained_feasibility,
)
from .structure import BilevelProblem, BilevelSolution


def _weighted_predictor_loss(prob: BilevelProblem, V: np.ndarray, W: np.ndarray) -> float:
    """Lower-level loss ``L_W = sum_k V_k (X1_k - X0_k W)^2``."""
    resid = prob.X1 - prob.X0 @ W
    return float(np.sum(V * resid ** 2))


def solve_bilevel(
    prob: BilevelProblem,
    *,
    feas_tol: float = 1e-8,
    eps_corner: float = 1e-6,
    refine: bool = True,
    refine_gap_tol: float = 1e-3,
) -> BilevelSolution:
    """Solve the bilevel SCM problem (Eq. 6-7).

    Parameters
    ----------
    prob : BilevelProblem
        Outcome and predictor matrices.
    feas_tol : float
        Tolerance for the Section 3.1 feasibility certificate.
    eps_corner : float
        Non-Archimedean tie-breaker for the corner lower-level problems.
    refine : bool
        Whether to run the Tykhonov descent when a gap remains.
    refine_gap_tol : float
        Skip refinement if the corner gap above the lower bound is below this
        fraction of the lower bound (the paper notes corners are usually
        optimal, so refinement rarely fires).
    """
    # Stage 1: unconstrained feasibility (Section 3.1).
    W_unc, V_unc, lower_bound, is_optimal = unconstrained_feasibility(prob, feas_tol=feas_tol)
    if is_optimal:
        return BilevelSolution(
            V=V_unc, W=W_unc,
            upper_loss=mspe(prob.y1_pre, prob.Y0_pre, W_unc),
            lower_loss=_weighted_predictor_loss(prob, V_unc, W_unc),
            lower_bound=lower_bound, stage="unconstrained", iterations=0,
            metadata={"certified": True},
        )

    # Stage 2: corner solutions (Section 3.2).
    V_c, W_c, upper_c, corner_losses = corner_solutions(prob, eps=eps_corner)
    V_best, W_best, upper_best, stage, iters = V_c, W_c, upper_c, "corner", 0

    # Stage 3: Tykhonov descent (Section 3.3), only if a meaningful gap remains.
    rel_gap = (upper_best - lower_bound) / max(abs(lower_bound), 1.0)
    if refine and rel_gap > refine_gap_tol:
        V_r, W_r, upper_r, iters = tykhonov_refine(prob, V_c)
        if upper_r < upper_best:
            V_best, W_best, upper_best, stage = V_r, W_r, upper_r, "tykhonov"

    return BilevelSolution(
        V=V_best, W=W_best,
        upper_loss=float(upper_best),
        lower_loss=_weighted_predictor_loss(prob, V_best, W_best),
        lower_bound=lower_bound, stage=stage, iterations=iters,
        metadata={
            "corner_losses": [float(x) for x in corner_losses],
            "unconstrained_upper": float(mspe(prob.y1_pre, prob.Y0_pre, W_unc)),
            "gap": float(upper_best - lower_bound),
        },
    )


def lower_level_weights(prob: BilevelProblem, V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Public wrapper: donor weights for fixed predictor weights ``V``."""
    return _lower_level_weights(prob, np.asarray(V, dtype=float), eps)
