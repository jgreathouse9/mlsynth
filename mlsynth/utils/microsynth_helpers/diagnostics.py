"""Balance diagnostics for MicroSynth.

Functions to assess whether the dual solver's weights actually
achieved covariate balance, how concentrated the weights are, and
how many effective control units remain after weighting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def standardized_mean_difference(
    X_T: np.ndarray,
    X_C: np.ndarray,
    w: np.ndarray | None = None,
) -> np.ndarray:
    """Per-covariate SMD between treated and (optionally weighted) controls.

    SMD is defined as ``(mean_T - mean_C) / pooled_sd`` where
    ``pooled_sd = sqrt((var_T + var_C) / 2)``. By convention
    ``|SMD| < 0.1`` is considered balanced.
    """
    mu_T = X_T.mean(axis=0)
    if w is None:
        mu_C = X_C.mean(axis=0)
    else:
        mu_C = w @ X_C
    pooled_sd = np.sqrt(0.5 * (X_T.var(axis=0) + X_C.var(axis=0)))
    pooled_sd = np.where(pooled_sd == 0, 1.0, pooled_sd)
    return (mu_T - mu_C) / pooled_sd


def effective_sample_size(w: np.ndarray) -> float:
    """Effective sample size, ``1 / sum(w^2)``.

    Equal weights give ``ESS = n_C``. A degenerate single-user
    solution gives ``ESS = 1``. Lower ESS means the weighted
    estimator depends on fewer effective observations.
    """
    s = float((w ** 2).sum())
    return float("inf") if s == 0 else 1.0 / s


def max_weight(w: np.ndarray) -> float:
    return float(np.max(w))


def feasibility_check(
    smd_after: np.ndarray,
    balance_tol: float,
) -> Tuple[bool, str]:
    """Did every balancing constraint achieve ``|SMD| < balance_tol``?

    If not, the treated group's covariate mean lies outside the
    convex hull of the controls' covariate matrix, and no choice of
    non-negative weights summing to 1 can satisfy all constraints
    exactly. The QP returns the closest feasible point but the
    estimator is biased.
    """
    max_abs = float(np.max(np.abs(smd_after)))
    if max_abs < balance_tol:
        return True, (
            f"Balance achieved (max |SMD| = {max_abs:.2e} < tol = "
            f"{balance_tol:.2e})."
        )
    return False, (
        f"Balance NOT achieved: max |SMD| = {max_abs:.4f} exceeds tol "
        f"= {balance_tol:.2e}. The treated group's covariate mean may "
        f"lie outside the convex hull of controls. Consider adding "
        f"more control users, dropping outlying treated users, or "
        f"relaxing the covariate set."
    )
