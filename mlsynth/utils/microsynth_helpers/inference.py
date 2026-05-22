"""Paired stratified bootstrap inference for MicroSynth.

Resample treated users and control users separately with replacement,
preserving the original ``(n_T, n_C)`` allocation. For each resample,
refit the dual solver and recompute the ATT. The bootstrap
distribution of ATT-estimates yields the CI.

Single-user weight bootstrapping is *not* used here -- it requires
re-standardization that complicates inference. Pair-wise resampling
on the user blocks is the standard ATT bootstrap and matches the
practice in Wang-Zubizarreta (2019) and the original
Robbins-Davenport reference implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .dual_solver import solve_microsynth_dual


def _bootstrap_one_estimate(
    X_T: np.ndarray,
    X_C: np.ndarray,
    Y_T: np.ndarray,
    Y_C: np.ndarray,
    max_iter: int,
    gtol: float,
) -> float:
    """One bootstrap rep: solve the dual on the resampled data."""
    xbar_T = X_T.mean(axis=0)
    res = solve_microsynth_dual(X_C, xbar_T, max_iter=max_iter, gtol=gtol)
    if not res.converged:
        return float("nan")
    if Y_T.ndim == 1:
        return float(Y_T.mean() - res.w @ Y_C)
    return float(Y_T.mean(axis=0).mean() - (res.w @ Y_C).mean())


def paired_bootstrap_ci(
    X_T: np.ndarray,
    X_C: np.ndarray,
    Y_T: np.ndarray,
    Y_C: np.ndarray,
    n_bootstrap: int,
    seed: int,
    max_iter: int = 500,
    gtol: float = 1e-8,
    ci_level: float = 0.95,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Paired stratified bootstrap on (treated, control) blocks.

    Returns
    -------
    se : float
        Bootstrap standard error of the ATT.
    ci : np.ndarray
        Percentile CI at ``ci_level``, shape ``(2,)``.
    boot_atts : np.ndarray
        Full bootstrap distribution, shape ``(n_complete,)``.
    n_complete : int
        Number of bootstrap reps that converged (out of ``n_bootstrap``).
    """
    rng = np.random.default_rng(seed)
    n_T = X_T.shape[0]
    n_C = X_C.shape[0]

    atts = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        t_idx = rng.integers(0, n_T, size=n_T)
        c_idx = rng.integers(0, n_C, size=n_C)
        X_T_b = X_T[t_idx]
        X_C_b = X_C[c_idx]
        if Y_T.ndim == 1:
            Y_T_b = Y_T[t_idx]
            Y_C_b = Y_C[c_idx]
        else:
            Y_T_b = Y_T[t_idx, :]
            Y_C_b = Y_C[c_idx, :]
        atts[b] = _bootstrap_one_estimate(
            X_T_b, X_C_b, Y_T_b, Y_C_b,
            max_iter=max_iter, gtol=gtol,
        )

    boot_atts = atts[np.isfinite(atts)]
    n_complete = int(boot_atts.size)
    if n_complete < 2:
        return (
            float("nan"),
            np.asarray([float("nan"), float("nan")]),
            boot_atts,
            n_complete,
        )

    se = float(np.std(boot_atts, ddof=1))
    alpha = 1.0 - ci_level
    lo, hi = np.percentile(boot_atts, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return se, np.asarray([float(lo), float(hi)]), boot_atts, n_complete
