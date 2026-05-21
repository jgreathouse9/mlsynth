"""Block-diagonal hierarchical-aggregation penalty matrix for mlSC.

The mlSC objective contains the ridge-type penalty

    sum_{s, c} (omega_sc - v_sc * w_s)^2,    w_s = sum_c omega_sc.

For each aggregate block ``s`` with population weights ``v_s in R^{C_s}``
satisfying ``1^T v_s = 1``, the within-block penalty equals
``omega_s^T Q_s omega_s`` where

    Q_s = (I - v_s 1^T)^T (I - v_s 1^T)
        = I - v_s 1^T - 1 v_s^T + ||v_s||^2 * 1 1^T.

The full penalty matrix ``Q in R^{M x M}`` is block-diagonal with these
``Q_s`` blocks. ``Q_s`` is positive semidefinite (kernel: ``v_s``), so the
mlSC objective remains a convex QP.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag


def build_block(v_s: np.ndarray) -> np.ndarray:
    """Return the single-block penalty matrix ``Q_s`` for one aggregate."""
    v_s = np.asarray(v_s, dtype=float).reshape(-1)
    C_s = v_s.shape[0]
    ones = np.ones(C_s)
    norm_sq = float(v_s @ v_s)
    return (
        np.eye(C_s)
        - np.outer(v_s, ones)
        - np.outer(ones, v_s)
        + norm_sq * np.outer(ones, ones)
    )


def build_penalty_matrix(
    v_population: np.ndarray, disagg_to_agg: np.ndarray
) -> np.ndarray:
    """Assemble the full block-diagonal ``Q`` matrix.

    Parameters
    ----------
    v_population : np.ndarray
        Length-``M`` population weights for each disaggregate column.
    disagg_to_agg : np.ndarray
        Length-``M`` integer block-membership index for each column.

    Returns
    -------
    np.ndarray
        Block-diagonal ``Q`` of shape ``(M, M)``.
    """

    v_population = np.asarray(v_population, dtype=float).reshape(-1)
    disagg_to_agg = np.asarray(disagg_to_agg, dtype=int).reshape(-1)
    if v_population.shape[0] != disagg_to_agg.shape[0]:
        raise ValueError("v_population and disagg_to_agg must have equal length.")

    blocks = []
    for s in sorted(set(disagg_to_agg.tolist())):
        mask = disagg_to_agg == s
        blocks.append(build_block(v_population[mask]))
    return block_diag(*blocks)
