"""Imbalance measures from Ben-Michael, Feller & Rothstein (2022, Eq. 5).

Given stacked pre-treatment outcomes (treated and donors aligned per
treated unit) and a weight matrix ``Gamma`` of shape ``(N, J)``,
``compute_q_sep`` returns the per-unit imbalance and ``compute_q_pool``
returns the pooled imbalance.

Definitions (paper Eq. 5 with all ``L_j = L`` common):

    q_sep(Gamma)^2  = (1/J) sum_j (1/L) || y_j^pre - Y_donors_j^pre @ gamma_j ||^2
    q_pool(Gamma)^2 = (1/L) || (1/J) sum_j (y_j^pre - Y_donors_j^pre @ gamma_j) ||^2
"""

from __future__ import annotations

import numpy as np


def residuals(Y_treated_pre: np.ndarray, Y_donors_pre: np.ndarray,
              Gamma: np.ndarray) -> np.ndarray:
    """Return per-(time, treated-unit) residuals, shape ``(L, J)``.

    ``Y_donors_pre`` has shape ``(L, N, J)`` and ``Gamma`` has shape
    ``(N, J)``. The residual for treated unit ``j`` at pre-period
    offset ``l`` is ``y_j[l] - sum_i gamma_ij * Y_donors_j[l, i]``.
    """
    # Contract the donor axis: Y_donors_pre[l, i, j] * Gamma[i, j] summed over i.
    fitted = np.einsum("lij,ij->lj", Y_donors_pre, Gamma)
    return Y_treated_pre - fitted


def compute_q_sep(Y_treated_pre: np.ndarray, Y_donors_pre: np.ndarray,
                  Gamma: np.ndarray) -> float:
    """Per-unit pre-treatment imbalance ``q_sep(Gamma)``."""
    res = residuals(Y_treated_pre, Y_donors_pre, Gamma)
    L, J = res.shape
    # ||res_j||^2 / L for each j, then average over j and square-root.
    per_unit_msq = (res ** 2).sum(axis=0) / L
    return float(np.sqrt(per_unit_msq.mean()))


def compute_q_pool(Y_treated_pre: np.ndarray, Y_donors_pre: np.ndarray,
                   Gamma: np.ndarray) -> float:
    """Pooled (avg-treated) pre-treatment imbalance ``q_pool(Gamma)``."""
    res = residuals(Y_treated_pre, Y_donors_pre, Gamma)
    L = res.shape[0]
    avg_residual = res.mean(axis=1)  # (L,)
    return float(np.sqrt((avg_residual ** 2).sum() / L))


def normalize_imbalances(
    q_sep: float, q_pool: float, q_sep_base: float, q_pool_base: float,
) -> tuple[float, float]:
    """Return ``(q_tilde_sep, q_tilde_pool)`` normalized by the
    separate-SCM baseline. Guards against zero baselines (returns the
    raw imbalance unchanged in that degenerate case).
    """
    q_tilde_sep = q_sep / q_sep_base if q_sep_base > 0 else q_sep
    q_tilde_pool = q_pool / q_pool_base if q_pool_base > 0 else q_pool
    return q_tilde_sep, q_tilde_pool
