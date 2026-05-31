"""Cao-Dowd spillover-adjusted treatment-effect estimator (eq. 5).

Given the leave-one-out SCM artifacts ``(a, B)`` from
:mod:`.scm_core` and a spillover-structure matrix ``A``, the
per-post-period parameter estimate is

    gamma_hat(T+1) = (A' M A)^{-1} A' (I - B)' [(I - B) Y_{T+1} - a]

with ``M = (I - B)' (I - B)``. The full effect vector is then
``alpha_hat = A @ gamma_hat``; its first entry is the
spillover-adjusted treatment effect on the treated unit, and entries
2..p+1 are the per-affected-unit spillover effects.

This module also provides a thin :func:`vanilla_scm_path` helper that
returns the standard SCM counterfactual using the treated unit's own
leave-one-out fit, so the orchestrator can produce the SCM-vs-SP
comparison without re-fitting.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


_M_RIDGE = 1e-8


def build_M(B: np.ndarray) -> np.ndarray:
    """Gram matrix ``M = (I - B)' (I - B) + 1e-8 * I``."""
    N = B.shape[0]
    I_B = np.eye(N) - B
    return I_B.T @ I_B + np.eye(N) * _M_RIDGE


def sp_estimate(
    Y_post: np.ndarray,
    *,
    a: np.ndarray,
    B: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute per-period spillover-adjusted effects.

    Parameters
    ----------
    Y_post : np.ndarray
        Shape ``(N, T1)`` post-treatment outcomes.
    a : np.ndarray
        Length-``N`` leave-one-out intercepts.
    B : np.ndarray
        Shape ``(N, N)`` leave-one-out SCM weight matrix.
    M : np.ndarray
        Shape ``(N, N)`` Gram matrix from :func:`build_M`.
    A : np.ndarray
        Shape ``(N, k)`` spillover-structure matrix.

    Returns
    -------
    gamma : np.ndarray
        Shape ``(k, T1)`` parameter estimates per post-period.
    alpha : np.ndarray
        Shape ``(N, T1)`` effect estimates per post-period.
    cond_AMA : float
        Condition number of ``A' M A`` (Assumption 1(d) diagnostic).
    """
    N, T1 = Y_post.shape
    I_B = np.eye(N) - B
    AMA = A.T @ M @ A
    cond_AMA = float(np.linalg.cond(AMA))
    AMA_inv = np.linalg.inv(AMA)
    # Each column of Y_post is one Y_{T+1}; vectorise across periods.
    residual = I_B @ Y_post - a[:, None]                # (N, T1)
    gamma = AMA_inv @ (A.T @ I_B.T @ residual)           # (k, T1)
    alpha = A @ gamma                                    # (N, T1)
    return gamma, alpha, cond_AMA


def vanilla_scm_path(
    Y_post: np.ndarray, *, a: np.ndarray, B: np.ndarray,
) -> np.ndarray:
    """Vanilla (no-spillover) SCM counterfactual for the treated unit.

    Returns a length-``T1`` vector ``a[0] + B[0, :] @ Y_post`` —
    the treated unit's own leave-one-out fit applied to the post-period
    outcomes.
    """
    return a[0] + B[0] @ Y_post
