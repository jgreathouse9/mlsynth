"""Rauch-Tung-Striebel smoother for TASC (Algorithm 6).

Operates on the output of the forward Kalman pass and produces smoothed
state estimates plus the smoother-gain matrices ``G_k`` required for the
M-step (Algorithm 7 computes ``C`` from ``G_{k-1}``).
"""

from __future__ import annotations

import numpy as np

from .filtering import _sym
from .structures import TASCFilteredStates, TASCParameters, TASCSmoothedStates


def rts_smoother(
    filtered: TASCFilteredStates,
    params: TASCParameters,
) -> TASCSmoothedStates:
    """Backward smoothing pass.

    Parameters
    ----------
    filtered : TASCFilteredStates
        Output of ``kalman_filter_pre`` or ``kalman_filter_full``.
        Index 0 holds the prior; indices 1..T hold filtered posteriors.
    params : TASCParameters
        Model parameters used in the forward pass.

    Returns
    -------
    TASCSmoothedStates
        Smoothed means and covariances at indices 0..T, plus the smoother
        gains ``G_k`` at indices 0..T-1. ``G[T]`` is zero-filled and unused.
    """

    A = params.A
    Q = params.Q

    m_f = filtered.m
    P_f = filtered.P
    T_plus_1, d = m_f.shape
    T = T_plus_1 - 1

    m_s = np.zeros_like(m_f)
    P_s = np.zeros_like(P_f)
    G = np.zeros_like(P_f)

    m_s[T] = m_f[T]
    P_s[T] = P_f[T]

    for k in range(T - 1, -1, -1):
        m_pred = A @ m_f[k]
        P_pred = _sym(A @ P_f[k] @ A.T + Q)
        # G_k = P_f[k] A^T P_pred^{-1}  ->  solve P_pred^T G_k^T = (P_f[k] A^T)^T = A P_f[k]^T
        G_k = np.linalg.solve(P_pred, A @ P_f[k].T).T
        G[k] = G_k

        m_s[k] = m_f[k] + G_k @ (m_s[k + 1] - m_pred)
        P_s[k] = _sym(P_f[k] + G_k @ (P_s[k + 1] - P_pred) @ G_k.T)

    return TASCSmoothedStates(m_s=m_s, P_s=P_s, G=G)
