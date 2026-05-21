"""Closed-form M-step for TASC (Algorithm 7).

Given the smoothed sufficient statistics, returns new ``theta'`` that
maximizes the expected complete-data log-likelihood. ``Q`` and ``R`` may be
constrained to be diagonal (the paper's default) via the corresponding
flags.
"""

from __future__ import annotations

import numpy as np

from .filtering import _sym
from .structures import TASCParameters, TASCSmoothedStates


def _maybe_diag(M: np.ndarray, diagonal: bool) -> np.ndarray:
    """Return ``Diag(M)`` if ``diagonal`` else the symmetrized matrix."""
    if diagonal:
        return np.diag(np.maximum(np.diag(M), 1e-10))
    return _sym(M)


def m_step(
    Y_pre: np.ndarray,
    smoothed: TASCSmoothedStates,
    prev_params: TASCParameters,
    diagonal_Q: bool = True,
    diagonal_R: bool = True,
) -> TASCParameters:
    """Maximum-likelihood parameter update (Algorithm 7).

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcomes, shape ``(N, T)``. The number of columns must
        equal the smoother's ``T``.
    smoothed : TASCSmoothedStates
        Output of the RTS smoother. Indices 0..T-1 supply ``m_{k-1}^s`` and
        ``G_{k-1}``; indices 1..T supply ``m_k^s``.
    prev_params : TASCParameters
        Current parameters, used to seed ``m_0`` and ``P_0`` updates.
    diagonal_Q, diagonal_R : bool
        If True (default), the corresponding covariance update is restricted
        to its diagonal as in the paper.
    """

    N, T = Y_pre.shape
    m_s = smoothed.m_s
    P_s = smoothed.P_s
    G = smoothed.G

    # Sigma = (1/T) sum_{k=1..T} (P_k^s + m_k^s m_k^{s T})
    Sigma = np.zeros_like(P_s[0])
    Phi = np.zeros_like(P_s[0])
    C = np.zeros_like(P_s[0])
    B = np.zeros((N, m_s.shape[1]))
    D = np.zeros((N, N))

    for k in range(1, T + 1):
        Sigma += P_s[k] + np.outer(m_s[k], m_s[k])
        Phi += P_s[k - 1] + np.outer(m_s[k - 1], m_s[k - 1])
        C += P_s[k] @ G[k - 1].T + np.outer(m_s[k], m_s[k - 1])
        B += np.outer(Y_pre[:, k - 1], m_s[k])
        D += np.outer(Y_pre[:, k - 1], Y_pre[:, k - 1])

    inv_T = 1.0 / T
    Sigma *= inv_T
    Phi *= inv_T
    C *= inv_T
    B *= inv_T
    D *= inv_T

    d = m_s.shape[1]
    ridge_d = 1e-10 * np.eye(d)

    A_new = np.linalg.solve(Phi + ridge_d, C.T).T  # A = C Phi^{-1}
    H_new = np.linalg.solve(Sigma + ridge_d, B.T).T  # H = B Sigma^{-1}

    Q_raw = Sigma - C @ A_new.T - A_new @ C.T + A_new @ Phi @ A_new.T
    R_raw = D - B @ H_new.T - H_new @ B.T + H_new @ Sigma @ H_new.T

    Q_new = _maybe_diag(Q_raw, diagonal_Q)
    R_new = _maybe_diag(R_raw, diagonal_R)

    m0_new = m_s[0].copy()
    diff = m_s[0] - prev_params.m0
    P0_new = _sym(P_s[0] + np.outer(diff, diff))

    return TASCParameters(
        A=A_new,
        H=H_new,
        Q=Q_new,
        R=R_new,
        m0=m0_new,
        P0=P0_new,
    )
