"""Kalman filter passes for TASC.

Implements Algorithm 4 (standard Kalman filter) and Algorithm 5 (Kalman
filter with infinite observation-noise variance on the target row) from
Rho, Illick, Narasipura, Abadie, Hsu, Misra (2026, arXiv:2601.03099).

The "infinite variance" trick (Sec 4.2) lets the post-treatment update use
only the donor rows to refine the latent state, while the target row's
contribution to the Kalman gain is zeroed out by Schur complement.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .structures import TASCFilteredStates, TASCParameters


def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetrize a square matrix to suppress numerical drift."""
    return 0.5 * (M + M.T)


def kalman_filter_step(
    y_k: np.ndarray,
    m_prev: np.ndarray,
    P_prev: np.ndarray,
    params: TASCParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single Kalman filter step (Algorithm 4)."""

    A = params.A
    H = params.H
    Q = params.Q
    R = params.R

    m_pred = A @ m_prev
    P_pred = _sym(A @ P_prev @ A.T + Q)

    v = y_k - H @ m_pred
    S = _sym(H @ P_pred @ H.T + R)
    # Solve S K^T = H P_pred^T   ->   K = P_pred H^T S^{-1}
    K = np.linalg.solve(S, H @ P_pred.T).T

    m_new = m_pred + K @ v
    P_new = _sym(P_pred - K @ S @ K.T)
    return m_new, P_new


def kalman_filter_inf_variance_step(
    y_donors_k: np.ndarray,
    m_prev: np.ndarray,
    P_prev: np.ndarray,
    params: TASCParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single Kalman filter step with ``R_{1,1} = inf`` (Algorithm 5).

    The target row is treated as missing. We partition

        H = [h_1^T; H_2],   R = diag(inf, R_2)

    so that the inverse innovation covariance has zero in the (1, 1) block by
    Schur complement, and only the donor block contributes to the update.

    Parameters
    ----------
    y_donors_k : np.ndarray
        Donor observations at time ``k``, shape ``(N - 1,)``.
    m_prev, P_prev : np.ndarray
        Previous filtered mean and covariance.
    params : TASCParameters
        Current model parameters.
    """

    A = params.A
    H = params.H
    Q = params.Q
    R = params.R

    H2 = H[1:, :]
    R2 = R[1:, 1:]

    m_pred = A @ m_prev
    P_pred = _sym(A @ P_prev @ A.T + Q)

    # Only the donor innovation contributes; the target row's residual is set
    # to zero (the augmented y has y_{1,k} <- h_1^T m_pred, see Algorithm 5).
    v_donors = y_donors_k - H2 @ m_pred

    S2 = _sym(H2 @ P_pred @ H2.T + R2)
    K2 = np.linalg.solve(S2, H2 @ P_pred.T).T  # (d, N - 1)

    m_new = m_pred + K2 @ v_donors
    P_new = _sym(P_pred - K2 @ S2 @ K2.T)
    return m_new, P_new


def kalman_filter_pre(
    Y_pre: np.ndarray,
    params: TASCParameters,
) -> TASCFilteredStates:
    """Run Algorithm 4 across the pre-treatment window.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(N, T0)``.
    params : TASCParameters
        Current model parameters.
    """

    N, T0 = Y_pre.shape
    d = params.A.shape[0]

    m = np.zeros((T0 + 1, d))
    P = np.zeros((T0 + 1, d, d))
    m[0] = params.m0
    P[0] = params.P0

    for k in range(1, T0 + 1):
        m[k], P[k] = kalman_filter_step(
            Y_pre[:, k - 1], m[k - 1], P[k - 1], params
        )

    return TASCFilteredStates(m=m, P=P)


def kalman_filter_full(
    Y_pre: np.ndarray,
    Y_post_donors: np.ndarray,
    params: TASCParameters,
) -> TASCFilteredStates:
    """Forward pass over all ``T`` periods (Algorithm 3, lines 1-3).

    The first ``T0`` updates use Algorithm 4; the remaining
    ``T - T0`` updates use Algorithm 5 with the target's observation variance
    set to infinity.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment slice of shape ``(N, T0)``.
    Y_post_donors : np.ndarray
        Post-treatment donor-only slice of shape ``(N - 1, T - T0)``.
    params : TASCParameters
        EM-learned model parameters.
    """

    N, T0 = Y_pre.shape
    T_post = Y_post_donors.shape[1] if Y_post_donors is not None else 0
    T = T0 + T_post
    d = params.A.shape[0]

    m = np.zeros((T + 1, d))
    P = np.zeros((T + 1, d, d))
    m[0] = params.m0
    P[0] = params.P0

    for k in range(1, T0 + 1):
        m[k], P[k] = kalman_filter_step(
            Y_pre[:, k - 1], m[k - 1], P[k - 1], params
        )

    for j in range(T_post):
        k = T0 + 1 + j
        m[k], P[k] = kalman_filter_inf_variance_step(
            Y_post_donors[:, j], m[k - 1], P[k - 1], params
        )

    return TASCFilteredStates(m=m, P=P)
