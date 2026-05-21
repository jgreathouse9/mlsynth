"""EM loop on pre-intervention data (Algorithm 2: ``EM_pre``).

Each outer iteration runs:

    1. Forward Kalman filter pass over ``Y_pre`` (Algorithm 4).
    2. Backward RTS smoother pass (Algorithm 6).
    3. Closed-form M-step update (Algorithm 7).

Optionally terminates early when the max-abs change in ``(A, H)`` falls
below ``em_tol``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .filtering import kalman_filter_pre
from .mstep import m_step
from .smoothing import rts_smoother
from .structures import (
    TASCFilteredStates,
    TASCParameters,
    TASCSmoothedStates,
)


def em_pre(
    Y_pre: np.ndarray,
    init_params: TASCParameters,
    n_em_iter: int,
    em_tol: Optional[float] = None,
    diagonal_Q: bool = True,
    diagonal_R: bool = True,
) -> Tuple[TASCParameters, np.ndarray, TASCFilteredStates, TASCSmoothedStates]:
    """Run ``EM_pre`` over the pre-treatment window.

    Parameters
    ----------
    Y_pre : np.ndarray
        Pre-treatment outcome matrix of shape ``(N, T0)``.
    init_params : TASCParameters
        Initial parameters ``theta^{(0)}``.
    n_em_iter : int
        Maximum number of EM iterations (``N_1`` in the paper).
    em_tol : float or None
        If not None, EM stops once the max absolute change in ``(A, H)`` is
        below this threshold.
    diagonal_Q, diagonal_R : bool
        Forwarded to ``m_step``.

    Returns
    -------
    params : TASCParameters
        Final parameter estimate.
    deltas : np.ndarray
        Max-abs change in ``(A, H)`` at each iteration (length equal to the
        number of iterations actually run).
    filtered : TASCFilteredStates
        Forward filtered states from the final iteration.
    smoothed : TASCSmoothedStates
        RTS smoothed states from the final iteration.
    """

    params = init_params
    deltas = []
    filtered: Optional[TASCFilteredStates] = None
    smoothed: Optional[TASCSmoothedStates] = None

    for _ in range(n_em_iter):
        filtered = kalman_filter_pre(Y_pre, params)
        smoothed = rts_smoother(filtered, params)
        new_params = m_step(
            Y_pre=Y_pre,
            smoothed=smoothed,
            prev_params=params,
            diagonal_Q=diagonal_Q,
            diagonal_R=diagonal_R,
        )

        delta = max(
            np.max(np.abs(new_params.A - params.A)),
            np.max(np.abs(new_params.H - params.H)),
        )
        deltas.append(delta)
        params = new_params

        if em_tol is not None and delta < em_tol:
            break

    # Refresh filtered / smoothed states under the final parameters so the
    # returned objects are consistent with what gets returned in TASCDesign.
    filtered = kalman_filter_pre(Y_pre, params)
    smoothed = rts_smoother(filtered, params)

    return params, np.asarray(deltas, dtype=float), filtered, smoothed
