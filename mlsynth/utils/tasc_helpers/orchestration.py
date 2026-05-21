"""Top-level TASC procedure (Algorithm 3).

Sequence:

    1. ``EM_pre`` on the pre-treatment data to learn ``theta``.
    2. Forward Kalman pass: Algorithm 4 on ``t = 1..T0`` and Algorithm 5 on
       ``t = T0 + 1..T``.
    3. Backward RTS smoother on the full window.
    4. Counterfactual + posterior CI computation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .em import em_pre
from .filtering import kalman_filter_full, kalman_filter_pre
from .inference import counterfactual_with_ci
from .setup import initialize_parameters
from .smoothing import rts_smoother
from .structures import (
    TASCDesign,
    TASCInference,
    TASCInputs,
)


def run_tasc(
    inputs: TASCInputs,
    d: int,
    n_em_iter: int,
    em_tol: Optional[float],
    diagonal_Q: bool,
    diagonal_R: bool,
    alpha: float,
    seed: Optional[int] = None,
):
    """Execute Algorithm 3 end-to-end.

    Returns
    -------
    design : TASCDesign
        Learned model with EM diagnostics and final filtered / smoothed states
        across all ``T`` periods.
    inference : TASCInference
        Counterfactual and posterior-based CIs.
    """

    init = initialize_parameters(inputs.Y_pre, d=d, seed=seed)
    params, deltas, _, _ = em_pre(
        Y_pre=inputs.Y_pre,
        init_params=init,
        n_em_iter=n_em_iter,
        em_tol=em_tol,
        diagonal_Q=diagonal_Q,
        diagonal_R=diagonal_R,
    )

    if inputs.Y_post_donors is not None:
        filtered_full = kalman_filter_full(
            Y_pre=inputs.Y_pre,
            Y_post_donors=inputs.Y_post_donors,
            params=params,
        )
    else:
        filtered_full = kalman_filter_pre(inputs.Y_pre, params)

    smoothed_full = rts_smoother(filtered_full, params)

    inference = counterfactual_with_ci(
        smoothed=smoothed_full,
        params=params,
        alpha=alpha,
    )

    design = TASCDesign(
        parameters=params,
        n_em_iter_used=len(deltas),
        em_param_deltas=deltas,
        filtered=filtered_full,
        smoothed=smoothed_full,
    )

    return design, inference


def summarize_effects(
    inputs: TASCInputs,
    inference: TASCInference,
) -> tuple[float, float]:
    """Compute ``ATT`` (post-period mean gap) and pre-period RMSE.

    Both are returned as plain ``float``\\s for storage on ``TASCResults``.
    """

    T0 = inputs.T0
    T = inputs.T
    y_obs = inputs.y_target
    y_hat = inference.counterfactual

    if T > T0:
        att = float(np.mean(y_obs[T0:] - y_hat[T0:]))
    else:
        att = float("nan")

    pre_resid = y_obs[:T0] - y_hat[:T0]
    pre_rmse = float(np.sqrt(np.mean(pre_resid ** 2)))
    return att, pre_rmse
