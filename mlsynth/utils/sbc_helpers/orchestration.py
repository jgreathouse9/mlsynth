"""Top-level SBC procedure.

Steps 1-4 of Shi, Xi, Xie (2025), Section 3.1:

    1. Hamilton-filter every unit's pre-treatment series into
       (trend, cycle).
    2. Use the treated unit's pre-treatment lags + its own AR
       coefficients to extrapolate the post-treatment trend.
    3. Fit a simplex SCM on the donor cycles to impute the treated
       cycle over the post-treatment window.
    4. Combine: y_hat_{1, t}(0) = tau_hat_{1, t} + c_hat_{1, t}.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .hamilton import cycle_matrix_pre
from .structures import SBCDesign, SBCInputs
from .synthetic import solve_sbc_weights
from .trend_forecast import forecast_treated_trend


def solve_sbc(
    inputs: SBCInputs,
    *,
    h: int,
    p: int,
    weights_mode: str = "simplex",
) -> SBCDesign:
    """Run the four SBC steps end-to-end.

    Parameters
    ----------
    inputs : SBCInputs
        From ``prepare_sbc_inputs``.
    h, p : int
        Hamilton-filter horizon and lag count.
    weights_mode : {"simplex", "unrestricted"}
        See ``synthetic.solve_sbc_weights``.

    Returns
    -------
    SBCDesign
    """

    Y_full = inputs.Y_full
    T = inputs.T
    T0 = inputs.T0
    horizon = T - T0

    # --- Step 1: trend / cycle decomposition over the pre window ---------
    fits, cycles_pre = cycle_matrix_pre(Y_full, T0=T0, h=h, p=p)
    treated_fit = fits[0]
    donor_fits = fits[1:]

    # Restrict to the rows where the Hamilton filter produced a cycle.
    valid_mask = ~np.isnan(cycles_pre[:, 0])
    cycles_pre_valid = cycles_pre[valid_mask]
    if cycles_pre_valid.shape[0] < 2:
        from ...exceptions import MlsynthEstimationError
        raise MlsynthEstimationError(
            "SBC needs at least 2 effective pre-period observations after "
            "applying the Hamilton filter; got "
            f"{cycles_pre_valid.shape[0]}."
        )

    cycles_treated = cycles_pre_valid[:, 0]
    cycles_donors = cycles_pre_valid[:, 1:]

    # --- Step 3: SCM weights on cycles -----------------------------------
    weights, intercept = solve_sbc_weights(
        cycles_treated=cycles_treated,
        cycles_donors=cycles_donors,
        weights_mode=weights_mode,
    )

    # Pre-period in-sample cycle fit RMSE (diagnostic).
    pred_pre = cycles_donors @ weights
    if intercept is not None:
        pred_pre = pred_pre + intercept
    pre_cycle_rmse = float(np.sqrt(np.mean((cycles_treated - pred_pre) ** 2)))

    # --- Step 2: extrapolate treated trend over the post window ----------
    if horizon > 0:
        trend_forecast = forecast_treated_trend(
            y_target=inputs.y_target,
            treated_fit=treated_fit,
            T0=T0,
            horizon=horizon,
        )

        # --- Step 3 (cont'd): synthetic cycle over the post window ------
        # Apply each donor's Hamilton filter to its post-treatment history
        # by recomputing the in-sample lagged design at post times.
        cycle_forecast = np.zeros(horizon, dtype=float)
        for step in range(horizon):
            t = T0 + step  # zero-indexed
            donor_cycles_t = np.empty(len(donor_fits), dtype=float)
            for j, fit in enumerate(donor_fits):
                slope = fit.coefficients[1:]
                idx = np.array(
                    [t - fit.h - k for k in range(fit.p)], dtype=int
                )
                trend_jt = float(
                    fit.coefficients[0] + slope @ Y_full[idx, j + 1]
                )
                donor_cycles_t[j] = Y_full[t, j + 1] - trend_jt
            cycle_forecast[step] = float(donor_cycles_t @ weights)
            if intercept is not None:
                cycle_forecast[step] += intercept

        counterfactual_post = trend_forecast + cycle_forecast
    else:
        trend_forecast = np.empty(0, dtype=float)
        cycle_forecast = np.empty(0, dtype=float)
        counterfactual_post = np.empty(0, dtype=float)

    return SBCDesign(
        weights=weights,
        weights_mode=weights_mode,
        intercept=intercept,
        treated_hamilton=treated_fit,
        donor_hamiltons=donor_fits,
        trend_forecast=trend_forecast,
        cycle_forecast=cycle_forecast,
        counterfactual_post=counterfactual_post,
        pre_cycle_rmse=pre_cycle_rmse,
    )


def summarize_effects(
    inputs: SBCInputs, design: SBCDesign
):
    """Build the full-window counterfactual and the ATT.

    Returns
    -------
    att : float
        ``mean(y_target - counterfactual)`` over post-treatment periods,
        or ``np.nan`` when no post window exists.
    counterfactual_full : np.ndarray
        Length-``T`` series. Pre-treatment is the observed treated
        series (no counterfactual estimated there); post-treatment is
        ``design.counterfactual_post``.
    treatment_effect : np.ndarray
        Length-``T`` ``y_target - counterfactual_full``. Zero pre,
        non-trivial post.
    """

    T0 = inputs.T0
    T = inputs.T
    counterfactual_full = inputs.y_target.copy()
    if T > T0:
        counterfactual_full[T0:] = design.counterfactual_post
        att = float(np.mean(inputs.y_target[T0:] - design.counterfactual_post))
    else:
        att = float("nan")

    treatment_effect = inputs.y_target - counterfactual_full
    return att, counterfactual_full, treatment_effect
